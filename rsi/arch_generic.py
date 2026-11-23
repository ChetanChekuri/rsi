"""Models other than Qwen3.5: a layer-streaming forward pass for the standard decoder family (llama, mistral, qwen2,
qwen3, granite, ...: RMSNorm, rotary attention with grouped KV heads, gated MLP), a tokenizer read straight from the
GGUF, and a way to open the right one for any model file.

    S = open_stream(gguf_path)        # Qwen35Stream for qwen35, GenericStream for the llama family
    check_stream(S, model_tag)        # does it predict the same next token as ollama? (verify before trusting it)
"""
import json, os, re, urllib.request
import numpy as np, torch, torch.nn.functional as F
from .arch_qwen35 import Qwen35Stream
from .gguf_io import GGUF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEOX = {"qwen2", "qwen3", "qwen2moe", "phi3", "stablelm", "gptneox", "olmo2", "exaone"}   # rotate halves; others rotate pairs
SPLIT = {   # pre-tokenizer regexes (tokenizer.ggml.pre)
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "llama-bpe": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "default": r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
}


def field(g, key, default=None):
    f = g.r.fields.get(key)
    return default if f is None else f.contents()


def gguf_tokenizer(path):
    """a `tokenizers` Tokenizer built from the GGUF's own vocabulary (byte-level BPE models: Qwen, Llama 3, Mistral
    Nemo, ...). SentencePiece ('llama') vocabularies are not supported here."""
    from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders
    g = GGUF(path)
    kind = field(g, "tokenizer.ggml.model")
    if kind != "gpt2":
        raise NotImplementedError(f"tokenizer type {kind!r}: only byte-level BPE ('gpt2') vocabularies are built from the GGUF")
    toks = field(g, "tokenizer.ggml.tokens"); merges = field(g, "tokenizer.ggml.merges") or []
    types = field(g, "tokenizer.ggml.token_type") or [1] * len(toks)
    pre = field(g, "tokenizer.ggml.pre", "default")
    pat = SPLIT.get(pre) or (SPLIT["qwen2"] if "qwen" in str(pre) else SPLIT["llama-bpe"] if "llama" in str(pre) else SPLIT["default"])
    tk = Tokenizer(models.BPE({t: i for i, t in enumerate(toks)}, [tuple(m.split(" ", 1)) for m in merges], ignore_merges=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(pat), behavior="isolated"),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([t for t, ty in zip(toks, types) if ty in (3, 4)])   # control and user-defined tokens
    return tk


class GenericStream(Qwen35Stream):
    """Layer-streaming forward pass for standard decoder-only transformers. Same interface as Qwen35Stream
    (run, enc, capture, score) and the same edit hooks."""

    def __init__(self, gguf_path, tokenizer=None, device="auto", base_dir="", lm_cache_gb=3.0):
        self.g = GGUF(gguf_path); m = self.g.meta; self.arch = self.g.arch
        self.L = int(m("block_count")); self.D = int(m("embedding_length"))
        self.H = int(m("attention.head_count")); kv = m("attention.head_count_kv", self.H)
        self.KV = int(kv if np.ndim(kv) == 0 else kv[0])
        self.hd = int(m("attention.key_length", self.D // self.H))
        self.rd = int(m("rope.dimension_count", self.hd)); self.theta = float(m("rope.freq_base", 10000.0))
        self.eps = float(m("attention.layer_norm_rms_epsilon", 1e-5)); self.neox = self.arch in NEOX
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device); self.wdt = torch.float16 if self.dev.type == "cuda" else torch.float32
        self.tok = tokenizer if tokenizer is not None else gguf_tokenizer(gguf_path)
        self.vgroup, self.base_dir, self.lm_cache_gb = "interleave", base_dir, lm_cache_gb
        self.inv = 1.0 / (self.theta ** (torch.arange(0, self.rd, 2, dtype=torch.float64, device=self.dev) / self.rd))
        self.lm_name = "output.weight" if "output.weight" in self.g.T else "token_embd.weight"
        self._lm = None; self.rec = [False] * self.L
        if any(n.endswith("ffn_gate_exps.weight") for n in self.g.T):
            raise NotImplementedError("mixture-of-experts layers are not implemented in GenericStream")

    def rope(self, x):
        if self.neox:
            return super().rope(x)
        T = x.shape[1]; d = self.rd
        ang = torch.outer(torch.arange(T, device=self.dev, dtype=torch.float64), self.inv)
        cos, sin = ang.cos().float()[None, :, None], ang.sin().float()[None, :, None]
        xr, xp = x[..., :d], x[..., d:]
        a, b = xr[..., 0::2], xr[..., 1::2]
        out = torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)
        return torch.cat([out, xp], -1)

    def attention(self, l, x, W):
        p = f"blk.{l}."; B, T, _ = x.shape; H, KV, hd = self.H, self.KV, self.hd
        def proj(n, heads):
            y = self.lin(x, W[p + n + ".weight"])
            if p + n + ".bias" in W:
                y = y + W[p + n + ".bias"]
            return y.reshape(B, T, heads, hd)
        q, k, v = proj("attn_q", H), proj("attn_k", KV), proj("attn_v", KV)
        if p + "attn_q_norm.weight" in W:
            q, k = self.rms(q, W[p + "attn_q_norm.weight"]), self.rms(k, W[p + "attn_k_norm.weight"])
        q, k = self.rope(q), self.rope(k)
        k = k.repeat_interleave(H // KV, 2); v = v.repeat_interleave(H // KV, 2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        o = self.lin(o.reshape(B, T, -1), W[p + "attn_output.weight"])
        return o + W[p + "attn_output.bias"] if p + "attn_output.bias" in W else o

    def block(self, l, x, W):
        p = f"blk.{l}."
        x = x + self.attention(l, self.rms(x, W[p + "attn_norm.weight"]), W)
        h = self.rms(x, W[p + "ffn_norm.weight"])
        return x + self.lin(F.silu(self.lin(h, W[p + "ffn_gate.weight"])) * self.lin(h, W[p + "ffn_up.weight"]), W[p + "ffn_down.weight"])


def open_stream(gguf_path, device="auto"):
    """the right streaming forward pass for a model file"""
    arch = GGUF(gguf_path).arch
    if arch == "qwen35":   # its own forward pass (gated delta-net layers); the tokenizer ships with rsi
        return Qwen35Stream(gguf_path, os.path.join(ROOT, "data", "qwen35_tokenizer.json"), device)
    return GenericStream(gguf_path, device=device)


def check_stream(S, model_tag, prompts=("The capital of France is", "def add(a, b):\n    return", "1, 2, 3, 4,"),
                 host=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")):
    """Does the forward pass agree with ollama? For each prompt: the stream's most likely next token vs the token
    ollama generates at temperature 0 (raw prompt, no template). Returns (agreed, total, details)."""
    res = []
    for p in prompts:
        ids = S.enc(p)   # one placeholder token after the prompt: its "prediction" is the stream's next-token guess
        out = S.run([ids + [ids[-1]]], score=[{"argmax": (len(ids), len(ids) + 1)}])[0]["argmax"][0]
        mine = S.tok.decode([int(out[-1])])
        body = {"model": model_tag, "prompt": p, "raw": True, "stream": False, "options": {"temperature": 0, "num_predict": 1, "seed": 0}}
        host_url = host if host.startswith("http") else "http://" + host
        r = json.loads(urllib.request.urlopen(urllib.request.Request(host_url + "/api/generate", json.dumps(body).encode(),
                                                                     {"Content-Type": "application/json"}), timeout=600).read())
        theirs = r.get("response", "")
        res.append({"prompt": p, "stream": mine, "ollama": theirs, "agree": mine.strip() == theirs.strip() and mine != ""})
    return sum(r["agree"] for r in res), len(res), res
