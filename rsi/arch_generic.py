"""Models other than Qwen3.5: a layer-streaming forward pass for the standard decoder family (GGUF architectures
llama, qwen2, qwen3: Llama 2/3.x, Mistral and Mixtral-free llama-arch models, Qwen2/2.5, Qwen3, ...: RMSNorm, rotary
attention with grouped KV heads and optional biases / q-k norms, gated SiLU MLP), tokenizers read straight from the
GGUF (byte-level BPE and SentencePiece), and a way to open the right one for any model file. Architectures whose
maths differ (Gemma, Phi-3, Granite, OLMo2, mixture-of-experts, ...) are refused with a clear error rather than run
wrongly: subclass GenericStream for them and confirm with check_stream.

    S = open_stream(gguf_path)        # Qwen35Stream for qwen35, GenericStream for the llama family
    ids = S.bos + S.enc(text)         # S.enc never adds BOS; models like Llama 3 need it (S.bos) or they go off track
    check_stream(S, model_tag)        # does it predict the same next token as ollama? (verify before trusting it)
"""
import heapq, json, os, re, urllib.request
import numpy as np, torch, torch.nn.functional as F
from .arch_qwen35 import Qwen35Stream
from .gguf_io import GGUF
from . import llm, network

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


class _Enc:
    def __init__(self, ids, offsets):
        self.ids, self.offsets = ids, offsets


class SPMTokenizer:
    """SentencePiece vocabularies (GGUF tokenizer model 'llama': Llama 2, Mistral 7B v0.1-0.3, TinyLlama, ...), encoded
    the way llama.cpp does it: split on special tokens, a space prefix at the start and after each special token, spaces
    as '▁', then repeatedly merge the adjacent pair whose concatenation is a vocabulary entry with the highest score;
    pieces not in the vocabulary fall back to byte tokens <0xNN>. Same interface as a `tokenizers` Tokenizer for what
    the streams use (encode(...).ids / .offsets, decode, get_vocab)."""

    def __init__(self, g):
        self.tokens = list(field(g, "tokenizer.ggml.tokens")); n = len(self.tokens)
        self.scores = list(field(g, "tokenizer.ggml.scores") or [0.0] * n)
        self.types = list(field(g, "tokenizer.ggml.token_type") or [1] * n)
        self.index = {}
        for i, t in enumerate(self.tokens):
            self.index.setdefault(t, i)
        self.space = bool(field(g, "tokenizer.ggml.add_space_prefix", True))
        self.unk = int(field(g, "tokenizer.ggml.unknown_token_id", 0))
        self.byte = {b: self.index.get(f"<0x{b:02X}>", self.unk) for b in range(256)}
        sp = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (2, 3, 4) and t), key=len, reverse=True)
        self.special = re.compile("|".join(re.escape(t) for t in sp)) if sp else None

    def get_vocab(self):
        return dict(self.index)

    def _merge(self, text):
        if not text:
            return []
        n = len(text); start = list(range(n)); end = [i + 1 for i in range(n)]; nxt = list(range(1, n)) + [-1]; prv = list(range(-1, n - 1))
        alive = [True] * n; heap = []

        def push(l, r):
            i = self.index.get(text[start[l]:end[r]])
            if i is not None:
                heapq.heappush(heap, (-self.scores[i], l, r, end[r]))
        for i in range(n - 1):
            push(i, i + 1)
        while heap:
            _, l, r, e = heapq.heappop(heap)
            if not (alive[l] and alive[r] and nxt[l] == r and end[r] == e):
                continue   # stale: one side changed since this pair was queued
            end[l] = end[r]; alive[r] = False; nxt[l] = nxt[r]
            if nxt[r] != -1:
                prv[nxt[r]] = l
            if prv[l] != -1:
                push(prv[l], l)
            if nxt[l] != -1:
                push(l, nxt[l])
        out, i = [], 0
        while i != -1:
            piece = text[start[i]:end[i]]; tid = self.index.get(piece)
            if tid is not None:
                out.append((tid, start[i], end[i]))
            else:
                out += [(self.byte[b], start[i], end[i]) for b in piece.encode("utf-8")]
            i = nxt[i]
        return out

    def encode(self, text, add_special_tokens=False):
        frags, pos = [], 0
        for m in (self.special.finditer(text) if self.special else []):
            frags += [(False, text[pos:m.start()], pos), (True, m.group(), m.start())]; pos = m.end()
        frags.append((False, text[pos:], pos))
        ids, offs, prev_special = [], [], True
        for special, piece, at in frags:
            if special:
                ids.append(self.index[piece]); offs.append((at, at + len(piece))); prev_special = True; continue
            if not piece:
                continue
            pre = " " if self.space and prev_special else ""
            for tid, a, b in self._merge((pre + piece).replace(" ", "\u2581")):
                ids.append(tid); offs.append((at + max(a - len(pre), 0), at + max(b - len(pre), 0)))
            prev_special = False
        return _Enc(ids, offs)

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            t, ty = self.tokens[int(i)], self.types[int(i)]
            if ty == 6 or re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t):
                out += bytes([int(t[3:5], 16)])
            elif ty != 3:   # control tokens print as nothing
                out += t.replace("\u2581", " ").encode("utf-8")
        return out.decode("utf-8", errors="replace")


def gguf_tokenizer(path):
    """the model's own tokenizer, built from its GGUF: byte-level BPE ('gpt2': Qwen, Llama 3, Mistral Nemo, ...) or
    SentencePiece ('llama': Llama 2, Mistral 7B, TinyLlama, ...)"""
    from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders
    g = GGUF(path)
    kind = field(g, "tokenizer.ggml.model")
    if kind == "llama":
        return SPMTokenizer(g)
    if kind != "gpt2":
        raise NotImplementedError(f"tokenizer type {kind!r}: built-in support covers byte-level BPE ('gpt2') and SentencePiece ('llama')")
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

    SUPPORTED = {"llama", "qwen2", "qwen3"}
    LAYER_TENSORS = {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q.bias", "attn_k.bias",
                     "attn_v.bias", "attn_output.bias", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight", "ffn_gate.weight",
                     "ffn_up.weight", "ffn_down.weight"}

    def __init__(self, gguf_path, tokenizer=None, device="auto", base_dir="", lm_cache_gb=3.0, force=False):
        self.g = GGUF(gguf_path); m = self.g.meta; self.arch = self.g.arch
        layer0 = {n[len("blk.0."):] for n in self.g.T if n.startswith("blk.0.")}
        unknown = sorted(layer0 - self.LAYER_TENSORS); missing = sorted({"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                                                         "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"} - layer0)
        scaling = str(m("rope.scaling.type", "none") or "none")
        if not force and (self.arch not in self.SUPPORTED or unknown or missing or scaling not in ("none", "linear")):
            raise NotImplementedError(
                f"GenericStream implements the llama / qwen2 / qwen3 layer maths; this GGUF is {self.arch!r}"
                + (f" with layer tensors it doesn't know ({', '.join(unknown)})" if unknown else "")
                + (f", missing {', '.join(missing)}" if missing else "") + (f", rope scaling {scaling!r}" if scaling not in ("none", "linear") else "")
                + ". Subclass GenericStream (override block/attention) for this architecture, or pass force=True if you know the maths "
                  "match, and confirm with check_stream().")
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
        add_bos, bos_id = field(self.g, "tokenizer.ggml.add_bos_token"), field(self.g, "tokenizer.ggml.bos_token_id")
        if add_bos is None:   # llama.cpp's default: BPE vocabularies get a BOS unless they say otherwise (Qwen: no)
            add_bos = bos_id is not None and "qwen" not in str(field(self.g, "tokenizer.ggml.pre", ""))
        self.bos = [int(bos_id)] if add_bos and bos_id is not None else []   # prefix S.bos + S.enc(text) for a document start
        self.vgroup, self.base_dir, self.lm_cache_gb = "interleave", base_dir, lm_cache_gb
        self.inv = 1.0 / (self.theta ** (torch.arange(0, self.rd, 2, dtype=torch.float64, device=self.dev) / self.rd))
        if scaling == "linear":   # positions compressed by a constant factor
            self.inv = self.inv / float(m("rope.scaling.factor", 1.0) or 1.0)
        if "rope_freqs.weight" in self.g.T:   # per-frequency divisors (Llama 3.1+ long-context scaling)
            self.inv = self.inv / torch.tensor(np.array(self.g.deq("rope_freqs.weight")), dtype=torch.float64, device=self.dev).flatten()[: self.rd // 2]
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


def open_stream(gguf_path, device="auto", force=False):
    """the right streaming forward pass for a model file (raises NotImplementedError, saying why, for architectures
    neither implements: write your own subclass for those)"""
    g = GGUF(gguf_path); arch = g.arch
    if arch == "qwen35":   # its own forward pass (gated delta-net layers); the tokenizer ships with rsi
        if any(n.endswith("_exps.weight") for n in g.T) and not force:
            raise NotImplementedError("Qwen35Stream implements the dense Qwen3.5 layers; this is a mixture-of-experts model")
        return Qwen35Stream(gguf_path, os.path.join(ROOT, "data", "qwen35_tokenizer.json"), device)
    return GenericStream(gguf_path, device=device, force=force)


def check_stream(S, model_tag, prompts=("The capital of France is", "def add(a, b):\n    return", "The opposite of hot is",
                                        "The answer to 7 times 8 is"),
                 host=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")):
    """Does the forward pass agree with ollama? For each prompt: the stream's most likely next token vs the token
    ollama generates at temperature 0 (raw prompt, no template). Returns (agreed, total, details). A rare mismatch can be
    a near-tie that ollama's quantized arithmetic flips: compare the two tokens' log-probs before suspecting a bug.
    Ollama's raw mode can also handle some prompts differently (on Qwen3.5, some prompts ending in a newline + text), so
    keep the test prompts plain."""
    res = []
    for p in prompts:
        ids = getattr(S, "bos", []) + S.enc(p)   # ollama adds the model's BOS to raw prompts too; one placeholder token after the prompt: its "prediction" is the stream's next-token guess
        out = S.run([ids + [ids[-1]]], score=[{"argmax": (len(ids), len(ids) + 1)}])[0]["argmax"][0]
        mine = S.tok.decode([int(out[-1])])
        body = {"model": model_tag, "prompt": p, "raw": True, "stream": False, "options": {"temperature": 0, "num_predict": 1, "seed": 0}}
        host_url = network.local_url(host)
        llm.model_caps({"ollama_url": host_url}, model_tag, require_local=True)
        r = json.loads(network.open_local(urllib.request.Request(host_url + "/api/generate", json.dumps(body).encode(),
                                                                     {"Content-Type": "application/json"}), timeout=600).read())
        theirs = r.get("response", "")
        res.append({"prompt": p, "stream": mine, "ollama": theirs, "agree": mine == theirs and mine != ""})
    return sum(r["agree"] for r in res), len(res), res
