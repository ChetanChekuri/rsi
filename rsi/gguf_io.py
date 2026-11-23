"""GGUF access for any model ollama runs (in-place Q8_0 writes validated byte-exact on Qwen3.5 and Llama 3.2).

- metadata + per-tensor dequantization for any ggml type gguf-py understands (Q8_0, K-quants, F16, BF16, ...)
- baking a list of edit specs into a copy of the master GGUF *in place* (byte layout unchanged), which requires the
  edited tensors to be Q8_0 / F16 / BF16 / F32. Use a Q8_0 master (a `-q8_0` tag) for large models.
- deriving a smaller run quantization with llama.cpp's llama-quantize (bundled with ollama)
- registering a GGUF with ollama, reusing the base model's Modelfile (projector, renderer, parser, parameters)
"""
import os, re, shutil, subprocess, tempfile
import numpy as np, gguf
from gguf import quants

INPLACE = {"Q8_0", "F16", "BF16", "F32"}


def q8_0(W):
    """Q8_0 quantization identical to llama.cpp for already-representable values (validated byte-exact on qwen35)."""
    x = W.reshape(W.shape[0], -1, 32).astype(np.float32); s = np.abs(x).max(-1, keepdims=True) / 127.0
    inv = np.divide(1.0, s, out=np.zeros_like(s), where=s != 0)
    q = np.clip(np.rint(x * inv), -127, 127).astype(np.int8)
    o = np.empty((W.shape[0], x.shape[1], 34), np.uint8)
    o[..., :2] = s.astype(np.float16).view(np.uint8); o[..., 2:] = q.view(np.uint8)
    return o.reshape(W.shape[0], -1)


class GGUF:
    def __init__(self, path, mode="r"):
        self.path = path
        self.r = gguf.GGUFReader(path, mode)
        self.arch = self.r.fields["general.architecture"].contents()
        self.T = {t.name: t for t in self.r.tensors}

    def meta(self, key, default=None):
        f = self.r.fields.get(f"{self.arch}.{key}")
        return default if f is None else f.contents()

    def shape(self, name):
        """logical numpy shape: (rows = output dim, cols = input dim) for matrices"""
        return tuple(int(x) for x in reversed(self.T[name].shape))

    def deq(self, name, rows=None):
        t = self.T[name]; data = t.data if rows is None else t.data[rows]
        tn = t.tensor_type.name
        if tn == "F32":
            out = np.asarray(data, dtype=np.float32)
        elif tn == "F16":
            out = np.asarray(data).astype(np.float32)
        else:
            out = quants.dequantize(np.asarray(data), t.tensor_type).astype(np.float32)
        if rows is None:
            return out.reshape(self.shape(name))
        return out.reshape(len(np.asarray(rows).reshape(-1)), *self.shape(name)[1:])

    def write(self, name, W):
        """overwrite tensor `name` in place (file opened with mode='r+')"""
        t = self.T[name]; tn = t.tensor_type.name
        assert tn in INPLACE, f"{name} is {tn}; in-place baking needs a Q8_0/F16/BF16/F32 master"
        if tn == "Q8_0":
            new = q8_0(W)
        elif tn == "F32":
            new = W.astype(np.float32)
        elif tn == "F16":
            new = W.astype(np.float16)
        else:
            new = quants.quantize(W.astype(np.float32), t.tensor_type)
        t.data[...] = new.reshape(t.data.shape).view(t.data.dtype)


def writers(g, l):
    """residual writers of layer l: mixer output (ssm_out or attn_output) and MLP down-projection(s), any architecture"""
    p = f"blk.{l}."
    mix = [p + n for n in ("ssm_out.weight", "attn_output.weight") if p + n in g.T][:1]
    return mix + [p + n for n in ("ffn_down.weight", "ffn_down_exps.weight", "ffn_down_shexp.weight") if p + n in g.T]


def _env(host=None):
    return dict(os.environ, OLLAMA_HOST=host) if host else None


def ollama_show(tag, host=None):
    mf = subprocess.run(["ollama", "show", "--modelfile", tag], capture_output=True, text=True, check=True, env=_env(host)).stdout
    froms = re.findall(r"^FROM (\S+)", mf, re.M)
    return froms[0], mf


def bake(base_path, out_path, specs, edit_fn):
    """copy base GGUF and apply `specs` in order. edit_fn(g, name, W, specs) -> W' or None (untouched)."""
    shutil.copyfile(base_path, out_path)
    g = GGUF(out_path, "r+"); n = 0
    for name in list(g.T):
        if not (name == "output_norm.weight" or re.match(r"blk\.\d+\.(ssm_out|attn_output|ffn_down)\.weight$", name)):
            continue
        W = g.deq(name)
        W2 = edit_fn(g, name, W, specs)
        if W2 is not None:
            g.write(name, W2); n += 1
    g.r.data.flush(); del g
    return n


def find_quantizer():
    """llama.cpp's llama-quantize: $RSI_LLAMA_QUANTIZE, on PATH, or next to ollama's own libraries"""
    cands = [os.environ.get("RSI_LLAMA_QUANTIZE", ""), shutil.which("llama-quantize") or ""]
    ol = shutil.which("ollama")
    if ol:
        root = os.path.dirname(os.path.dirname(os.path.realpath(ol)))
        cands += [os.path.join(root, "lib", "ollama", "llama-quantize"), os.path.join(root, "lib", "llama-quantize")]
    cands += ["/usr/local/lib/ollama/llama-quantize", "/usr/lib/ollama/llama-quantize", "/opt/ollama/lib/ollama/llama-quantize"]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError("llama-quantize not found (it ships with some ollama builds and with llama.cpp). Install llama.cpp "
                            "or set RSI_LLAMA_QUANTIZE; until then use F16/Q8_0 GGUFs, or `ollama create --quantize` from F16.")


def quantize(src, dst, qtype):
    """re-quantize a whole GGUF; raises with llama-quantize's own message on failure"""
    q = find_quantizer()
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(q) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    p = subprocess.run([q, "--allow-requantize", src, dst, qtype], capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-1500:])
    return dst


def derive_run_quant(src, dst, qtype):
    quantize(src, dst, qtype)


DIRECTIVES = ("PARAMETER", "SYSTEM", "TEMPLATE", "ADAPTER", "MESSAGE", "LICENSE", "RENDERER", "PARSER")


def merge_modelfile(base, extra):
    """apply Modelfile lines from `extra` on top of `base`: directives it sets replace the base's (PARAMETERs per key)"""
    if not (extra or "").strip():
        return base
    keys, params = set(), set()
    for l in extra.splitlines():
        m = re.match(r"^(\w+)\s*(\S*)", l)
        if m and m.group(1).upper() in DIRECTIVES:
            if m.group(1).upper() == "PARAMETER":
                params.add(m.group(2).lower())
            else:
                keys.add(m.group(1).upper())
    out = base
    for k in keys:
        out = re.sub(rf'^{k} """.*?"""[ \t]*$', "", out, flags=re.M | re.S | re.I)
        out = re.sub(rf"^{k} .*$", "", out, flags=re.M | re.I)
    for k in params:
        out = re.sub(rf"^PARAMETER {re.escape(k)} .*$", "", out, flags=re.M | re.I)
    return re.sub(r"\n{3,}", "\n\n", out).rstrip() + "\n" + extra.strip() + "\n"


def register(tag, gguf_path, base_tag, host=None, extra=""):
    """ollama create `tag`: the weights in gguf_path (or base_tag's own weights when None), every other Modelfile line
    of base_tag (projector, renderer, parser, parameters), then the Modelfile lines in `extra` on top. `host` must be
    a server allowed to write the model store."""
    blob, mf = ollama_show(base_tag, host)
    if gguf_path:
        mf = mf.replace(f"FROM {blob}", f"FROM {os.path.abspath(gguf_path)}", 1)
    mf = merge_modelfile(mf, extra)
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(mf); path = f.name
    try:
        p = subprocess.run(["ollama", "create", tag, "-f", path], capture_output=True, text=True, env=_env(host))
        if p.returncode:
            raise RuntimeError((p.stderr or p.stdout).strip()[-800:])
    finally:
        os.unlink(path)
    return mf


def unregister(tag, host=None):
    subprocess.run(["ollama", "rm", tag], capture_output=True, env=_env(host))
