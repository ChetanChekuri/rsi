"""Helpers for working on yourself from code in your workspace. You don't have to use them: write any code you like.

    from rsi_lib import *
    ctx = info()                        # goal, your model name, your weights (GGUF path), generation, hardware, ...
    list_tensors(r"blk\\.20\\.")          # [(name, shape, storage type)]
    W = read_tensor("blk.20.ffn_down.weight")                          # float32 numpy, [out, in]
    write_model({"blk.20.ffn_down.weight": W * 0.98}, "a.gguf")       # fast: copy + overwrite in place (same types)
    write_gguf("b.gguf", tensors={...}, types={...}, metadata={...}, drop=[...])   # anything: new tensors, types, metadata
    requantize("b.gguf", "b-q4.gguf", "Q4_K_M")                        # llama.cpp quantizer
    chat("rsi-xyz:b", "What is 17*23?")                                # talk to any model on the ollama server
    test_results()                      # the transcripts of your latest run_tests

Then call the ollama_create tool to register a GGUF as a model, and run_tests to compare it with yourself.
Only the workspace persists. Inside the sandbox `pip install` puts packages in .pkgs/ (already on the path).
"""
import glob, json, os, re, shutil, subprocess, urllib.request
import numpy as np

WS = os.environ.get("RSI_WORKSPACE", os.getcwd())   # /workspace inside the sandbox


def info():
    """the loop's context (rewritten at the start of every cycle and whenever you switch models)"""
    return json.load(open(os.path.join(WS, ".rsi_context.json")))


def current_gguf():
    """path of the GGUF holding your current weights (read-only)"""
    return info()["current_gguf"]


def master_gguf():
    """the original model's editable (Q8_0/F16) weights"""
    return info()["editable_master_gguf"]


def open_gguf(path=None):
    from rsi.gguf_io import GGUF
    return GGUF(path or current_gguf())


def list_tensors(pattern=".*", path=None):
    g = open_gguf(path)
    return [(n, g.shape(n), g.T[n].tensor_type.name) for n in g.T if re.search(pattern, n)]


def read_tensor(name, path=None):
    """a tensor as float32 numpy (matrices are [out, in])"""
    return open_gguf(path).deq(name)


def metadata(path=None):
    """all metadata fields of a GGUF as {key: value} (long arrays such as the vocabulary are summarised)"""
    import gguf
    r = gguf.GGUFReader(path or current_gguf()); out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        try:
            v = f.contents()
        except Exception:  # noqa: BLE001
            continue
        out[k] = v if not (isinstance(v, list) and len(v) > 64) else f"<{len(v)} items>"
    return out


def write_model(edits, out, base=None):
    """Copy `base` (default: your current weights) to `out` and overwrite tensors in place (same shape; the tensor
    must be stored as Q8_0, F16, BF16 or F32). edits: {name: float32 array}. Fast; returns the absolute path."""
    from rsi.gguf_io import GGUF
    out = out if os.path.isabs(out) else os.path.join(WS, out)
    shutil.copyfile(base or current_gguf(), out)
    g = GGUF(out, "r+")
    for name, W in edits.items():
        g.write(name, np.asarray(W, dtype=np.float32))
    g.r.data.flush(); del g
    return out


def _encode(W, typ):
    import gguf
    from gguf import GGMLQuantizationType as Q
    from rsi.gguf_io import q8_0
    W = np.asarray(W, dtype=np.float32); t = Q[typ.upper()]
    if t == Q.F32:
        return W, t
    if t == Q.F16:
        return W.astype(np.float16), t
    if t == Q.Q8_0:
        return q8_0(W.reshape(-1, W.shape[-1]) if W.ndim > 1 else W.reshape(1, -1)).reshape(*W.shape[:-1], -1), t
    return gguf.quants.quantize(W, t), t


def _value_type(v):
    from gguf import GGUFValueType as VT
    if isinstance(v, bool):
        return VT.BOOL, None
    if isinstance(v, int):
        return (VT.UINT32 if 0 <= v < 2**32 else VT.INT64), None
    if isinstance(v, float):
        return VT.FLOAT32, None
    if isinstance(v, str):
        return VT.STRING, None
    if isinstance(v, (list, tuple)):
        return VT.ARRAY, (_value_type(v[0])[0] if v else VT.INT32)
    raise TypeError(f"unsupported metadata value {v!r}")


def write_gguf(out, tensors=None, types=None, metadata=None, drop=(), base=None):
    """Write a new GGUF from `base` (default: your current weights) with any changes:
      tensors   {name: float32 array, or a function f(old float32 array) -> new array}: replace or add tensors (any shape)
      types     {name: "F32" | "F16" | "BF16" | "Q8_0" | "Q4_0" | ...}: storage type for tensors you replace or add
                (default: the old type if it can be written, else Q8_0; norms and 1-D tensors default to F32)
      metadata  {key: value}: set or add metadata (e.g. {"qwen35.block_count": 20}); value None removes a key
      drop      tensor names to leave out (e.g. to remove layers; then fix the metadata and renumber)
    Unchanged tensors are streamed from the base file; the ones you replace are held in memory until written.
    Returns the absolute path."""
    import gguf
    from gguf import GGMLQuantizationType as Q
    out = out if os.path.isabs(out) else os.path.join(WS, out)
    tensors, types, metadata, drop = dict(tensors or {}), dict(types or {}), dict(metadata or {}), set(drop)
    r = gguf.GGUFReader(base or current_gguf())
    w = gguf.GGUFWriter(out, arch=r.fields["general.architecture"].contents(), endianess=r.endianess)
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        if f.name in metadata:
            v = metadata.pop(f.name)
            if v is None:
                continue
            vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(f.name, v, vt, sub_type=st)
        else:
            vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(f.name, f.contents(), vt, sub_type=st)
    for k, v in metadata.items():
        if v is not None:
            vt, st = _value_type(v); w.add_key_value(k, v, vt, sub_type=st)
    old = {t.name: t for t in r.tensors}
    order = [n for n in old if n not in drop] + [n for n in tensors if n not in old]
    plan = {}
    for n in order:
        if n in tensors:
            def make(n=n):
                v = tensors[n]
                if callable(v):
                    from rsi.gguf_io import GGUF
                    v = v(GGUF(base or current_gguf()).deq(n))
                v = np.asarray(v, dtype=np.float32)
                typ = types.get(n) or (old[n].tensor_type.name if n in old and old[n].tensor_type.name in ("F32", "F16", "BF16", "Q8_0") else
                                      ("F32" if v.ndim == 1 or n.endswith("norm.weight") else "Q8_0"))
                return _encode(v, typ)
            data, t = make(); plan[n] = (data, t)
            w.add_tensor_info(n, data.shape, data.dtype, data.nbytes, t)
        else:
            t = old[n]; plan[n] = None
            w.add_tensor_info(n, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for n in order:
        w.write_tensor_data(plan[n][0] if plan[n] is not None else old[n].data, tensor_endianess=r.endianess)
    w.close()
    return out


def requantize(src, dst, qtype="Q4_K_M"):
    """re-quantize a whole GGUF with llama.cpp's llama-quantize (bundled with ollama)"""
    q = "/usr/local/lib/ollama/llama-quantize"
    dst = dst if os.path.isabs(dst) else os.path.join(WS, dst)
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(q) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    p = subprocess.run([q, "--allow-requantize", src, dst, qtype], capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    return dst


def chat(model, messages, think=True, tools=None, **options):
    """one reply from any model on the ollama server. messages: a string or a list of {role, content}.
    Returns the message dict (content, thinking, tool_calls)."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    body = {"model": model, "messages": messages, "stream": False, "think": think, "options": options}
    if tools:
        body["tools"] = tools
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    req = urllib.request.Request(f"http://{host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=3600) as resp:
        return json.loads(resp.read())["message"]


def test_results(run=None):
    """{model: [records]} for a test run (default: the latest). Each record: id, prompt, answer, thinking, content,
    tool_calls, done_reason, tokens, score."""
    runs = sorted(glob.glob(os.path.join(WS, "test_runs", "c*")), key=os.path.getmtime)
    if not runs:
        return {}
    d = os.path.join(WS, "test_runs", run) if run else runs[-1]
    return {os.path.basename(f)[:-6]: [json.loads(l) for l in open(f)] for f in sorted(glob.glob(os.path.join(d, "*.jsonl")))}


def writers(layer, path=None):
    """the matrices that write into the residual stream at `layer` (attention / linear-attention output, MLP down)"""
    from rsi.gguf_io import writers as w
    return w(open_gguf(path), layer)


def n_layers(path=None):
    return int(open_gguf(path).meta("block_count"))


def stream(path=None, device="auto"):
    """a layer-streaming forward pass over a GGUF (activations, log-likelihoods; ~1 GB resident even for 27B).
    Qwen3.5/3.8: rsi/arch_qwen35.py. llama, mistral, qwen2, qwen3, granite...: rsi/arch_generic.py. Anything else:
    write your own (subclass GenericStream). Verify a stream with check_stream() before trusting it. Tokens:
    S.bos + S.enc(text) (S.enc never adds the start token; models such as Llama 3 need it)."""
    from rsi.arch_generic import open_stream
    return open_stream(path or current_gguf(), device)


def check_stream(S=None, model=None):
    """(agreed, total, details): does the stream predict the same next token as ollama does for this model?"""
    from rsi.arch_generic import check_stream as chk
    return chk(S or stream(), model or info()["current_model"])


def doubt_direction(records, post_answer=True, path=None):
    """Per-layer direction separating 'Wait…/Hmm…' lines from normal reasoning steps in test records (one example of
    finding a behaviour inside yourself). Returns (D [layers x width], rows with held-out AUC per layer)."""
    from rsi import directions
    S = stream(path)
    recs = [dict(r, gold=str(r.get("answer") or ""), kind="num" if re.fullmatch(r"-?[\d.]+", str(r.get("answer") or "")) else "word")
            for r in records if r.get("thinking")]
    byid = {r["id"]: {"prompt": r.get("prompt", "")} for r in recs}
    seqs, caps, y, g = directions.decision_points(S, recs, byid, post_answer_only=post_answer)
    acts = S.run(seqs, capture=caps)[0]["pos"]
    rows, D = directions.stats(acts, y, g)
    return D, rows


def ablate(W, r, a):
    """remove a fraction `a` of the component along unit vector r from what W writes: W - a r r^T W"""
    r = r / np.linalg.norm(r)
    return W - a * np.outer(r, r @ W)


def amplify(W, r, b):
    """strengthen the component along r: W + b r r^T W"""
    r = r / np.linalg.norm(r)
    return W + b * np.outer(r, r @ W)
