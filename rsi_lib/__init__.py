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
    m = measure_gradient([(chat_text("17 x 6? Just the number."), "102"), ...])   # which way weights want to move
    gradient_step(m, "step1.gguf")                                     # one small step, as a new GGUF
    g = grow_mlp("wide.gguf", extra=256, layers=range(8, 16))         # new neurons (start inactive), or grow_layers
    m = measure_gradient(ex, path=g["path"], exact=g["new"]); gradient_step(m, "w1.gguf", kind="exact")   # train them

Then call the ollama_create tool to register a GGUF as a model, and run_tests to compare it with yourself.
Only the workspace persists. Inside the offline sandbox, pip can install local wheels into .pkgs/ (already on the path).
"""
import glob, json, os, re, secrets, shutil
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
    from rsi.gguf_io import GGUF, atomic_output
    out = out if os.path.isabs(out) else os.path.join(WS, out)
    base = base or current_gguf()
    with atomic_output(base, out) as staging:
        shutil.copyfile(base, staging)
        g = GGUF(staging, "r+")
        for name, W in edits.items():
            g.write(name, np.asarray(W, dtype=np.float32))
        g.r.data.flush()
        del g
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
    from rsi.gguf_io import atomic_output
    out = out if os.path.isabs(out) else os.path.join(WS, out)
    base = base or current_gguf()
    with atomic_output(base, out) as staging:
        _write_gguf(staging, tensors, types, metadata, drop, base)
    return out


def _write_gguf(out, tensors, types, metadata, drop, base):
    import gguf
    tensors, types, metadata, drop = dict(tensors or {}), dict(types or {}), dict(metadata or {}), set(drop)
    r = gguf.GGUFReader(base)
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
                    v = v(GGUF(base).deq(n))
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


def requantize(src, dst, qtype="Q4_K_M"):
    """re-quantize a whole GGUF with llama.cpp's llama-quantize (found next to ollama, on PATH, or $RSI_LLAMA_QUANTIZE)"""
    from rsi.gguf_io import quantize
    dst = dst if os.path.isabs(dst) else os.path.join(WS, dst)
    return quantize(src, dst, qtype)


def chat(model, messages, think=True, tools=None, **options):
    """one reply from any model on the ollama server. messages: a string or a list of {role, content}.
    think: True/False/"low"/"medium"/"high" (ignored for models that can't think). tools: function schemas (models
    without native tool calling get them as text; you still get tool_calls back). options: ollama options
    (num_predict, num_ctx, temperature, seed, ...). Returns the message dict (content, thinking, tool_calls)."""
    from rsi import llm
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    cfg = {"ollama_url": host if host.startswith("http") else f"http://{host}", "num_ctx": options.pop("num_ctx", 8192),
           "num_predict": options.pop("num_predict", 4096)}
    return llm.chat(cfg, model, messages, think=think, tools=tools, seed=options.pop("seed", 0), options=options, timeout=3600)["message"]


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


def stream(path=None, device="auto", force=False):
    """a layer-streaming forward pass over a GGUF (activations, log-likelihoods; ~1 GB resident even for 27B).
    Qwen3.5/3.8: rsi/arch_qwen35.py. GGUF architectures llama (Llama 2/3, Mistral 7B, ...), qwen2, qwen3:
    rsi/arch_generic.py (BPE and SentencePiece tokenizers from the GGUF). Anything else raises NotImplementedError:
    subclass GenericStream for it. Verify a stream with check_stream() before trusting it. Tokens:
    S.bos + S.enc(text) (S.enc never adds the start token; models such as Llama 3 need it)."""
    from rsi.arch_generic import open_stream
    return open_stream(path or current_gguf(), device, force)


def check_stream(S=None, model=None):
    """(agreed, total, details): does the stream predict the same next token as ollama does for this model?"""
    from rsi.arch_generic import check_stream as chk
    return chk(S or stream(), model or info()["current_model"])


def chat_text(user, S=None, system=None, thinking=False, assistant_prefix=""):
    """your chat template (from the GGUF) rendered for one user turn, so an answer can follow; for measure_gradient"""
    from rsi.gradstep import chat_text as ct
    return ct(S or stream(), user, system, thinking, assistant_prefix)


def measure_gradient(examples, path=None, S=None, exact=None, **kw):
    """Which way do your weights want to move on these examples? Nothing is applied.
    examples: [(prompt_text, answer_text)]; the loss is on the answer tokens only (build prompts with chat_text, in
    the exact format you are used in). path: the weights to measure (default: your current ones; measure a candidate
    you built to take the next step from it). Options: layers (default: second half), loss="margin" (correct vs your
    top wrong token; "nll" also available but chases cheap shifts), targets (matrix names), rank, batch.
    Returns a dict whose "summary" says: loss before, answer-token accuracy, how much the examples agree on one
    direction (agreement_between_halves: ~0 none, ~0.5-0.8 a shared skill, ~1 a blanket shift), and which matrices
    the gradient is in. See the playbook: "Training yourself in observed steps"."""
    from rsi.gradstep import measure
    if S is not None and path is not None and not os.path.samefile(path, S.g.path):
        raise ValueError("path must identify the same model as the supplied stream")
    S = S if S is not None else stream(path)
    m = measure(S, examples, exact=exact, **kw)
    m["base"] = os.fspath(S.g.path)
    return m


def practice_problems(seed=None, per_family=16):
    """Fresh local development examples with known answers. Keep training and development seeds separate."""
    from rsi.procedural import ANSWER_FORMAT, generate
    if not 1 <= per_family <= 10000:
        raise ValueError("per_family must be between 1 and 10000")
    rows = generate(secrets.randbits(128) if seed is None else seed, per_family)
    return [dict(row, prompt=row["prompt"] + ANSWER_FORMAT) for row in rows]


def train_step(out, problems=None, *, base=None, kind=None, size=None, layers=None, device="auto"):
    """Train once on supplied problem rows or fresh local practice. Rows may override the formatted answer with completion."""
    S = stream(base, device=device)
    exact = grown(S.g.path)
    kind = kind or ("exact" if exact else "gains")
    if kind == "exact" and not exact:
        raise ValueError("exact training needs the grown model's parameter map")
    seed = secrets.randbits(128) if problems is None else None
    rows = practice_problems(seed, per_family=8) if problems is None else problems
    examples = []
    for row in rows:
        answer = row["completion"] if "completion" in row else "Answer: " + str(row["answer"])
        if not isinstance(answer, str) or not answer.strip() or not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError("training rows need a nonempty prompt and answer")
        examples.append((chat_text(row["prompt"], S=S), answer))
    options = {"targets": ()} if kind == "exact" else {}
    if layers is None:
        if kind == "exact":
            layers = sorted({int(name.split(".")[1]) for name in exact})
        else:
            layers = range(max(0, S.L - 4), S.L)
    measurement = measure_gradient(examples, S=S, exact=exact if kind == "exact" else None, layers=layers, **options)
    if kind == "exact" and measurement["summary"]["exact_missing"]:
        raise ValueError("some new parameters were not used by the forward pass")
    outcome = gradient_step(measurement, out, kind=kind, size=0.01 if size is None and kind == "exact" else size)
    return dict(outcome, training=dict(measurement["summary"], generated_seed=seed))


def gradient_step(m, out, kind="gains", size=None, top=None):
    """One small step against a measurement, written as a new GGUF (from the weights that were measured).
    kind="gains": scale the output columns of the `top` neurons with the largest gain gradients (default 3% of the
       measured neurons) so that the mean change is `size` (default 0.015 = 1.5%); each change is capped at 30%.
    kind="lowrank": add the measured top gradient directions of each target matrix (attention/MLP outputs), scaled
       so the total change is `size` (default 0.003 = 0.3%) of the measured matrices' norm. Can change behaviour
       more than gains, including how you answer (check every format you are used in).
    Returns {"path", "kind", "size", "changed"}. Register it with ollama_create, test it with run_tests, and measure
    the next step from it (measure_gradient(path=...)) only if it passed."""
    from rsi.gguf_io import GGUF
    if size is not None and (not np.isfinite(size) or size <= 0):
        raise ValueError("size must be positive and finite")
    base = m["base"]; g = GGUF(base); edits, changed = {}, 0
    if kind == "gains":
        G = m["gain_grads"].mean(0); layers = m["gain_layers"]; n = G.shape[1]
        size = 0.015 if size is None else size; top = top or max(1, int(0.03 * G.size))
        idx = np.argsort(-np.abs(G).ravel())[:top]; s = size / (np.abs(G.ravel()[idx]).mean() + 1e-30)
        d = np.clip(-s * G.ravel()[idx], -0.3, 0.3); cols = {}
        for i, dd in zip(idx, d):
            cols.setdefault(layers[int(i) // n], {})[int(i) % n] = 1.0 + float(dd)
        for l, c in cols.items():
            name = f"blk.{l}.ffn_down.weight"; W = np.array(g.deq(name), dtype=np.float32); js = list(c)
            W[:, js] *= np.array([c[j] for j in js], dtype=np.float32); edits[name] = W; changed += len(js)
    elif kind == "lowrank":
        size = 0.003 if size is None else size
        names = {k: f"blk.{l}.{kind_}.weight" for k in m["factors"] for kind_, l in [k]}
        Ws = {k: np.array(g.deq(nm), dtype=np.float32) for k, nm in names.items()}
        wn = np.sqrt(sum(float((W ** 2).sum()) for W in Ws.values())); gn = np.sqrt(sum(e[0] for e in m["energy"].values()))
        eta = size * wn / (gn + 1e-30)
        for k, (U, V) in m["factors"].items():
            edits[names[k]] = Ws[k] - eta * (U @ V.T); changed += 1
    elif kind == "exact":   # train only the parameters measured with exact= (e.g. the ones you grew)
        size = 0.1 if size is None else size

        def ref_rms(name, W, spec):
            """typical size of this matrix's existing weights (for a brand-new all-zero matrix: the same matrix in
            another layer)"""
            old = W if spec == "all" else (np.delete(W, spec[1], 1) if spec[0] == "cols" else np.delete(W, spec[1], 0))
            r = float(np.sqrt((old ** 2).mean())) if old.size else 0.0
            if r == 0.0:
                suffix = name.split(".", 2)[2]
                for other in sorted(n for n in g.T if n.endswith("." + suffix) and n != name):
                    r = float(np.sqrt((g.deq(other) ** 2).mean()))
                    if r:
                        break
            return r
        for name, (spec, G) in m["exact"].items():
            W = np.array(g.deq(name), dtype=np.float32); gr = float(np.sqrt((G ** 2).mean()))
            if gr == 0.0:
                continue
            step = -(size * ref_rms(name, W, spec) / gr) * G   # RMS of the change = size x the typical weight
            if spec == "all":
                W += step
            elif spec[0] == "cols":
                W[:, spec[1]] += step
            else:
                W[spec[1], :] += step
            edits[name] = W; changed += 1
    else:
        raise ValueError("kind is 'gains', 'lowrank' or 'exact'")
    if not edits:
        raise ValueError("the measurement produced no weight updates")
    if any(not np.isfinite(W).all() for W in edits.values()):
        raise ValueError("the weight update contains a non-finite value")
    if kind == "exact":
        types = {}
        for name, (spec, _) in m["exact"].items():
            if name not in edits or spec == "all":
                continue
            storage = g.T[name].tensor_type
            if storage.name not in ("F32", "F16", "BF16"):
                # Quantized blocks can share a scale between edited and frozen columns. Promotion to F32
                # keeps the original dequantized values exactly for every frozen entry.
                types[name] = "F32"
        path = write_gguf(out, tensors=edits, types=types, base=base)
    else:
        try:
            path = write_model(edits, out, base=base)
        except AssertionError:   # tensor types that can't be overwritten in place: re-encode
            path = write_gguf(out, tensors=edits, base=base)
    if os.path.exists(base + ".new.json"):   # a grown model: the map of its new parameters carries over
        shutil.copyfile(base + ".new.json", path + ".new.json")
    return {"path": path, "kind": kind, "size": size, "changed": changed}


def grow_mlp(out, extra=256, layers=None, init=1.0, base=None):
    """Grow: `extra` new MLP neurons in `layers` (default all), written as a new GGUF. Their output weights start at
    zero, so you behave exactly as before until you train them. Returns {"path", "new"}: `new` maps each new
    parameter slice; train them with measure_gradient(examples, path=..., exact=new) and
    gradient_step(m, out, kind="exact"). Also saved as <path>.new.json."""
    from rsi.grow import widen_mlp
    out = out if os.path.isabs(out) else os.path.join(WS, out)
    new = widen_mlp(base or current_gguf(), out, extra=extra, layers=layers, init=init)
    return {"path": out, "new": new}


def grow_layers(out, after, n=1, copy_from=None, base=None):
    """Grow: `n` new transformer layers after layer `after`, each a copy of a neighbour with its output matrices at
    zero, so it passes its input through until trained. Later layers are renumbered. Returns {"path", "new"} (all
    matrices of the new layers); train them with measure_gradient(..., exact=new) and gradient_step(kind="exact")."""
    from rsi.grow import add_layers
    out = out if os.path.isabs(out) else os.path.join(WS, out)
    new = add_layers(base or current_gguf(), out, after=after, n=n, copy_from=copy_from)
    return {"path": out, "new": new}


def grown(path):
    """the map of new parameters saved with a grown GGUF (or {})"""
    try:
        return json.load(open(path + ".new.json"))
    except OSError:
        return {}


def doubt_direction(records, post_answer=True, path=None):
    """Per-layer direction separating 'Wait…/Hmm…' lines from normal reasoning steps in test records (one example of
    finding a behaviour inside yourself). Returns (D [layers x width], rows with held-out AUC per layer)."""
    from rsi import directions
    S = stream(path)
    recs = [dict(r, gold=str(r.get("answer") or ""), kind="num" if re.fullmatch(r"-?[\d.]+", str(r.get("answer") or "")) else "word")
            for r in records if r.get("thinking")]
    byid = {r["id"]: {"prompt": r.get("prompt", "")} for r in recs}
    from rsi.gradstep import chat_text
    fmt = lambda q: chat_text(S, q, thinking=True)   # your own chat template, opened for thinking
    seqs, caps, y, g = directions.decision_points(S, recs, byid, post_answer_only=post_answer, fmt=fmt)
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
