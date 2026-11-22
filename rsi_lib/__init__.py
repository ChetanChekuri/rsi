"""Helpers for working on your own weights from inside the rsi sandbox.

    from rsi_lib import *
    ctx = info()                       # goal, your model's ollama name, the path of your weights (GGUF), generation, ...
    names = list_tensors(r"blk\\.20\\.")  # what is in your weights
    W = read_tensor("blk.20.ffn_down.weight")
    out = write_model({"blk.20.ffn_down.weight": W * 0.98}, "candidate.gguf")   # a new model file in the workspace

You are not limited to these helpers: write any code you like. Only the workspace (the current directory) persists;
everything else in the sandbox is read-only or thrown away. The network is off inside the sandbox (use the
web_search / fetch_page tools instead). To turn a GGUF into a model you can run or test, call the ollama_create tool.
"""
import glob, json, os, re, shutil
import numpy as np

WS = os.environ.get("RSI_WORKSPACE", os.getcwd())


def info():
    """the loop's context for this cycle (written into the workspace at the start of every cycle)"""
    return json.load(open(os.path.join(WS, ".rsi_context.json")))


def current_gguf():
    """path of the GGUF file holding your current weights (read-only)"""
    return info()["current_gguf"]


def open_gguf(path=None):
    from rsi.gguf_io import GGUF
    return GGUF(path or current_gguf())


def list_tensors(pattern=".*", path=None):
    """[(name, shape, storage type)] for tensors whose name matches `pattern`"""
    g = open_gguf(path)
    return [(n, g.shape(n), g.T[n].tensor_type.name) for n in g.T if re.search(pattern, n)]


def read_tensor(name, path=None):
    """a tensor as a float32 numpy array (matrices are [out, in])"""
    return open_gguf(path).deq(name)


def write_model(edits, out, base=None):
    """Copy `base` (default: your current weights) to `out` in the workspace and overwrite the given tensors.
    edits: {tensor name: float32 array with the same shape}. Returns the absolute path. Tensors stored as Q8_0, F16,
    BF16 or F32 can be overwritten in place. Each copy is the full model size: delete files you no longer need."""
    from rsi.gguf_io import GGUF
    out = out if os.path.isabs(out) else os.path.join(WS, out)
    shutil.copyfile(base or current_gguf(), out)
    g = GGUF(out, "r+")
    for name, W in edits.items():
        g.write(name, np.asarray(W, dtype=np.float32))
    g.r.data.flush(); del g
    return out


def writers(layer, path=None):
    """names of the matrices that write into the residual stream at `layer` (attention/linear-attention output, MLP down)"""
    from rsi.gguf_io import writers as w
    return w(open_gguf(path), layer)


def n_layers(path=None):
    return int(open_gguf(path).meta("block_count"))


def stream(path=None, device="auto"):
    """a layer-streaming forward pass over a GGUF: capture activations, score continuations (see rsi/arch_qwen35.py)"""
    from rsi.arch_qwen35 import Qwen35Stream
    return Qwen35Stream(path or current_gguf(), info()["tokenizer"], device)


def practice_traces(pattern="*"):
    """your practice runs (each line: problem id, your thinking, answer, correct, ...), newest first"""
    files = sorted(glob.glob(os.path.join(WS, "practice", pattern + ".jsonl")), key=os.path.getmtime, reverse=True)
    return {os.path.basename(f): [json.loads(l) for l in open(f)] for f in files}


def doubt_direction(traces, post_answer=True, path=None):
    """Per-layer direction separating 'Wait…/Hmm…' lines from normal reasoning steps in `traces` (one example of
    finding a behaviour inside yourself). Returns (D [layers x width], rows with held-out AUC per layer)."""
    from rsi import directions
    S = stream(path)
    byid = {r["id"]: {"prompt": r.get("prompt", "")} for r in traces}
    seqs, caps, y, g = directions.decision_points(S, traces, byid, post_answer_only=post_answer)
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
