"""Closed-form edit operators. A spec is a plain dict, so it can be logged, replayed and baked byte-identically.

  {"op": "ablate",  "dir": "dirs/c1.npz", "dir_layer": 17, "layers": [12, 23], "strength": 0.5}   W <- (I - a r r^T) W
  {"op": "amplify", "dir": "dirs/c2.npz", "dir_layer": 17, "layers": [8, 23],  "strength": 0.5}   W <- (I + b r r^T) W
  {"op": "sharpen", "strength": 1.6}                                                                output_norm <- k * output_norm
  {"op": "layer_scale", "layers": [20, 20], "which": "mlp", "strength": 0.95}                        W <- s W

`dir` files hold `d` with shape (n_layers, d_model); the direction used is d[dir_layer] normalised to unit length.
Writers: mixer output (ssm_out / attn_output) and MLP down-projection (ffn_down) of each layer in `layers` (inclusive).
"""
import re
import numpy as np

OPS = ("ablate", "amplify", "sharpen", "layer_scale")
_cache = {}


def direction(spec, base_dir=""):
    key = (spec["dir"], spec["dir_layer"])
    if key not in _cache:
        import os
        path = spec["dir"] if os.path.isabs(spec["dir"]) else os.path.join(base_dir, spec["dir"])
        d = np.load(path)["d"][spec["dir_layer"]].astype(np.float32)
        _cache[key] = d / np.linalg.norm(d)
    return _cache[key]


def validate(spec, n_layers):
    op = spec.get("op")
    if op not in OPS:
        return f"unknown op {op!r}; choose one of {OPS}"
    s = spec.get("strength")
    if not isinstance(s, (int, float)):
        return "strength must be a number"
    if op == "ablate" and not 0 < s <= 1:
        return "ablate strength must be in (0, 1]"
    if op == "amplify" and not 0 < s <= 2:
        return "amplify strength must be in (0, 2]"
    if op == "sharpen" and not 0.5 <= s <= 3:
        return "sharpen strength must be in [0.5, 3]"
    if op == "layer_scale" and not 0.5 <= s <= 1.5:
        return "layer_scale strength must be in [0.5, 1.5]"
    if op != "sharpen":
        lo, hi = spec.get("layers", [None, None])
        if not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo <= hi < n_layers):
            return f"layers must be [lo, hi] with 0 <= lo <= hi < {n_layers}"
    if op in ("ablate", "amplify"):
        if not isinstance(spec.get("dir_layer"), int) or not spec.get("dir"):
            return "ablate/amplify need 'dir' and integer 'dir_layer'"
    if op == "layer_scale" and spec.get("which") not in ("mixer", "mlp", "both"):
        return "layer_scale needs which in mixer|mlp|both"
    return None


def ops_for(name, specs, base_dir=""):
    """the operations `specs` imply for tensor `name`, in order: [("rank1", r, c) | ("scale", s)]"""
    out = []
    if name == "output_norm.weight":
        for s in specs:
            if s["op"] == "sharpen":
                out.append(("scale", float(s["strength"])))
        return out
    m = re.match(r"blk\.(\d+)\.(ssm_out|attn_output|ffn_down)\.weight$", name)
    if not m:
        return out
    l, kind = int(m.group(1)), ("mlp" if m.group(2) == "ffn_down" else "mixer")
    for s in specs:
        if s["op"] == "sharpen" or not (s["layers"][0] <= l <= s["layers"][1]):
            continue
        if s["op"] == "layer_scale":
            if s["which"] in (kind, "both"):
                out.append(("scale", float(s["strength"])))
        else:
            c = -float(s["strength"]) if s["op"] == "ablate" else float(s["strength"])
            out.append(("rank1", direction(s, base_dir), c))
    return out


def apply_np(W, ops):
    for op in ops:
        if op[0] == "scale":
            W = W * op[1]
        else:
            r, c = op[1], op[2]
            W = W + c * np.outer(r, r @ W)
    return W


def apply_torch(W, ops):
    import torch
    for op in ops:
        if op[0] == "scale":
            W = W * op[1]
        else:
            r = torch.as_tensor(op[1], device=W.device, dtype=torch.float32)
            Wf = W.float()
            W = (Wf + op[2] * torch.outer(r, r @ Wf)).to(W.dtype)
    return W


def bake_fn(base_dir=""):
    def fn(g, name, W, specs):
        ops = ops_for(name, specs, base_dir)
        return apply_np(W, ops) if ops else None
    return fn
