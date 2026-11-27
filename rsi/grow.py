"""Growing a model: new parameters it can then train, added so that at first nothing changes.

    widen_mlp(base, out, extra=256, layers=None)   more MLP neurons per layer: their input rows start random (so they
                                                   see something), their output column starts at ZERO, so the model
                                                   computes exactly what it did before
    add_layers(base, out, after, n=1)               new transformer blocks inserted after layer `after`, copies of a
                                                   neighbour whose residual writers (attention / linear-attention output
                                                   and MLP down) are ZERO, so each new block passes its input through

Both write a new GGUF with updated shapes and layer metadata. Quantized matrices can be stored at higher precision
to preserve their existing values. Verify the untrained grown model in the intended inference runtime. The returned map
identifies the new parameters:
    {"blk.20.ffn_down.weight": ["cols", [3584, ..., 3839]], "blk.20.ffn_up.weight": ["rows", [...]], ...}
which measure_gradient(exact=new) and gradient_step(kind="exact") use to train only those parameters. The map is also
saved next to the GGUF as <out>.new.json.
"""
import json, zlib
import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q, GGUFValueType as VT
from .gguf_io import GGUF, atomic_output, q8_0

WRITABLE = ("F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1")
WRITERS = ("ffn_down", "attn_output", "ssm_out")
ARCHITECTURES = {"llama", "qwen2", "qwen3", "qwen35"}


def encode(W, typ):
    """float32 array -> (data, ggml type) in `typ` (F32 / F16 / BF16 / Q8_0 / any type gguf-py can quantize)"""
    W = np.asarray(W, dtype=np.float32); t = Q[typ.upper()]
    if t == Q.F32:
        return W, t
    if t == Q.F16:
        return W.astype(np.float16), t
    if t == Q.Q8_0:
        return q8_0(W.reshape(-1, W.shape[-1]) if W.ndim > 1 else W.reshape(1, -1)).reshape(*W.shape[:-1], -1), t
    return gguf.quants.quantize(W, t), t


def _type_for(g, name):
    typ = g.T[name].tensor_type.name
    return typ if typ in WRITABLE else "F32"


def _layout(shape, typ):
    """(byte shape, numpy dtype, nbytes) of a tensor of logical `shape` stored as `typ`, without encoding it"""
    t = Q[typ.upper()]
    if t == Q.F32:
        return tuple(shape), np.float32, int(np.prod(shape)) * 4
    if t == Q.F16:
        return tuple(shape), np.float16, int(np.prod(shape)) * 2
    bs = gguf.quant_shape_to_byte_shape(tuple(shape), t)
    return tuple(bs), np.uint8, int(np.prod(bs))


def rewrite(base, out, entries, set_meta=None):
    """write `out` from `base`, one tensor at a time (memory stays at one tensor):
    entries = ordered [(name, ("raw", old_name)) | (name, ("packed", make, logical_shape, type))] where make()
    returns the encoded data; set_meta {key: value} replaces metadata (same value type as before)."""
    with atomic_output(base, out) as staging:
        _rewrite(base, staging, entries, set_meta)
    return out


def _rewrite(base, out, entries, set_meta):
    r = gguf.GGUFReader(base); set_meta = dict(set_meta or {})
    w = gguf.GGUFWriter(out, arch=r.fields["general.architecture"].contents(), endianess=r.endianess)
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]; st = f.types[-1] if vt == VT.ARRAY else None
        w.add_key_value(f.name, set_meta.pop(f.name) if f.name in set_meta else f.contents(), vt, sub_type=st)
    for k, v in set_meta.items():   # new keys
        if isinstance(v, list):
            w.add_key_value(k, v, VT.ARRAY, sub_type=VT.BOOL if v and isinstance(v[0], bool) else VT.UINT32)
        else:
            w.add_key_value(k, v, VT.UINT32 if isinstance(v, int) else VT.FLOAT32)
    old = {t.name: t for t in r.tensors}; sizes = {}
    for name, src in entries:
        if src[0] == "raw":
            t = old[src[1]]
            w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        else:
            bshape, dt, nb = _layout(src[2], src[3]); sizes[name] = nb
            w.add_tensor_info(name, bshape, dt, nb, Q[src[3].upper()])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for name, src in entries:
        if src[0] == "raw":
            w.write_tensor_data(old[src[1]].data, tensor_endianess=r.endianess)
        else:
            d = src[1]()
            assert d.nbytes == sizes[name], f"{name}: encoded {d.nbytes} bytes, expected {sizes[name]}"
            w.write_tensor_data(d, tensor_endianess=r.endianess); del d
    w.close()
    return out


def _save_new(out, new):
    with open(out + ".new.json", "w") as f:
        json.dump(new, f, indent=1)
    return new


def widen_mlp(base, out, extra=256, layers=None, init=1.0, seed=0):
    """add `extra` neurons to the MLP of `layers` (default: all). New gate/up rows are random with the spread of the
    existing rows times `init`; new down columns are zero, so the model computes exactly what it did before. Every
    layer gets the same width (one feed_forward_length, which every runtime accepts): in layers you didn't choose the
    extra neurons are all-zero and stay dead. `extra` is rounded up to a multiple of 32 (quantization blocks)."""
    g = _dense_model(base); a = g.arch; L = int(g.meta("block_count"))
    chosen = set(range(L) if layers is None else {int(l) for l in layers})
    if int(extra) <= 0 or not chosen or not chosen <= set(range(L)) or not np.isfinite(init) or init <= 0:
        raise ValueError("use positive extra/init and select one or more valid layers")
    extra = -(-int(extra) // 32) * 32
    ff = g.meta("feed_forward_length")
    if isinstance(ff, list) and len(set(int(v) for v in ff)) > 1:
        raise NotImplementedError("this model already has different MLP sizes per layer")
    entries, new = [], {}
    for name in g.T:
        parts = name.split("."); l = int(parts[1]) if parts[0] == "blk" else None
        if l is not None and name.endswith(("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")):
            live = l in chosen; rows, cols = g.shape(name); down = name.endswith("ffn_down.weight")
            typ = _type_for(g, name)
            if down:                                   # (D, n_ff): new columns at zero, so outputs are unchanged
                shape = (rows, cols + extra); idx = ["cols", list(range(cols, cols + extra))]
            else:                                      # (n_ff, D): new rows like the existing ones (zero = dead)
                shape = (rows + extra, cols); idx = ["rows", list(range(rows, rows + extra))]

            def make(name=name, live=live, down=down, rows=rows, cols=cols, typ=typ):
                stored = g.T[name]
                old = stored.data if typ == stored.tensor_type.name else g.deq(name)
                if down or not live:
                    added = np.zeros((rows, extra) if down else (extra, cols), np.float32)
                else:
                    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
                    added = (rng.standard_normal((extra, cols)) * (float(g.deq(name).std()) * init)).astype(np.float32)
                return np.concatenate([old, encode(added, typ)[0]], axis=1 if down else 0)

            if live:
                new[name] = idx
            entries.append((name, ("packed", make, shape, typ)))
        else:
            entries.append((name, ("raw", name)))
    width = int(ff[0] if isinstance(ff, list) else ff) + extra
    rewrite(base, out, entries, {f"{a}.feed_forward_length": [width] * L if isinstance(ff, list) else width})
    return _save_new(out, new)


def _dense_model(base):
    g = GGUF(base)
    if g.arch not in ARCHITECTURES or any("_exps." in name or "_shexp." in name for name in g.T):
        raise NotImplementedError(f"growth supports dense {', '.join(sorted(ARCHITECTURES))} models")
    for layer in range(int(g.meta("block_count"))):
        for matrix in ("ffn_gate", "ffn_up", "ffn_down"):
            if f"blk.{layer}.{matrix}.weight" not in g.T:
                raise ValueError(f"missing MLP tensor {matrix} in block {layer}")
    return g


def add_layers(base, out, after, n=1, copy_from=None):
    """insert `n` new blocks after layer `after` (-1 = before the first). Each copies layer `copy_from` (default:
    `after`, or 0) with its residual writers set to zero, so it passes its input through unchanged; later layers are
    renumbered. Per-layer metadata lists (layer types, sizes) get the copied layer's entry."""
    g = _dense_model(base); a = g.arch; L = int(g.meta("block_count")); after = int(after); n = int(n)
    src = int(copy_from) if copy_from is not None else max(after, 0)
    if n < 1:
        raise ValueError("n must be positive")
    if not (-1 <= after < L and 0 <= src < L):
        raise ValueError(f"after must be in -1..{L - 1} and copy_from in 0..{L - 1}")
    blk = lambda name: int(name.split(".")[1]) if name.startswith("blk.") else None
    entries, new = [], {}
    for name in g.T:   # existing tensors, later blocks renumbered, raw bytes untouched
        l = blk(name)
        out_name = name if l is None or l <= after else f"blk.{l + n}." + name.split(".", 2)[2]
        entries.append((out_name, ("raw", name)))
    for k in range(n):
        tgt = after + 1 + k
        for name in [x for x in g.T if blk(x) == src]:
            suffix = name.split(".", 2)[2]; shape = g.shape(name); nn = f"blk.{tgt}.{suffix}"
            if suffix.rsplit(".", 1)[0] in WRITERS:
                typ = _type_for(g, name)
                make = lambda shape=shape, typ=typ: encode(np.zeros(shape, np.float32), typ)[0]
                entries.append((nn, ("packed", make, shape, typ)))
            else:
                entries.append((nn, ("raw", name)))
            if len(shape) == 2 and suffix != "ssm_conv1d.weight":
                new[nn] = "all"
    meta = {f"{a}.block_count": L + n}
    for key, f in g.r.fields.items():   # per-layer lists: insert the copied layer's entry
        if key.startswith(a + ".") and f.types[0] == VT.ARRAY and len(f.types) > 1 and f.types[-1] != VT.STRING:
            v = f.contents()
            if isinstance(v, list) and len(v) == L:
                meta[key] = v[:after + 1] + [v[src]] * n + v[after + 1:]
    recurrent = f"{a}.attention.recurrent_layers"
    if a == "qwen35" and recurrent not in meta:
        v = [f"blk.{i}.ssm_out.weight" in g.T for i in range(L)]
        meta[recurrent] = v[:after + 1] + [v[src]] * n + v[after + 1:]
    rewrite(base, out, entries, meta)
    return _save_new(out, new)
