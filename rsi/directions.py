"""Contrasts -> per-layer directions with honest separation statistics.

A contrast is two sets of residual-stream samples that differ in one behaviour and as little else as possible.
For each layer we report:
  norm_ratio   |mean(pos) - mean(neg)| / mean |x|           (how big the difference is)
  consistency  in-sample accuracy of the midpoint threshold   (does it separate at all)
  heldout_auc  2-fold AUC with folds split BY TRACE/PROBLEM   (does it generalise; positions from one trace never
                                                               appear on both sides of the split)
"""
import bisect, hashlib, re
import numpy as np
from .detectors import DOUBT, answer_pos
from .probes import chat_prefix

CONTRASTS = {
    "decision_points_doubt_vs_step": "line starts in dev thinking traces: the next line opens with a doubt marker (positive) vs. a normal step (negative)",
    "post_answer_doubt_vs_step": "the same, restricted to lines after the gold answer first appeared (isolates rumination)",
    "reference_vs_own": "the same problems: reference traces (positive) vs. the subject's own traces (negative), mean residual over the reasoning at matched length",
    "tool_call_vs_prose": "the last prompt position on tool prompts where the subject called the tool (positive) vs. answered in prose (negative)",
}


def _tok_index(ends, c):
    """index of the last token whose span ends at or before char c"""
    return bisect.bisect_right(ends, c) - 1


def decision_points(stream, records, items_by_id, post_answer_only=False, max_tokens=3072, per_trace=30, seed=0):
    rng = np.random.default_rng(seed)
    seqs, caps, labels, groups = [], [], [], []
    for gi, r in enumerate(records):
        th = r.get("thinking") or ""
        if not th:
            continue
        prompt = chat_prefix(items_by_id[r["id"]]["prompt"])
        text = prompt + th
        ids, offs = stream.enc_offsets(text)
        ids, offs = ids[:max_tokens], offs[:max_tokens]
        ends = [o[1] for o in offs]
        ap = answer_pos(th, r["gold"], r["kind"]); ap = len(prompt) + ap if ap >= 0 else -1
        pos, neg = [], []
        for m in re.finditer(r"\n+", th):
            c = len(prompt) + m.end()
            if c >= len(text) or (post_answer_only and not (ap >= 0 and c > ap)):
                continue
            t = _tok_index(ends, c)
            if t < 1 or t >= len(ids):
                continue
            (pos if DOUBT.match(text[c:c + 80]) else neg).append(t)
        if not pos:
            continue
        pos = list(rng.permutation(pos)[:per_trace]); neg = list(rng.permutation(neg)[:len(pos)])
        P = sorted(set(int(p) for p in pos + neg))
        lab = {int(p): 1 for p in pos}
        seqs.append(ids); caps.append({"pos": P}); labels += [lab.get(p, 0) for p in P]; groups += [gi] * len(P)
    return seqs, caps, np.array(labels), np.array(groups)


def reference_vs_own(stream, ref_items, records, max_tokens=2048):
    """ref_items: rows {id, prompt, thinking, answer}; records: the subject's own traces on the same prompts"""
    seqs, caps, labels, groups = [], [], [], []
    own = {r["id"]: r for r in records}
    for gi, it in enumerate(ref_items):
        if it["id"] not in own:
            continue
        prompt = chat_prefix(it["prompt"]); n0 = len(stream.enc(prompt))
        ref_ids = stream.enc(prompt + it["thinking"].strip() + "\n</think>\n\n" + it["answer"].strip())[:max_tokens]
        o = own[it["id"]]
        own_ids = stream.enc(prompt + (o.get("thinking") or "").strip() + "\n</think>\n\n" + (o.get("content") or "").strip())
        L = len(ref_ids) - n0
        own_ids = own_ids[:n0 + L]
        if L < 8 or len(own_ids) - n0 < 8:
            continue
        for ids, y in ((ref_ids, 1), (own_ids, 0)):
            seqs.append(ids); caps.append({"span": (n0, len(ids))}); labels.append(y); groups.append(gi)
    return seqs, caps, np.array(labels), np.array(groups)


def _auc(y, s):
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 3 or n0 < 3:
        return None
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stats(per_layer, y, groups):
    """per_layer: {l: (n, D)} in the same order as y/groups -> (table rows, directions (L, D))"""
    fold = np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 2 for g in groups])
    rows, dirs = [], []
    for l in sorted(per_layer):
        X = per_layer[l].astype(np.float32)
        d = X[y == 1].mean(0) - X[y == 0].mean(0)
        proj = X @ d; mid = (proj[y == 1].mean() + proj[y == 0].mean()) / 2
        cons = float(np.mean((proj > mid) == (y == 1)))
        aucs = []
        for f in (0, 1):
            tr, te = fold != f, fold == f
            if (y[tr] == 1).sum() < 3 or (y[tr] == 0).sum() < 3:
                continue
            dd = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)
            a = _auc(y[te], X[te] @ dd)
            if a is not None:
                aucs.append(a)
        rows.append({"layer": l, "norm_ratio": round(float(np.linalg.norm(d) / np.linalg.norm(X, axis=1).mean()), 3),
                     "consistency": round(cons, 3), "heldout_auc": round(float(np.mean(aucs)), 3) if aucs else None})
        dirs.append(d)
    return rows, np.stack(dirs)
