"""Trace detectors: turn reasoning traces into counts the pilot can reason about.

Every detector returns a number per trace; `summarize` splits those numbers by outcome (correct / wrong-finished /
unfinished), because a behaviour is only a suspect if it is more common in failures than in successes.
`excerpts` cuts trace windows around detected events so the pilot reads the evidence, not whole 8k-token traces.
"""
import re
from collections import Counter
import numpy as np

DOUBT = re.compile(r"^[ \t>*\-#]*(?:\*\*)?(Wait|Hmm+|Actually|Hold on|Hang on|Oh,? wait|But wait|Double[- ]check|Let me (?:double[- ]check|re-?check|verify|re-?read|re-?examine|reconsider)|Re-?check|Re-?reading|Hmm,)\b",
                   re.M | re.I)
ARITH = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/×x÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def answer_pos(text, gold, kind):
    """char offset where the gold answer first appears in `text` (heuristic; -1 if never)"""
    if not text or not gold:
        return -1
    if kind == "choice":
        pats = [rf"(?:answer|option|choice)\s*(?:is|:)?\s*\**\(?{re.escape(gold)}\)?\b", rf"^\s*\**\(?{re.escape(gold)}[\).:]"]
    elif kind == "num":
        pats = [rf"(?<!\d)(?<!\d\.){re.escape(gold)}(?!\.?\d)"]  # 16 matches "= 16." but not 116, 0.16 or 16.5
    else:
        pats = [rf"\b{re.escape(gold)}\b"]
    best = -1
    for p in pats:
        m = re.search(p, text, re.I | re.M)
        if m and (best < 0 or m.start() < best):
            best = m.start()
    return best


def detect(rec):
    th = rec.get("thinking") or ""; ct = rec.get("content") or ""
    ap = answer_pos(th, rec["gold"], rec["kind"])
    doubts = [m.start() for m in DOUBT.finditer(th)]
    words = th.split()
    grams = Counter(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7)))
    lines = Counter(l.strip() for l in th.splitlines() if len(l.strip()) > 20)
    arith_err = 0
    for a, op, b, c in ARITH.findall(th):
        try:
            a, b, c = float(a), float(b), float(c)
            v = {"+": a + b, "-": a - b, "*": a * b, "×": a * b, "x": a * b, "/": a / b if b else None, "÷": a / b if b else None}[op]
            if v is not None and abs(v - c) > 1e-6 * max(1, abs(v)) and abs(round(v, 2) - c) > 0.011:
                arith_err += 1
        except (ValueError, KeyError):
            pass
    unfinished = rec.get("done_reason") == "length"
    return {
        "unfinished": int(unfinished),
        "answer_seen_in_thinking": int(ap >= 0),
        "answer_seen_but_unfinished": int(ap >= 0 and unfinished),
        "answer_seen_final_wrong": int(ap >= 0 and not unfinished and not rec["correct"]),
        "doubt_markers": len(doubts),
        "doubt_after_answer": sum(d > ap for d in doubts) if ap >= 0 else 0,
        "tokens_after_answer_frac": round((len(th) - ap) / max(1, len(th)), 3) if ap >= 0 else 0.0,
        "repeated_ngrams": sum(1 for v in grams.values() if v >= 3),
        "repeated_lines": sum(1 for v in lines.values() if v >= 3),
        "arith_errors": arith_err,
        "missing_answer_line": int(not unfinished and "Answer:" not in ct),
        "non_ascii_burst": int(len(re.findall(r"[一-鿿]", th + ct)) > 20),
        "tokens": rec.get("tokens", 0),
        "_answer_pos": ap, "_doubt_pos": doubts,
    }


def outcome(rec):
    return "unfinished" if rec.get("done_reason") == "length" else ("correct" if rec["correct"] else "wrong")


def summarize(records):
    """table: detector -> {outcome: mean, ...} plus totals; and accuracy by suite"""
    D = [detect(r) for r in records]
    keys = [k for k in D[0] if not k.startswith("_")] if D else []
    groups = {"correct": [], "wrong": [], "unfinished": []}
    for r, d in zip(records, D):
        groups[outcome(r)].append(d)
    table = {}
    for k in keys:
        table[k] = {g: (round(float(np.mean([d[k] for d in v])), 3) if v else None) for g, v in groups.items()}
        table[k]["total"] = round(float(np.sum([d[k] for d in D])), 3)
    by_suite = {}
    for r in records:
        s = by_suite.setdefault(r["suite"], [0, 0, 0]); s[0] += r["correct"]; s[1] += 1; s[2] += r.get("done_reason") == "length"
    acc = {k: {"correct": v[0], "n": v[1], "acc": round(v[0] / v[1], 3), "unfinished": v[2]} for k, v in by_suite.items()}
    doubt_total = sum(len(d["_doubt_pos"]) for d in D)
    doubt_post = sum(d["doubt_after_answer"] for d in D)
    return {"n": len(records), "accuracy_by_suite": acc,
            "counts": {k: len(v) for k, v in groups.items()},
            "detectors_mean_by_outcome": table,
            "doubt_events_total": doubt_total, "doubt_events_after_answer": doubt_post}, D


def excerpts(records, D, k=5, width=700):
    """trace windows around the most diagnostic events of failing traces (wrong or unfinished)"""
    fails = [(r, d) for r, d in zip(records, D) if not r["correct"]]
    fails.sort(key=lambda rd: -(rd[1]["doubt_after_answer"] + 5 * rd[1]["repeated_lines"] + 3 * rd[1]["arith_errors"]))
    out = []
    for r, d in fails[:k]:
        th = r.get("thinking") or ""
        if d["_answer_pos"] >= 0 and d["doubt_after_answer"]:
            c = next(p for p in d["_doubt_pos"] if p > d["_answer_pos"]); why = "first doubt marker after the gold answer appeared"
        elif d["repeated_lines"]:
            c = len(th) // 2; why = "middle of a trace with repeated lines"
        else:
            c = max(0, len(th) - width); why = "end of the trace"
        s = max(0, c - width // 2)
        out.append({"id": r["id"], "outcome": outcome(r), "gold": r["gold"], "pred": r["pred"], "why_here": why,
                    "answer_first_seen_at_char": d["_answer_pos"], "trace_chars": len(th),
                    "excerpt": ("…" if s else "") + th[s:s + width] + ("…" if s + width < len(th) else "")})
    return out
