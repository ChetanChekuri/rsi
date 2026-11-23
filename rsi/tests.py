"""Tests the model writes for itself, and running them.

A test suite is a folder in the workspace, tests/<name>/ (a single .jsonl file works too):
  problems.jsonl   one problem per line: {"id": ..., "prompt": ..., "answer": ... (optional), any other fields}
  grade.py         optional: def grade(problem, response) -> float in [0, 1] (or bool). `response` is a dict with
                   "content" (the final reply), "thinking", "tool_calls" ([{name, args, result}]), "done_reason", "tokens".
                   It runs in the sandbox, without internet. Without grade.py, the final answer (the last
                   "Answer: ..." line, else the last line) is compared with "answer" (or any of "answers").
  settings.json    optional default run settings for the suite

Run settings, picked by the model for each run (defaults come from rsi settings):
  think          true / false / "low" / "medium" / "high"
  num_predict    token budget per reply (thinking included)
  internet       the tested model gets web_search and fetch_page
  code           the tested model gets run_python (its own sandbox, with internet only if `internet` is on)
  max_tool_steps how many tool rounds the tested model may use per problem
  system         a system prompt for the tested model
  temperature, seed, n (only the first n problems)
"""
import hashlib, json, math, os, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from . import llm, research, sandbox, probes

RUN_KEYS = ("think", "num_predict", "internet", "code", "max_tool_steps", "system", "temperature", "seed", "n")
CODE_TOOL = {"type": "function", "function": {"name": "run_python", "description": "Run Python code and see its output (stdout/stderr).",
                                              "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
WEB_TOOLS = [t for t in research.TOOLS if t["function"]["name"] in ("web_search", "fetch_page")]


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


# ------------------------------------------------------------------------------------------------ suites
def load_suite(ws, suite):
    """-> (folder or file path, problems, grade.py source or None, suite default settings)"""
    p = suite if os.path.isabs(suite) else os.path.join(ws, suite)
    if not os.path.exists(p) and os.path.exists(os.path.join(ws, "tests", suite)):
        p = os.path.join(ws, "tests", suite)
    p = os.path.realpath(p)
    if not (p == os.path.realpath(ws) or p.startswith(os.path.realpath(ws) + os.sep)):
        raise ValueError("test suites must be inside the workspace")
    if os.path.isdir(p):
        pf, gf, sf = (os.path.join(p, f) for f in ("problems.jsonl", "grade.py", "settings.json"))
    else:
        pf, gf, sf = p, None, None
    if not os.path.exists(pf):
        have = sorted(os.path.relpath(os.path.dirname(f), os.path.join(ws, "tests")) for f in
                      __import__("glob").glob(os.path.join(ws, "tests", "**", "problems.jsonl"), recursive=True))
        raise ValueError(f"no test suite at {os.path.relpath(p, ws)}. Suites you have: {', '.join(have) or 'none yet'}. A suite is "
                         "a folder tests/<name>/ with problems.jsonl (one JSON object per line with id, prompt, answer)")
    problems = []
    for i, l in enumerate(open(pf)):
        if l.strip():
            try:
                d = json.loads(l)
            except ValueError:
                raise ValueError(f"line {i + 1} of {os.path.relpath(pf, ws)} is not valid JSON")
            d.setdefault("id", f"p{i}")
            if "prompt" not in d:
                raise ValueError(f"problem {d['id']} has no prompt")
            problems.append(d)
    grade = open(gf).read() if gf and os.path.exists(gf) else None
    try:
        defaults = json.load(open(sf)) if sf and os.path.exists(sf) else {}
    except ValueError:
        raise ValueError(f"{os.path.relpath(sf, ws)} must be one JSON object, e.g. {{\"think\": false, \"num_predict\": 2000}}")
    if grade is None and not all(("answer" in d or "answers" in d) for d in problems):
        raise ValueError("without grade.py every problem needs an answer (or answers)")
    return p, problems, grade, defaults


def default_grade(problem, response):
    pred = probes.extract(response.get("content", ""))
    p = re.sub(r"[\s*$\\,]", " ", pred).strip().lower()
    for a in problem.get("answers") or [problem.get("answer")]:
        a = str(a).strip().lower().replace(",", "")
        if re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", a):
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", p)
            if nums and any(abs(float(x) - float(a)) <= 1e-6 * max(1, abs(float(a))) for x in (nums[0], nums[-1])):
                return 1.0
        elif re.fullmatch(r"[a-e]", a):
            m = re.findall(r"\b([a-e])\b", p)
            if m and m[0] == a:
                return 1.0
        elif p == a or re.search(rf"(?<!\w){re.escape(a)}(?!\w)", p):
            return 1.0
    return 0.0


def write_library(ws, n=200):
    """ready-made suites the model may use: GSM8K and ARC-Challenge test problems (as a forgetting check, say)"""
    lib = os.path.join(ws, "tests", "library")
    pool = probes.general_pool()
    for name, prefix in (("gsm8k", "general:gsm8k"), ("arc_challenge", "general:arc")):
        d = os.path.join(lib, name)
        if os.path.exists(os.path.join(d, "problems.jsonl")):
            continue
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "problems.jsonl"), "w") as f:
            for it in [x for x in pool if x["suite"] == prefix][:n]:
                f.write(json.dumps({"id": it["id"], "prompt": it["prompt"], "answer": it["gold"]}) + "\n")
    readme = os.path.join(lib, "README.md")
    if not os.path.exists(readme):
        open(readme, "w").write("Ready-made suites (200 problems each, answers included): gsm8k (grade-school math), "
                                "arc_challenge (science multiple choice). Use them to check that a change doesn't make you "
                                "worse at things outside the goal. Your own suites go in tests/<name>/.\n")


# ------------------------------------------------------------------------------------------------ running
def model_digests(cfg):
    try:
        with urllib.request.urlopen(cfg["ollama_url"] + "/api/tags", timeout=20) as r:
            return {m["name"]: m["digest"] for m in json.loads(r.read())["models"]}
    except Exception:  # noqa: BLE001
        return {}


def attempt(cfg, model, problem, opts, box, ws, gate_dir, slot):
    """the tested model answers one problem, with the tools the run settings allow"""
    tools = ([CODE_TOOL] if opts.get("code") else []) + (WEB_TOOLS if opts.get("internet") else [])
    msgs = ([{"role": "system", "content": opts["system"]}] if opts.get("system") else []) + [{"role": "user", "content": problem["prompt"]}]
    calls, tokens, t0, r = [], 0, time.time(), None
    extra = {k: opts[k] for k in ("temperature",) if opts.get(k) is not None}
    for step in range(int(opts.get("max_tool_steps", 6)) + 1):
        r = llm.chat(cfg, model, msgs, think=opts.get("think", True), tools=tools or None, num_predict=opts.get("num_predict"),
                     seed=int(opts.get("seed", 0)), options=extra,
                     live={"slot": slot, "title": f"test: {model} on {problem['id']}", "prompt": problem["prompt"][:1500],
                           "gold": str(problem.get("answer", ""))[:200]})
        m = r["message"]; tokens += r.get("eval_count", 0) or 0
        tc = m.get("tool_calls") or []
        if not tc or step == int(opts.get("max_tool_steps", 6)):
            break
        msgs.append({"role": "assistant", "content": m.get("content", ""), "tool_calls": tc})
        for c in tc[:4]:
            fn = c.get("function", {}); name = fn.get("name", ""); args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if name == "run_python" and opts.get("code"):
                d = os.path.join(box, re.sub(r"\W", "_", str(problem["id"]))); os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"cell{len(calls)}.py"); open(path, "w").write(str(args.get("code", "")))
                rc, out = sandbox.run_in_workspace(d, ["python", path], timeout=120, mem="2G", internet=bool(opts.get("internet")),
                                                   ollama_url=cfg["ollama_url"], gate_dir=gate_dir)
                out = f"[exit {rc}]\n{out.strip()[-4000:]}"
            elif name in ("web_search", "fetch_page") and opts.get("internet"):
                out = research.execute(name, args)[:4000]
            else:
                out = f"tool {name} is not available"
            calls.append({"name": name, "args": args, "result": out[:2000]})
            msgs.append({"role": "tool", "content": out, "tool_name": name})
    m = r["message"]
    return {"id": problem["id"], "model": model, "prompt": problem["prompt"], "answer": problem.get("answer"), "content": m.get("content", ""), "thinking": m.get("thinking", ""), "tool_calls": calls,
            "done_reason": r.get("done_reason"), "tokens": tokens, "seconds": round(time.time() - t0, 1)}


def grade_all(cfg, ws, suite_path, grade_src, problems, records, gate_dir):
    """scores for records (list of dicts); grade.py runs in the sandbox without internet"""
    byid = {p["id"]: p for p in problems}
    if not grade_src:
        return [default_grade(byid[r["id"]], r) for r in records]
    d = os.path.join(ws, ".grading"); os.makedirs(d, exist_ok=True)
    tag = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
    inp, out = os.path.join(d, f"in-{tag}.jsonl"), os.path.join(d, f"out-{tag}.json")
    with open(inp, "w") as f:
        for r in records:
            f.write(json.dumps({"problem": byid[r["id"]], "response": r}) + "\n")
    gpath = os.path.join(suite_path, "grade.py")
    rc, log = sandbox.run_in_workspace(ws, ["python", "-m", "rsi.grade", gpath, inp, out], timeout=900, mem=cfg.get("sandbox_mem") or "4G",
                                       internet=False, ollama_url=cfg["ollama_url"], gate_dir=gate_dir)
    try:
        scores = json.load(open(out))
    except (OSError, ValueError):
        raise RuntimeError(f"grade.py failed: {log.strip()[-1500:]}")
    for f in (inp, out):
        try:
            os.unlink(f)
        except OSError:
            pass
    return scores


def run(cfg, ws, suite, models, settings=None, run_id=None, gate_dir=None, status=None):
    """run a suite on several models with the same settings; the first model is the baseline for comparisons"""
    path, problems, grade_src, defaults = load_suite(ws, suite)
    opts = {"think": cfg.get("test_think", True), "num_predict": cfg.get("test_num_predict") or cfg.get("num_predict", 4096),
            "internet": False, "code": False, "max_tool_steps": cfg.get("test_max_tool_steps", 6), "seed": 0}
    opts.update({k: v for k, v in defaults.items() if k in RUN_KEYS})
    opts.update({k: v for k, v in (settings or {}).items() if k in RUN_KEYS and v is not None})
    if opts.get("n"):
        problems = problems[:int(opts["n"])]
    digests = model_digests(cfg)
    for m in models:
        if digests and m not in digests and m + ":latest" not in digests:
            raise ValueError(f"no model named {m} on the ollama server")
    run_id = run_id or time.strftime("run-%m%d-%H%M%S-") + os.urandom(2).hex()
    out_dir = os.path.join(ws, "test_runs", run_id); os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(ws, "test_runs", ".cache"); os.makedirs(cache_dir, exist_ok=True)
    suite_key = hashlib.sha1((json.dumps(problems, sort_keys=True) + (grade_src or "") +
                              json.dumps({k: opts.get(k) for k in RUN_KEYS if k != "n"}, sort_keys=True)).encode()).hexdigest()
    results = {}
    for m in models:
        key = hashlib.sha1((suite_key + m + digests.get(m, digests.get(m + ":latest", m))).encode()).hexdigest()[:20]
        cache = os.path.join(cache_dir, key + ".jsonl")
        done = {}
        if os.path.exists(cache):
            for l in open(cache):
                r = json.loads(l); done[r["id"]] = r
        todo = [p for p in problems if p["id"] not in done]
        if status:
            status(f"testing {m} on {len(problems)} problems from {os.path.basename(path)}" + (f" ({len(problems) - len(todo)} cached)" if done else ""))
        box = os.path.join(out_dir, "box", re.sub(r"\W", "_", m))
        lock = threading.Lock()

        def one(p, m=m, box=box):
            slot = "subject-" + threading.current_thread().name.split("_")[-1]
            return attempt(cfg, m, p, opts, box, ws, gate_dir, slot)
        with open(cache, "a") as f, ThreadPoolExecutor(max(1, int(cfg.get("test_workers") or cfg.get("workers") or 1))) as ex:
            for rec in ex.map(one, todo):
                with lock:
                    f.write(json.dumps(rec) + "\n"); f.flush(); done[rec["id"]] = rec
        llm.unload(cfg, m)
        recs = [done[p["id"]] for p in problems]
        for r, s in zip(recs, grade_all(cfg, ws, path, grade_src, problems, recs, gate_dir)):
            r["score"] = float(s)
        with open(os.path.join(out_dir, re.sub(r"[^\w.-]", "_", m) + ".jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        results[m] = recs
    summary = summarize(results, models)
    summary.update(run_id=run_id, suite=os.path.relpath(path, ws), n=len(problems), settings=opts,
                   files=os.path.relpath(out_dir, ws), at=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summary, results


def summarize(results, models):
    base = models[0]; B = {r["id"]: r for r in results[base]}; rows = {}
    for m in models:
        R = results[m]
        row = {"score": round(sum(r["score"] for r in R) / max(1, len(R)), 3), "finished": round(sum(r["done_reason"] != "length" for r in R) / max(1, len(R)), 2),
               "mean_tokens": round(sum(r["tokens"] for r in R) / max(1, len(R))), "tool_calls": sum(len(r["tool_calls"]) for r in R)}
        if m != base:
            ids = [r["id"] for r in R if r["id"] in B]
            A_ = {r["id"]: r for r in R}
            fixed = sum(A_[i]["score"] >= 0.5 > B[i]["score"] for i in ids); broken = sum(B[i]["score"] >= 0.5 > A_[i]["score"] for i in ids)
            row.update(vs_baseline={"baseline": base, "fixed": fixed, "broken": broken, "p": round(sign_p(fixed, broken), 4)})
        rows[m] = row
    return {"baseline": base, "models": rows}


def failures(results, model, k=3, width=500):
    """short excerpts of failed attempts, for the model to read"""
    out = []
    for r in results.get(model, []):
        if r["score"] < 0.5 and len(out) < k:
            out.append({"id": r["id"], "done_reason": r["done_reason"], "tokens": r["tokens"],
                        "thinking_end": (r.get("thinking") or "")[-width:], "reply_end": (r.get("content") or "")[-width:]})
    return out
