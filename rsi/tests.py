"""Tests the model writes for itself, and running them.

A test suite is a folder in the workspace, tests/<name>/ (a single .jsonl file works too):
  problems.jsonl   one problem per line: {"id": ..., "prompt": ..., "answer": ... (optional), any other fields}
                   Tool-calling problems can add "tools" (function schemas the tested model is given), "tool_results"
                   ({tool name: canned result}, default {"status": "ok"}) and "messages" (a whole conversation to
                   continue instead of a single prompt). grade.py then checks response["tool_calls"].
  grade.py         optional: def grade(problem, response) -> float in [0, 1] (or bool). `response` is a dict with
                   "content" (the final reply), "thinking", "tool_calls" ([{name, args, result}]), "done_reason", "tokens".
                   It runs in the sandbox, without internet. Without grade.py, the final answer (the last
                   "Answer: ..." line, else the last line) is compared with "answer" (or any of "answers").
  rubric.md        for goals with no checkable answer (writing, design, advice, ...): a grading scale written BEFORE
                   the first run. A judge model scores every response strictly by it, criterion by criterion, without
                   knowing which model wrote it. Once a rubric has been used it is frozen (as is its judge): changing
                   it means making a new suite, so scores stay comparable. Problems may add "reference" (notes for the
                   judge). Precedence: grade.py, then rubric.md, then answers.
  settings.json    optional default run settings for the suite

Run settings, picked by the model for each run (defaults come from rsi settings):
  think          true / false / "low" / "medium" / "high"
  num_predict    token budget per reply (thinking included)
  internet       the tested model gets web_search and fetch_page
  code           the tested model gets run_python (its own offline sandbox; `internet` enables only Tinyfish research tools)
  max_tool_steps how many tool rounds the tested model may use per problem
  system         a system prompt for the tested model
  temperature, top_k, top_p, min_p, presence_penalty, repeat_penalty (unset = the model's own Modelfile values)
  seed, n (only the first n problems)
"""
import hashlib, json, math, os, re, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from . import llm, network, research, sandbox, workspace_files

SAMPLING = ("temperature", "top_k", "top_p", "min_p", "presence_penalty", "repeat_penalty")   # override the Modelfile's
RUN_KEYS = ("think", "num_predict", "internet", "code", "max_tool_steps", "system", "seed", "n") + SAMPLING
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
    problems, ids = [], set()
    for i, l in enumerate(workspace_files.read(ws, pf).splitlines()):
        if l.strip():
            try:
                d = json.loads(l)
            except ValueError:
                raise ValueError(f"line {i + 1} of {os.path.relpath(pf, ws)} is not valid JSON")
            if not isinstance(d, dict):
                raise ValueError(f"line {i + 1} must be a JSON object")
            d.setdefault("id", f"p{i}")
            if not isinstance(d["id"], (str, int)) or d["id"] in ids:
                raise ValueError(f"invalid or duplicate problem id on line {i + 1}")
            ids.add(d["id"])
            if not isinstance(d.get("prompt"), str) or not d["prompt"].strip():
                raise ValueError(f"problem {d['id']} has no prompt")
            problems.append(d)
    if not problems:
        raise ValueError("a test suite must contain at least one problem")
    grade = workspace_files.read(ws, gf) if gf and os.path.lexists(gf) else None
    rf = os.path.join(p, "rubric.md") if os.path.isdir(p) else None
    if grade is None and rf and os.path.lexists(rf):
        grade = {"rubric": workspace_files.read(ws, rf)}
    try:
        defaults = json.loads(workspace_files.read(ws, sf)) if sf and os.path.lexists(sf) else {}
        if not isinstance(defaults, dict):
            raise ValueError("settings must be an object")
    except ValueError:
        raise ValueError(f"{os.path.relpath(sf, ws)} must be one JSON object, e.g. {{\"think\": false, \"num_predict\": 2000}}")
    if grade is None and not all(("answer" in d or "answers" in d) for d in problems):
        raise ValueError("every problem needs an answer, unless the suite has a grade.py or a rubric.md (for goals without a checkable answer)")
    return p, problems, grade, defaults


def committed_answer(content):
    lines = (content or "").strip().splitlines()
    if not lines:
        return None
    match = re.fullmatch(r"answer:\s*(.+)", lines[-1].strip().strip("*"), re.I)
    return match[1].strip() if match else None


NUMBER = re.compile(r"[+-]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[+-]?\d+)?", re.I)


def normalize_answer(value):
    answer = str(value).strip().strip("*$` ").rstrip(".").strip().casefold()
    if NUMBER.fullmatch(answer):
        return Decimal(answer.replace(",", "").replace("$", ""))
    return " ".join(answer.split())


def default_grade(problem, response):
    content = (response.get("content") or "").strip()
    if not content:
        return 0.0
    pred = committed_answer(content) or content.splitlines()[-1]
    return float(any(normalize_answer(pred) == normalize_answer(answer)
                     for answer in problem.get("answers") or [problem.get("answer")]))


# ------------------------------------------------------------------------------------------------ running
def model_digests(cfg):
    try:
        with network.open_local(cfg["ollama_url"] + "/api/tags", timeout=20) as r:
            return {m["name"]: m["digest"] for m in json.loads(r.read())["models"]}
    except Exception:  # noqa: BLE001
        return {}


def digest_for(tags, model):
    return tags.get(model) or tags.get(model + ":latest")


def modelfile_params(cfg, model):
    """the sampling parameters baked into a model's Modelfile (they apply unless a run overrides them)"""
    try:
        req = urllib.request.Request(cfg["ollama_url"] + "/api/show", json.dumps({"model": model}).encode(), {"Content-Type": "application/json"})
        txt = json.loads(network.open_local(req, timeout=30).read()).get("parameters", "")
        return {l.split()[0]: l.split()[1] for l in txt.splitlines() if len(l.split()) >= 2}
    except Exception:  # noqa: BLE001
        return {}


def attempt(cfg, model, problem, opts, box, ws, gate_dir, slot):
    """the tested model answers one problem, with the tools the run settings allow"""
    own = [t if "function" in t else {"type": "function", "function": t} for t in problem.get("tools") or []]
    tools = own + ([CODE_TOOL] if opts.get("code") else []) + (WEB_TOOLS if opts.get("internet") else [])
    own_names = {t["function"]["name"] for t in own}
    msgs = ([{"role": "system", "content": opts["system"]}] if opts.get("system") else []) + (
        problem["messages"] if problem.get("messages") else [{"role": "user", "content": problem["prompt"]}])
    calls, tokens, t0, r = [], 0, time.time(), None
    extra = {k: opts[k] for k in SAMPLING if opts.get(k) is not None}
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
            if not isinstance(args, dict):
                args = {}
            if name == "run_python" and opts.get("code"):
                with tempfile.TemporaryDirectory(prefix="rsi-test-code-") as d:
                    rc, out = sandbox.run_in_workspace(d, ["python", "-c", str(args.get("code", ""))], timeout=120,
                                                       mem="2G", internet=False,
                                                       ollama_url=cfg["ollama_url"], gate_dir=gate_dir)
                out = f"[exit {rc}]\n{out.strip()[-4000:]}"
            elif name in own_names:   # the problem's own tools answer with canned results
                res = (problem.get("tool_results") or {}).get(name, '{"status": "ok"}')
                out = res if isinstance(res, str) else json.dumps(res)
            elif name in ("web_search", "fetch_page") and opts.get("internet"):
                out = research.execute(name, args)[:4000]
            else:
                out = f"tool {name} is not available"
            calls.append({"name": name, "args": args, "result": out[:2000]})
            msgs.append({"role": "tool", "content": out, "tool_name": name})
        for c in tc[4:]:
            msgs.append({"role": "tool", "content": "not run: at most 4 tool calls per reply", "tool_name": c.get("function", {}).get("name", "")})
    m = r["message"]
    return {"id": problem["id"], "model": model, "prompt": problem["prompt"], "answer": problem.get("answer"), "content": m.get("content", ""), "thinking": m.get("thinking", ""), "tool_calls": calls,
            "done_reason": r.get("done_reason"), "tokens": tokens, "seconds": round(time.time() - t0, 1)}


JUDGE_SCHEMA = {"type": "object", "properties": {
    "criteria": {"type": "array", "items": {"type": "object", "properties": {
        "criterion": {"type": "string"}, "evidence": {"type": "string"}, "points": {"type": "number"}, "max_points": {"type": "number"}},
        "required": ["criterion", "evidence", "points", "max_points"]}}}, "required": ["criteria"]}


def judge_one(cfg, judge, rubric, problem, response):
    """strict rubric grading of one response by the judge model; returns (score 0..1, per-criterion detail)"""
    tools = "".join(f"\n- called {c['name']}({json.dumps(c['args'])[:300]}) -> {c['result'][:300]}" for c in response.get("tool_calls") or [])
    prompt = ("You are a strict grader. Grade the RESPONSE to the TASK using only the RUBRIC. Go through every criterion in "
              "the rubric, in order. For each, quote or point to the evidence in the response, then award points exactly as "
              "the rubric's scale says. Apply the scale literally: when the response falls between two levels, give the lower "
              "one; give no credit for anything the rubric doesn't reward; length, confidence and polish earn nothing unless "
              "the rubric says so. You do not know who wrote the response.\n\n"
              f"# RUBRIC\n{rubric}\n\n# TASK\n{problem['prompt']}\n"
              + (f"\n# NOTES FOR THE GRADER\n{problem['reference']}\n" if problem.get("reference") else "")
              + f"\n# RESPONSE\n{response.get('content', '')}\n" + (f"\n# TOOLS THE RESPONDER USED{tools}\n" if tools else "")
              + "\nAnswer in JSON: one entry per rubric criterion.")
    d = None
    for think in (True, False):   # if thinking eats the whole budget, ask again without it
        r = llm.chat(cfg, judge, [{"role": "user", "content": prompt}], think=think, fmt=JUDGE_SCHEMA,
                     num_predict=max(8192, int(cfg.get("test_num_predict") or 0)), options={"temperature": 0},
                     live={"slot": "judge", "title": f"grading {problem['id']} by the rubric ({judge})", "prompt": problem["prompt"][:1500]})
        try:
            d = json.loads(r["message"]["content"])
            if d.get("criteria"):
                break
        except ValueError:
            pass
    if not d or not d.get("criteria"):
        raise RuntimeError(f"the judge {judge} gave no usable verdict for {problem['id']}")
    crit = [dict(c, points=min(max(float(c["points"]), 0.0), float(c["max_points"]))) for c in d["criteria"] if float(c.get("max_points") or 0) > 0]
    if not crit:
        raise ValueError("the judge returned no criteria")
    return sum(c["points"] for c in crit) / sum(c["max_points"] for c in crit), crit


def grade_all(cfg, ws, suite_path, grade_src, problems, records, gate_dir, judge=None):
    """scores for records (list of dicts); grade.py runs in the sandbox without internet; a rubric is applied by `judge`"""
    byid = {p["id"]: p for p in problems}
    if isinstance(grade_src, dict):
        out = []
        for r in records:
            if "rubric_score" not in r or r.get("judge") != judge:
                r["rubric_score"], r["rubric_detail"] = judge_one(cfg, judge, grade_src["rubric"], byid[r["id"]], r); r["judge"] = judge
            out.append(r["rubric_score"])
        llm.unload(cfg, judge)
        return out
    if not grade_src:
        return [default_grade(byid[r["id"]], r) for r in records]
    tag = os.urandom(8).hex()
    d = os.path.join(ws, ".grading", tag)
    gpath, inp, out = (os.path.join(d, name) for name in ("grade.py", "input.jsonl", "scores.json"))
    workspace_files.write(ws, gpath, grade_src)
    workspace_files.write(ws, inp, "".join(json.dumps({"problem": byid[r["id"]], "response": r}) + "\n" for r in records))
    # Keep suite-local imports working while executing the captured grader source.
    argv = ["python", "-m", "rsi.grade", gpath, inp, out, suite_path]
    log = ""
    try:
        rc, log = sandbox.run_in_workspace(ws, argv, timeout=900, mem=cfg.get("sandbox_mem") or "4G",
                                           internet=False, ollama_url=cfg["ollama_url"], gate_dir=gate_dir)
        scores = json.loads(workspace_files.read(ws, out)) if rc == 0 else None
        if not isinstance(scores, list) or len(scores) != len(records) or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
            raise ValueError("grader must produce one finite score in [0, 1] per record")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"grade.py failed: {exc}. {log[-1500:]}") from exc
    finally:
        for path in (inp, out, gpath):
            try:
                workspace_files.unlink(ws, path)
            except OSError:
                pass
    return scores


def rubric_lock(lock_path, ws, path, grade_src, judge, digests):
    """a rubric (and its judge) is fixed the first time it is used; returns the judge to use"""
    if not isinstance(grade_src, dict):
        return None
    locks = json.load(open(lock_path)) if lock_path and os.path.exists(lock_path) else {}
    key = os.path.relpath(path, ws); h = hashlib.sha1(grade_src["rubric"].encode()).hexdigest()
    if key in locks:
        if locks[key]["sha1"] != h:
            raise ValueError(f"{key}/rubric.md changed after it was first used ({locks[key]['at']}). Rubrics are fixed so scores stay "
                             "comparable: put the new rubric in a new suite (copy the problems) instead.")
        fixed_judge = locks[key]["judge"]
        digest = digest_for(digests, fixed_judge)
        if not digest or locks[key].get("digest", digest) != digest:
            raise ValueError(f"the rubric judge for {key} changed or is unavailable")
        judge = fixed_judge
    digest = digest_for(digests, judge)
    if not digest:
        raise ValueError(f"cannot verify the rubric judge {judge}")
    locks[key] = {"sha1": h, "judge": judge, "digest": digest, "at": time.strftime("%Y-%m-%d %H:%M")}
    if lock_path:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        json.dump(locks, open(lock_path, "w"), indent=1)
    return judge


def run(cfg, ws, suite, models, settings=None, run_id=None, gate_dir=None, status=None, lock_path=None, judge=None):
    """run a suite on several models with the same settings; the first model is the baseline for comparisons.
    For a rubric suite, `judge` (or settings["judge"], or the judge_model setting) grades; it is fixed on first use."""
    path, problems, grade_src, defaults = load_suite(ws, suite)
    digests = model_digests(cfg)
    judge = rubric_lock(lock_path, ws, path, grade_src,
                        (settings or {}).get("judge") or cfg.get("judge_model") or judge or models[0], digests)
    opts = {"think": cfg.get("test_think", True), "num_predict": cfg.get("test_num_predict") or cfg.get("num_predict", 4096),
            "internet": False, "code": False, "max_tool_steps": cfg.get("test_max_tool_steps", 6), "seed": 0}
    opts.update({k: v for k, v in defaults.items() if k in RUN_KEYS})
    opts.update({k: v for k, v in (settings or {}).items() if k in RUN_KEYS and v is not None})
    if opts.get("n") is not None:
        n = int(opts["n"])
        if n < 1:
            raise ValueError("n must be at least 1")
        problems = problems[:n]
    if not digests:
        raise RuntimeError("cannot verify model digests on the inference server")
    if not cfg.get("research", False):
        opts["internet"] = False
    for m in models:
        if m not in digests and m + ":latest" not in digests:
            raise ValueError(f"no model named {m} on the ollama server")
    run_id = run_id or time.strftime("run-%m%d-%H%M%S-") + os.urandom(2).hex()
    out_dir = os.path.join(ws, "test_runs", run_id)
    cache_dir = os.path.join(ws, "test_runs", ".cache")
    suite_key = hashlib.sha256(json.dumps([2, problems, grade_src, opts, cfg["ollama_url"], cfg.get("num_ctx"),
                                          digest_for(digests, judge) if judge else None], sort_keys=True).encode()).hexdigest()
    results = {}
    for m in models:
        key = hashlib.sha256((suite_key + m + digest_for(digests, m)).encode()).hexdigest()[:24]
        cache = os.path.join(cache_dir, key + ".jsonl")
        done = {}
        if os.path.lexists(cache):
            for l in workspace_files.read(ws, cache).splitlines():
                r = json.loads(l); done[r["id"]] = r
        todo = [p for p in problems if p["id"] not in done]
        if status:
            status(f"testing {m} on {len(problems)} problems from {os.path.basename(path)}" + (f" ({len(problems) - len(todo)} cached)" if done else ""))
        box = os.path.join(out_dir, "box", re.sub(r"\W", "_", m))
        lock = threading.Lock()

        def one(p, m=m, box=box):
            slot = "subject-" + threading.current_thread().name.split("_")[-1]
            return attempt(cfg, m, p, opts, box, ws, gate_dir, slot)
        with workspace_files.open_file(ws, cache, "a") as f, ThreadPoolExecutor(max(1, int(cfg.get("test_workers") or cfg.get("workers") or 1))) as ex:
            for rec in ex.map(one, todo):
                with lock:
                    f.write(json.dumps(rec) + "\n"); f.flush(); done[rec["id"]] = rec
        llm.unload(cfg, m)
        recs = [done[p["id"]] for p in problems]
        for r, s in zip(recs, grade_all(cfg, ws, path, grade_src, problems, recs, gate_dir, judge)):
            r["score"] = float(s)
        if isinstance(grade_src, dict):   # keep the judge's verdicts so unchanged answers aren't judged again
            with workspace_files.open_file(ws, cache, "w") as f:
                for r in done.values():
                    f.write(json.dumps(r) + "\n")
        with workspace_files.open_file(ws, os.path.join(out_dir, re.sub(r"[^\w.-]", "_", m) + ".jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        results[m] = recs
    summary = summarize(results, models)
    summary.update(model_sampling_defaults={m: modelfile_params(cfg, m) for m in models})
    summary.update(run_id=run_id, suite=os.path.relpath(path, ws), n=len(problems), settings=opts,
                   model_digests={m: digest_for(digests, m) for m in models}, graded_by=(
                   f"rubric, judged by {judge}" if judge else ("grade.py" if grade_src else "answers")),
                   files=os.path.relpath(out_dir, ws), at=time.strftime("%Y-%m-%d %H:%M"))
    workspace_files.write(ws, os.path.join(out_dir, "summary.json"), json.dumps(summary, indent=1))
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
