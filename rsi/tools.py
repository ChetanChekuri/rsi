"""The main model's tools. Everything it does goes through here:

  run_python / run_shell     any code in its workspace (internet, pip, GPU; the sandbox walls are in rsi settings)
  read_file / write_file / list_files
  web_search / arxiv_search / fetch_page
  ollama_create              turn a GGUF it built, or an existing model plus Modelfile lines (parameters, system
                             prompt, template, LoRA adapter), into a runnable model
  ollama_list / ollama_delete
  start_job / job_status / job_log / stop_job / wait    long work in the background (training, big runs)
  run_tests                  run one of its test suites on any models, with settings it picks
  switch_model               become one of the models it built (the conversation carries over)
  finish                     end the cycle: summary, plans, pick
"""
import json, os, re, shlex, shutil, tempfile, time
from . import research, sandbox, gguf_io, goal_eval, llm, promotion, tests, workspace_files


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S, I, B, O = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}, {"type": "object"}
CODE_TOOLS = [
    _fn("run_python", "Run Python code in your workspace (the current directory). numpy, torch, gguf and `from rsi_lib import *` "
        "are available; code is offline (pip can use local wheels). Returns stdout/stderr.", {"code": S, "timeout_s": I}, ["code"]),
    _fn("run_shell", "Run a shell command in your workspace (bash). Code is offline; local files, installed packages and Ollama inference are available.", {"command": S, "timeout_s": I}, ["command"]),
    _fn("read_file", "Read a text file from your workspace (/workspace), e.g. tests/dev/problems.jsonl.", {"path": S}, ["path"]),
    _fn("write_file", "Write a text file in your workspace (/workspace; creates folders). Paths are relative to it, e.g. tests/tools/problems.jsonl.",
        {"path": S, "content": S}, ["path", "content"]),
    _fn("list_files", "List files in your workspace (/workspace), sizes in MB. Omit path for everything.", {"path": S}, []),
    _fn("start_job", "Start a long shell command in the background (training, data generation, big conversions) in the same "
        "sandbox as run_shell, with no time limit. You keep working; when it ends you are told, with its last output lines.",
        {"name": S, "command": S}, ["name", "command"]),
    _fn("start_training", "Train new model weights in the background. Use problems_path for a JSONL curriculum you wrote "
        "(prompt/answer rows), or omit it for fresh local problems. base defaults to the current weights. Set extra_neurons "
        "to widen before training. The output GGUF can be registered and tested when the job finishes.",
        {"name": S, "out": S, "problems_path": S, "base": S, "extra_neurons": I, "layers": {"type": "array", "items": I},
         "kind": S, "size": {"type": "number"}}, ["name", "out"]),
    _fn("job_status", "List your background jobs: running, done, failed, stopped.", {}, []),
    _fn("job_log", "The last lines of a job's output.", {"name": S, "lines": I}, ["name"]),
    _fn("stop_job", "Stop a running background job.", {"name": S}, ["name"]),
    _fn("wait", "Wait (up to `minutes`) until a background job ends or the human writes, instead of polling.", {"minutes": I}, ["minutes"]),
]
MODEL_TOOLS = [
    _fn("ollama_create", "Register a runnable model. Give gguf_path (a GGUF you built in your workspace) and/or modelfile "
        "(Modelfile lines applied on top: PARAMETER, SYSTEM, TEMPLATE, ADAPTER <path to a LoRA GGUF in your workspace>, MESSAGE). "
        "from_model (default: your current self) supplies everything you don't replace. Returns the model's name.",
        {"name": S, "gguf_path": S, "modelfile": S, "from_model": S}, ["name"]),
    _fn("ollama_list", "List the models you created, which one you are now, and your lineage.", {}, []),
    _fn("ollama_delete", "Delete a model you created (frees disk). You cannot delete your current self.", {"name": S}, ["name"]),
    _fn("run_tests", "Run a test suite from your workspace (tests/<name>/ with problems.jsonl and optional grade.py) on one or more "
        "models with the same settings. The first model is the baseline the others are compared with (default: just you). "
        "settings: think (true/false/low/medium/high), num_predict, internet (tested model gets web search), code (tested model "
        "gets run_python), max_tool_steps, system, temperature, top_k, top_p, min_p, presence_penalty, repeat_penalty (unset = the "
        "model's Modelfile values, shown in the result), seed, n, judge (for a rubric suite's first run: who grades). A suite with rubric.md (and no answers) is graded strictly by that rubric; the rubric and judge are fixed on first use. Returns scores, fixed/broken vs the baseline with a "
        "sign-test p, and failing excerpts; full transcripts are saved in test_runs/.",
        {"suite": S, "models": {"type": "array", "items": S}, "settings": O}, ["suite"]),
    _fn("switch_model", "Submit one candidate per cycle for independent evaluation against yourself on fresh, private "
        "goal tasks and per-family regression guards. Infrastructure failures can resume the same submission. If it passes, the conversation continues in that model. Run development tests first to choose a candidate.",
        {"model": S, "reason": S}, ["model", "reason"]),
    _fn("finish", "End this cycle: summarise what you did and learned, list 2-4 goals for the next cycle, and pick one.",
        {"summary": S, "next_plans": {"type": "array", "items": S}, "pick": I, "reason": S}, ["summary", "next_plans", "pick", "reason"]),
]


def tool_list(cfg):
    return CODE_TOOLS + (research.TOOLS if cfg.get("research", False) else []) + MODEL_TOOLS


ACTIVITY = {"run_python": "code", "run_shell": "code", "read_file": "code", "write_file": "code", "list_files": "code", "start_job": "code", "start_training": "code", "job_status": "code",
            "job_log": "code", "stop_job": "code", "wait": "waiting",
            "web_search": "research", "arxiv_search": "research", "fetch_page": "research", "ollama_create": "build",
            "ollama_list": "build", "ollama_delete": "build", "run_tests": "test", "switch_model": "switch", "finish": "plan"}


class Toolbox:
    def __init__(self, loop, k):
        self.L, self.k, self.ws = loop, k, loop.workspace
        self.finished = None; self.log = []; self.switches = []; self.test_runs = []

    @property
    def cfg(self):
        return self.L.cfg

    # ---------------------------------------------------------------- helpers
    def _inside(self, path):
        path = str(path or ".")
        if path == "/workspace" or path.startswith("/workspace/"):   # the usual convention: an alias for the workspace
            path = path[len("/workspace"):].lstrip("/") or "."
        p = os.path.abspath(path if os.path.isabs(path) else os.path.join(self.ws, path))
        if os.path.commonpath([p, self.ws]) != self.ws:
            raise ValueError(f"{path} is outside your workspace (/workspace). Use a path relative to it.")
        return p

    def _owned(self):
        return self.L.s.read("state/models.json", []) or []

    def _tag(self, name):
        """the full tag for a short name, 'self', or a full tag"""
        name = (name or "").strip()
        if name in ("", "self", "current", "me"):
            return self.L.tag
        owned = [m["tag"] for m in self._owned()]
        if name in owned:
            return name
        for t in owned:
            if t.split(":")[-1] == name.split(":")[-1] and (":" not in name or name.startswith("rsi-")):
                return t
        return name

    def call(self, name, args):
        t0 = time.time()
        try:
            fn = getattr(self, "t_" + name, None)
            if fn:
                out = fn(**args)
            elif name in ("web_search", "arxiv_search", "fetch_page") and self.cfg.get("research", False):
                out = research.execute(name, args, allow_local=bool(self.cfg.get("sandbox_local")))
            else:
                out = f"unknown tool {name}"
        except TypeError as e:
            out = f"bad arguments for {name}: {e}"
        except Exception as e:  # noqa: BLE001
            out = f"error: {type(e).__name__}: {e}"
        out = str(out)
        if self.cfg.get("sandbox_isolate", True):   # one name for the workspace everywhere the model looks
            out = out.replace(os.path.realpath(self.ws), "/workspace").replace(self.ws, "/workspace")
        self.log.append({"tool": name, "args": {k: (v[:300] if isinstance(v, str) else v) for k, v in args.items()},
                         "seconds": round(time.time() - t0, 1), "result": out[:1500]})
        return out if len(out) <= 12000 else out[:6000] + f"\n…[{len(out) - 12000} characters cut]…\n" + out[-6000:]

    # ---------------------------------------------------------------- code and files
    def _run(self, argv, timeout_s):
        c = self.cfg; cap = int(c.get("max_run_s") or 0)
        t = int(timeout_s or (cap or 3600)); t = min(t, cap) if cap else t
        rc, out = sandbox.run_in_workspace(self.ws, argv, timeout=t, mem=c.get("sandbox_mem") or None, cpu_quota=c.get("cpu_quota"),
                                           internet=c.get("sandbox_internet", False), local=c.get("sandbox_local", False),
                                           isolate=c.get("sandbox_isolate", True), ollama_url=c["ollama_url"], gate_dir=self.L.gate_dir,
                                           ro=self.L.ro_dirs)
        return f"[exit {rc}]\n{out.strip()[-30000:] or '(no output)'}"

    def t_run_python(self, code, timeout_s=None):
        return self._run(["python", "-c", code], timeout_s)

    def t_run_shell(self, command, timeout_s=None):
        return self._run(["bash", "-c", command], timeout_s)

    def t_read_file(self, path):
        return workspace_files.read(self.ws, self._inside(path), 40000)

    def t_write_file(self, path, content):
        p = self._inside(path)
        workspace_files.write(self.ws, p, content)
        return f"wrote {len(content)} characters to {os.path.relpath(p, self.ws)}"

    def t_list_files(self, path="."):
        base = self._inside(path); rows = []
        if os.path.realpath(base) != base:
            raise ValueError("list_files requires a directory without symbolic links")
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("box",))
            for f in sorted(files):
                if not f.startswith("."):
                    fp = os.path.join(root, f); rows.append(f"{os.path.relpath(fp, self.ws)}  {os.lstat(fp).st_size / 1e6:.1f} MB")
            if len(rows) > 400:
                rows.append("…"); break
        return "\n".join(rows) or "(empty)"

    # ---------------------------------------------------------------- background jobs
    def t_start_job(self, name, command):
        return self.L.jobs.start(name, command, self.k)

    def t_start_training(self, name, out, problems_path=None, base=None, extra_neurons=None, layers=None, kind=None, size=None):
        out = self._inside(out)
        if not out.endswith(".gguf"):
            raise ValueError("training output must have a .gguf extension")
        if os.path.lexists(out):
            raise ValueError("pick a new output file for this training step")
        problems = self._inside(problems_path) if problems_path else None
        base = self._inside(base) if base else self.L.gguf_of(self.L.tag)
        options = {"out": out, "problems": problems, "base": base, "extra": extra_neurons,
                   "layers": layers, "kind": kind, "size": size}
        source = '''import json, os
from rsi_lib import grow_mlp, n_layers, train_step
options = json.loads(ARGS)
os.makedirs(os.path.dirname(options["out"]), exist_ok=True)
path = options.pop("problems")
rows = None
if path:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
if (extra := options.pop("extra")) is not None:
    wide = options["out"] + ".grown.gguf"
    if os.path.lexists(wide):
        raise ValueError("the growth output already exists")
    if options["layers"] is None:
        options["layers"] = [n_layers(options["base"]) - 1]
    grow_mlp(wide, extra=extra, layers=options["layers"], base=options["base"])
    options["base"] = wide
print(json.dumps(train_step(problems=rows, **options), indent=2))
'''
        command = shlex.join(["python", "-u", "-c", "ARGS = " + repr(json.dumps(options)) + "\n" + source])
        return self.L.jobs.start(name, command, self.k)

    def t_job_status(self):
        self.L.jobs.poll()
        t = self.L.jobs.table()
        return json.dumps({n: {k: j.get(k) for k in ("state", "command", "started", "minutes", "exit_code", "log")} for n, j in t.items()},
                          indent=1) if t else "no jobs yet"

    def t_job_log(self, name, lines=60):
        out = self.L.jobs.tail(name, min(int(lines or 60), 400))
        return f"no job named {name}" if out is None else (out or "(no output yet)")

    def t_stop_job(self, name):
        return self.L.jobs.stop(name)

    def t_wait(self, minutes=10):
        if not self.L.jobs.running():
            return "no jobs are running"
        end = time.time() + max(1, min(int(minutes or 10), 24 * 60)) * 60; inbox = len(self.L._inbox())
        self.L.status_activity("waiting", "waiting for " + ", ".join(self.L.jobs.running()))
        while time.time() < end:
            ended = self.L.jobs.poll()
            if ended:
                return "\n\n".join(self.L.jobs.summary(n, j) for n, j in ended)
            if len(self.L._inbox()) > inbox:
                return "the human sent a message (you'll see it next)"
            time.sleep(5)
        return f"still running after {minutes} min: {', '.join(self.L.jobs.running())}"

    # ---------------------------------------------------------------- models
    def t_ollama_create(self, name, gguf_path=None, modelfile=None, from_model=None):
        if not gguf_path and not (modelfile or "").strip():
            return "give gguf_path, modelfile lines, or both"
        tags = tests.model_digests(self.cfg)
        if not tests.digest_for(tags, self.L.tag):
            return "refused: could not verify the current model on the inference server"
        live = [m for m in self._owned() if tests.digest_for(tags, m["tag"])]
        cap = int(self.cfg.get("max_models") or 0)
        if cap and len(live) >= cap:
            return f"refused: you already have {len(live)} models (the limit is {cap}); delete one you don't need first"
        p = self._inside(gguf_path) if gguf_path else None
        extra = modelfile or ""
        if re.search(r"^\s*FROM\b", extra, re.M | re.I):
            return "put the weights in gguf_path or from_model, not in a FROM line"
        base = self._tag(from_model) if from_model else self.L.tag
        llm.model_caps(self.cfg, base, require_local=True)
        short = re.sub(r"[^a-z0-9._-]", "-", name.lower().split(":")[-1])[:40].strip("-.") or "model"
        tag = f"rsi-{self.cfg['name']}:{short}"
        recorded_tags = {m["tag"] for m in self._owned() + self.L.lineage["generations"]}
        admin_url = self.cfg["ollama_admin_url"].rstrip("/")
        admin_tags = tags if admin_url == self.cfg["ollama_url"].rstrip("/") else tests.model_digests({"ollama_url": admin_url})
        if not tests.digest_for(admin_tags, base):
            return "refused: could not verify the source model on the admin server"
        if tests.digest_for(tags, tag) or tests.digest_for(admin_tags, tag) or tag in recorded_tags:
            return f"{tag} already exists or is in the lineage; pick a new name"
        with tempfile.TemporaryDirectory(prefix="register-", dir=self.L.state) as staging:
            def stage(source, suffix):
                with workspace_files.open_file(self.ws, source, "rb") as src, tempfile.NamedTemporaryFile(
                        dir=staging, suffix=suffix, delete=False) as dst:
                    if src.read(4) != b"GGUF":
                        raise ValueError("expected a GGUF file")
                    src.seek(0)
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                    return dst.name

            def fix_adapter(match):
                source = self._inside(match[1].strip().strip(chr(34)))
                return f'ADAPTER "{stage(source, ".gguf")}"'

            staged = stage(p, ".gguf") if p else None
            staged_extra = re.sub(r"^[ \t]*ADAPTER[ \t]+([^\r\n]+)$", fix_adapter, extra, flags=re.M | re.I)
            mf = gguf_io.register(tag, staged, base, self.L.admin, staged_extra)
        digest = tests.digest_for(tests.model_digests(self.cfg), tag)
        if not digest:
            raise RuntimeError("registered model digest is unavailable on the inference server")
        owned = self._owned() + [{"tag": tag, "digest": digest, "from": os.path.relpath(p, self.ws) if p else base,
                                  "modelfile": extra[:2000], "cycle": self.k, "at": time.strftime("%H:%M")}]
        self.L.s.write("state/models.json", owned)
        self.L.note("note", f"registered {tag}")
        shown = re.sub(r'LICENSE """.*?"""', 'LICENSE (…)', mf, flags=re.S)
        return f"registered as {tag} (use this name with run_tests / switch_model / ollama_delete)\nModelfile:\n{shown.strip()[:1500]}"

    def _live_owned(self):
        have = set(tests.model_digests(self.cfg))
        return [m for m in self._owned() if not have or m["tag"] in have]

    def t_ollama_delete(self, name):
        tag = self._tag(name); owned = self._owned()
        if tag == self.L.tag:
            return "refused: that is your current self"
        if self.L.probation and any(g["gen"] == self.L.probation["prev"] and g["tag"] == tag for g in self.L.lineage["generations"]):
            return "refused: that model is the fallback during probation"
        if tag not in {m["tag"] for m in owned}:
            return f"refused: {tag} was not created by you (you may only delete your own models)"
        judges = {k: v["judge"] for k, v in (self.L.s.read("state/rubrics.json", {}) or {}).items()}
        goal_judges = {v["judge"] for v in (self.L.s.read(goal_eval.STATE, {}) or {}).values()}
        if tag in goal_judges:
            return f"refused: {tag} is a frozen goal evaluator"
        if tag in judges.values():
            return f"refused: {tag} is the fixed judge of rubric suite(s) {', '.join(k for k, v in judges.items() if v == tag)}"
        gguf_io.unregister(tag, self.L.admin)
        self.L.s.write("state/models.json", [m for m in owned if m["tag"] != tag])
        return f"deleted {tag}"

    def t_ollama_list(self):
        return json.dumps({"current_self": self.L.tag, "generation": self.L.lineage["current"], "your_models": self._live_owned(),
                           "lineage": self.L.lineage["generations"][-10:]}, indent=1, default=str)

    # ---------------------------------------------------------------- tests and switching
    def t_run_tests(self, suite, models=None, settings=None):
        ms = [self._tag(m) for m in (models or ["self"])]
        if isinstance(settings, str):
            settings = json.loads(settings) if settings.strip() else {}
        self.L.status_activity("test")
        summ, res = tests.run(self.cfg, self.ws, suite, ms, settings or {}, run_id=f"c{self.k}-{len(self.test_runs) + 1}-{time.strftime('%H%M%S')}",
                              gate_dir=self.L.gate_dir, status=lambda d: self.L.status_activity("test", d),
                              lock_path=self.L.s.p("state/rubrics.json"), judge=self.L.tag)
        self.test_runs.append(summ)
        self.L.record_test(self.k, summ)
        out = dict(summ, failing_excerpts={m: tests.failures(res, m, k=2) for m in ms})
        return json.dumps(out, indent=1, default=str)

    def t_switch_model(self, model, reason=""):
        tag = self._tag(model)
        if tag == self.L.tag:
            return "you already are that model"
        record = next((m for m in self._owned() if m["tag"] == tag), None)
        if record is None:
            return f"refused: {tag} is not a model you built"
        if self.L.probation:
            return "refused: the current model is still on probation"
        report = promotion.evaluate(self.L, record, self.k)
        self.L.note("result", "independent evaluation: " + json.dumps(report))
        if not report["accepted"]:
            return "candidate rejected by independent evaluation: " + json.dumps(report)
        before = self.L.tag
        self.L.switch(tag, reason, self.k, record["digest"])
        promotion.activated(self.L, record["digest"])
        self.switches.append({"from": before, "to": tag, "reason": reason, "at": time.strftime("%H:%M")})
        return (f"You are now {tag} (generation {self.L.lineage['current']}). This conversation continues in the new model. "
                f"Call list_files first to pass the tool-use check. If that check fails, the loop switches back to {before}.")

    def t_finish(self, summary, next_plans, pick=0, reason=""):
        plans = [str(p) for p in (next_plans or [])][:4] or ["continue toward the main goal"]
        self.finished = {"summary": summary, "next_plans": plans, "picked": min(max(int(pick or 0), 0), len(plans) - 1), "pick_reason": reason}
        return "cycle finished"
