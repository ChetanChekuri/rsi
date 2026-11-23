"""The main model's tools. Everything it does goes through here:

  run_python / run_shell     any code in its workspace (internet, pip, GPU; the sandbox walls are in rsi settings)
  read_file / write_file / list_files
  web_search / arxiv_search / fetch_page
  ollama_create              turn a GGUF it built, or an existing model plus Modelfile lines (parameters, system
                             prompt, template, LoRA adapter), into a runnable model
  ollama_list / ollama_delete
  run_tests                  run one of its test suites on any models, with settings it picks
  switch_model               become one of the models it built (the conversation carries over)
  finish                     end the cycle: summary, plans, pick
"""
import json, os, re, time
from . import research, sandbox, gguf_io, tests


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S, I, B, O = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}, {"type": "object"}
CODE_TOOLS = [
    _fn("run_python", "Run Python code in your workspace (the current directory). numpy, torch, gguf and `from rsi_lib import *` "
        "are available; `pip install` works. Returns stdout/stderr.", {"code": S, "timeout_s": I}, ["code"]),
    _fn("run_shell", "Run a shell command in your workspace (bash). curl, git, pip, ollama (run/show/list) work.", {"command": S, "timeout_s": I}, ["command"]),
    _fn("read_file", "Read a text file from your workspace.", {"path": S}, ["path"]),
    _fn("write_file", "Write a text file in your workspace (creates folders).", {"path": S, "content": S}, ["path", "content"]),
    _fn("list_files", "List files in your workspace (sizes in MB).", {"path": S}, []),
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
        "gets run_python), max_tool_steps, system, temperature, seed, n, judge (for a rubric suite's first run: who grades). A suite with rubric.md (and no answers) is graded strictly by that rubric; the rubric and judge are fixed on first use. Returns scores, fixed/broken vs the baseline with a "
        "sign-test p, and failing excerpts; full transcripts are saved in test_runs/.",
        {"suite": S, "models": {"type": "array", "items": S}, "settings": O}, ["suite"]),
    _fn("switch_model", "Become one of the models you built. Your whole conversation carries over to it and it continues "
        "from here. Only switch when your tests show it is better.", {"model": S, "reason": S}, ["model", "reason"]),
    _fn("finish", "End this cycle: summarise what you did and learned, list 2-4 goals for the next cycle, and pick one.",
        {"summary": S, "next_plans": {"type": "array", "items": S}, "pick": I, "reason": S}, ["summary", "next_plans", "pick", "reason"]),
]


def tool_list(cfg):
    return CODE_TOOLS + (research.TOOLS if cfg.get("research", True) else []) + MODEL_TOOLS


ACTIVITY = {"run_python": "code", "run_shell": "code", "read_file": "code", "write_file": "code", "list_files": "code",
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
        p = os.path.realpath(path if os.path.isabs(path) else os.path.join(self.ws, path))
        if not (p == os.path.realpath(self.ws) or p.startswith(os.path.realpath(self.ws) + os.sep)):
            raise ValueError("only paths inside your workspace are allowed")
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
            elif name in ("web_search", "arxiv_search", "fetch_page") and self.cfg.get("research", True):
                out = research.execute(name, args)
            else:
                out = f"unknown tool {name}"
        except TypeError as e:
            out = f"bad arguments for {name}: {e}"
        except Exception as e:  # noqa: BLE001
            out = f"error: {type(e).__name__}: {e}"
        out = str(out)
        self.log.append({"tool": name, "args": {k: (v[:300] if isinstance(v, str) else v) for k, v in args.items()},
                         "seconds": round(time.time() - t0, 1), "result": out[:1500]})
        return out if len(out) <= 12000 else out[:6000] + f"\n…[{len(out) - 12000} characters cut]…\n" + out[-6000:]

    # ---------------------------------------------------------------- code and files
    def _run(self, argv, timeout_s):
        c = self.cfg; cap = int(c.get("max_run_s") or 0)
        t = int(timeout_s or (cap or 3600)); t = min(t, cap) if cap else t
        rc, out = sandbox.run_in_workspace(self.ws, argv, timeout=t, mem=c.get("sandbox_mem") or None, cpu_quota=c.get("cpu_quota"),
                                           internet=c.get("sandbox_internet", True), local=c.get("sandbox_local", False),
                                           isolate=c.get("sandbox_isolate", True), ollama_url=c["ollama_url"], gate_dir=self.L.gate_dir)
        return f"[exit {rc}]\n{out.strip()[-30000:] or '(no output)'}"

    def t_run_python(self, code, timeout_s=None):
        os.makedirs(os.path.join(self.ws, ".cells"), exist_ok=True)
        path = os.path.join(self.ws, ".cells", f"c{self.k}_{int(time.time() * 1000) % 10**9}.py")
        open(path, "w").write(code)
        return self._run(["python", path], timeout_s)

    def t_run_shell(self, command, timeout_s=None):
        return self._run(["bash", "-c", command], timeout_s)

    def t_read_file(self, path):
        return open(self._inside(path), errors="replace").read()[:40000]

    def t_write_file(self, path, content):
        p = self._inside(path); os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(content)
        return f"wrote {len(content)} characters to {os.path.relpath(p, self.ws)}"

    def t_list_files(self, path="."):
        base = self._inside(path); rows = []
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("box",))
            for f in sorted(files):
                if not f.startswith("."):
                    fp = os.path.join(root, f); rows.append(f"{os.path.relpath(fp, self.ws)}  {os.path.getsize(fp) / 1e6:.1f} MB")
            if len(rows) > 400:
                rows.append("…"); break
        return "\n".join(rows) or "(empty)"

    # ---------------------------------------------------------------- models
    def t_ollama_create(self, name, gguf_path=None, modelfile=None, from_model=None):
        if not gguf_path and not (modelfile or "").strip():
            return "give gguf_path, modelfile lines, or both"
        live = self._live_owned()
        cap = int(self.cfg.get("max_models") or 0)
        if cap and len(live) >= cap:
            return f"refused: you already have {len(live)} models (the limit is {cap}); delete one you don't need first"
        p = self._inside(gguf_path) if gguf_path else None
        if p and not os.path.exists(p):
            return f"no such file: {gguf_path}"
        extra = modelfile or ""
        if re.search(r"^\s*FROM\b", extra, re.M | re.I):
            return "put the weights in gguf_path or from_model, not in a FROM line"
        def fix_adapter(m):
            return f"ADAPTER {self._inside(m.group(1).strip().strip(chr(34)))}"
        extra = re.sub(r"^\s*ADAPTER\s+(.+)$", fix_adapter, extra, flags=re.M | re.I)
        base = self._tag(from_model) if from_model else self.L.tag
        short = re.sub(r"[^a-z0-9._-]", "-", name.lower().split(":")[-1])[:40].strip("-.") or "model"
        tag = f"rsi-{self.cfg['name']}:{short}"
        if tag == self.L.tag:
            return f"{tag} is your current self; pick another name"
        mf = gguf_io.register(tag, p, base, self.L.admin, extra)
        owned = [m for m in self._owned() if m["tag"] != tag] + [{"tag": tag, "from": os.path.relpath(p, self.ws) if p else base,
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
        if tag not in {m["tag"] for m in owned}:
            return f"refused: {tag} was not created by you (you may only delete your own models)"
        judges = {k: v["judge"] for k, v in (self.L.s.read("state/rubrics.json", {}) or {}).items()}
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
        tested = set(self.L.s.read("state/tested.json", []) or []) | set(ms)
        self.L.s.write("state/tested.json", sorted(tested))
        self.L.record_test(self.k, summ)
        out = dict(summ, failing_excerpts={m: tests.failures(res, m, k=2) for m in ms})
        return json.dumps(out, indent=1, default=str)

    def t_switch_model(self, model, reason=""):
        tag = self._tag(model)
        if tag == self.L.tag:
            return "you already are that model"
        if tag not in {m["tag"] for m in self._owned()}:
            return f"refused: {tag} is not a model you built"
        if self.cfg.get("require_tests_before_switch", True) and tag not in set(self.L.s.read("state/tested.json", []) or []):
            return f"refused: run_tests on {tag} first (switching requires a test of the new model)"
        before = self.L.tag
        self.L.switch(tag, reason, self.k)
        self.switches.append({"from": before, "to": tag, "reason": reason, "at": time.strftime("%H:%M")})
        return (f"You are now {tag} (generation {self.L.lineage['current']}). This conversation continues in the new model. "
                f"If it can't carry on the work, the loop switches back to {before} automatically.")

    def t_finish(self, summary, next_plans, pick=0, reason=""):
        plans = [str(p) for p in (next_plans or [])][:4] or ["continue toward the main goal"]
        self.finished = {"summary": summary, "next_plans": plans, "picked": min(max(int(pick or 0), 0), len(plans) - 1), "pick_reason": reason}
        return "cycle finished"
