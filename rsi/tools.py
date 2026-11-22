"""The tools the model uses to work on itself. Everything it does goes through here:

  run_python / run_shell   any code, inside the sandboxed workspace (<folder>/workspace)
  read_file / write_file / list_files   files in the workspace
  web_search / arxiv_search / fetch_page   the web (TinyFish, arXiv)
  ollama_create / ollama_delete / ollama_list   turn a GGUF it built into a runnable model; remove models it made
  practice   solve fresh practice problems with any of its models (to test ideas)
  submit     propose a model as its successor: the loop's own held-out test decides
  finish     end the cycle with a summary and plans for the next one
"""
import json, os, re, time
from . import llm, research, sandbox, gguf_io, detectors


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S, I = {"type": "string"}, {"type": "integer"}
TOOLS = [
    _fn("run_python", "Run Python code in your workspace sandbox (numpy, torch and `from rsi_lib import *` are available). "
        "Returns stdout/stderr. The workspace persists; the network is off.", {"code": S, "timeout_s": I}, ["code"]),
    _fn("run_shell", "Run a shell command in your workspace sandbox.", {"command": S, "timeout_s": I}, ["command"]),
    _fn("read_file", "Read a file from your workspace.", {"path": S}, ["path"]),
    _fn("write_file", "Write a file in your workspace (creates folders).", {"path": S, "content": S}, ["path", "content"]),
    _fn("list_files", "List files in your workspace (sizes in MB).", {"path": S}, []),
    *research.TOOLS,
    _fn("ollama_create", "Register a GGUF file from your workspace as a runnable ollama model. Returns the model's name.",
        {"name": S, "gguf_path": S}, ["name", "gguf_path"]),
    _fn("ollama_delete", "Delete an ollama model that you created (frees disk). You cannot delete your current self.", {"name": S}, ["name"]),
    _fn("ollama_list", "List the models you created, and which one is your current self.", {}, []),
    _fn("practice", "Have a model (your current self or one you created) solve fresh practice problems for the goal. "
        "Returns accuracy, failure counts and the path of the saved traces.", {"model": S, "n": I}, ["model"]),
    _fn("submit", "Propose a model you created as your successor. The loop tests it against your current self on fresh "
        "held-out problems; it replaces you only if it is better without getting worse at general problems. Once per cycle.",
        {"model": S, "note": S}, ["model", "note"]),
    _fn("finish", "End this cycle: summarise what you did and learned, list 2-4 plans for the next cycle, and pick one.",
        {"summary": S, "next_plans": {"type": "array", "items": S}, "pick": I, "reason": S}, ["summary", "next_plans", "pick", "reason"]),
]


class Toolbox:
    def __init__(self, loop, k):
        self.L, self.k, self.cfg = loop, k, loop.cfg
        self.ws = loop.workspace
        self.submitted = self.finished = None; self.n_practice = 0; self.log = []

    # ---------------------------------------------------------------- helpers
    def _inside(self, path):
        p = os.path.realpath(os.path.join(self.ws, path)) if not os.path.isabs(path) else os.path.realpath(path)
        if not (p == os.path.realpath(self.ws) or p.startswith(os.path.realpath(self.ws) + os.sep)):
            raise ValueError("only paths inside your workspace are allowed")
        return p

    def _owned(self):
        return self.L.s.read("state/models.json", []) or []

    def _tag(self, name):
        """accept either the full tag or the short name the model chose"""
        owned = {m["tag"]: m for m in self._owned()}
        if name in owned or name == self.L.tag:
            return name
        for t in owned:
            if t.split(":")[-1] == name.split(":")[-1]:   # "sharp" or "rsi-xyz:sharp"
                return t
        return name

    def call(self, name, args):
        t0 = time.time()
        try:
            out = getattr(self, "t_" + name)(**args) if hasattr(self, "t_" + name) else (
                research.execute(name, args) if name in ("web_search", "arxiv_search", "fetch_page") else f"unknown tool {name}")
        except TypeError as e:
            out = f"bad arguments for {name}: {e}"
        except Exception as e:  # noqa: BLE001
            out = f"error: {type(e).__name__}: {e}"
        out = str(out)
        self.log.append({"tool": name, "args": {k: (v[:300] if isinstance(v, str) else v) for k, v in args.items()},
                         "seconds": round(time.time() - t0, 1), "result": out[:1500]})
        return out if len(out) <= 7000 else out[:3500] + f"\n…[{len(out) - 7000} characters cut]…\n" + out[-3500:]

    # ---------------------------------------------------------------- code and files
    def _run(self, argv, timeout_s):
        c = self.cfg
        rc, out = sandbox.run_in_workspace(self.ws, argv, timeout=min(int(timeout_s or 600), c.get("max_run_s", 1800)),
                                           mem=c.get("sandbox_mem", "4G"), cpu_quota=c.get("cpu_quota"),
                                           network=c.get("sandbox_network", False))
        return f"[exit {rc}]\n{out.strip()[-20000:] or '(no output)'}"

    def t_run_python(self, code, timeout_s=600):
        os.makedirs(os.path.join(self.ws, ".cells"), exist_ok=True)
        path = os.path.join(self.ws, ".cells", f"c{self.k}_{int(time.time() * 1000) % 10**9}.py")
        open(path, "w").write(code)
        return self._run(["python", path], timeout_s)

    def t_run_shell(self, command, timeout_s=300):
        return self._run(["bash", "-c", command], timeout_s)

    def t_read_file(self, path):
        p = self._inside(path)
        return open(p, errors="replace").read()[:20000]

    def t_write_file(self, path, content):
        p = self._inside(path); os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(content)
        return f"wrote {len(content)} characters to {os.path.relpath(p, self.ws)}"

    def t_list_files(self, path="."):
        base = self._inside(path); rows = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if not f.startswith("."):
                    fp = os.path.join(root, f); rows.append(f"{os.path.relpath(fp, self.ws)}  {os.path.getsize(fp) / 1e6:.1f} MB")
            if len(rows) > 300:
                break
        return "\n".join(rows) or "(empty)"

    # ---------------------------------------------------------------- ollama
    def t_ollama_create(self, name, gguf_path):
        p = self._inside(gguf_path)
        if not os.path.exists(p):
            return f"no such file: {gguf_path}"
        short = re.sub(r"[^a-z0-9._-]", "-", name.lower())[:40] or "model"
        tag = f"rsi-{self.cfg['name']}:{short}"
        gguf_io.register(tag, p, self.cfg["master_tag"], self.L.admin)
        owned = [m for m in self._owned() if m["tag"] != tag] + [{"tag": tag, "from": os.path.relpath(p, self.ws), "cycle": self.k,
                                                                    "at": time.strftime("%H:%M")}]
        self.L.s.write("state/models.json", owned)
        return f"registered as {tag} (use this name with practice / submit / ollama_delete)"

    def t_ollama_delete(self, name):
        tag = self._tag(name); owned = self._owned()
        if tag == self.L.tag:
            return "refused: that is your current self"
        if tag not in {m["tag"] for m in owned}:
            return f"refused: {tag} was not created by you (you may only delete your own models)"
        gguf_io.unregister(tag, self.L.admin)
        self.L.s.write("state/models.json", [m for m in owned if m["tag"] != tag])
        return f"deleted {tag}"

    def t_ollama_list(self):
        owned = self._owned()
        return json.dumps({"current_self": self.L.tag, "generation": self.L.lineage["current"], "your_models": owned}, indent=1)

    # ---------------------------------------------------------------- practice, submit, finish
    def t_practice(self, model, n=6):
        tag = self._tag(model)
        if tag != self.L.tag and tag not in {m["tag"] for m in self._owned()}:
            return f"unknown model {model}; use your current self ({self.L.tag}) or one you registered"
        n = max(1, min(int(n or 6), 30)); self.n_practice += 1
        fams = max(1, len(self.L.fac.meta)); per = max(1, -(-n // fams))
        items = self.L.fac.sample("dev", self.k * 100 + self.n_practice, per)[:n]
        path = os.path.join(self.ws, "practice", f"{tag.split(':')[-1]}-c{self.k}-{self.n_practice}.jsonl")
        recs = llm.solve(self.cfg, tag, items, path, who=f"practice, {tag}")
        summ, D = detectors.summarize(recs)
        ex = detectors.excerpts(recs, D, k=2, width=400)
        return json.dumps({"model": tag, "accuracy": round(sum(r["correct"] for r in recs) / len(recs), 3), "n": len(recs),
                           "unfinished": summ["counts"]["unfinished"], "doubt_lines_after_answer": summ["doubt_events_after_answer"],
                           "mean_tokens": round(sum(r["tokens"] for r in recs) / len(recs)), "by_type": summ["accuracy_by_suite"],
                           "failing_excerpts": ex, "traces_file": os.path.relpath(path, self.ws)}, indent=1)

    def t_submit(self, model, note=""):
        if self.submitted:
            return "you already submitted a model this cycle"
        tag = self._tag(model)
        if tag not in {m["tag"] for m in self._owned()}:
            return f"submit a model you registered with ollama_create (got {model})"
        self.submitted = {"model": tag, "note": note, **self.L.held_out_test(self.k, tag)}
        r = self.submitted
        return json.dumps({k: r[k] for k in ("kept", "goal", "general", "verdict")}, indent=1)

    def t_finish(self, summary, next_plans, pick=0, reason=""):
        plans = [str(p) for p in (next_plans or [])][:4] or ["continue"]
        self.finished = {"summary": summary, "next_plans": plans, "picked": min(max(int(pick or 0), 0), len(plans) - 1), "pick_reason": reason}
        return "cycle finished"
