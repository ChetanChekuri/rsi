"""The RSI loop. You give one main prompt (the goal); after that the model runs itself, one cycle after another, in a
single conversation that carries across cycles (and across models, when it switches to a new version of itself).

In a cycle the model decides everything: it writes tests for the goal (problems + a grader) and picks how they run
(thinking, token budget, whether the tested model gets internet or code), tests itself, researches, changes itself
any way it can with code (weights, quantization, adapters, Modelfile parameters, training), registers the results
with ollama, tests them against itself, and either switches to a better one (the conversation moves into it) or
stays. It ends the cycle with plans for the next one and picks one.

Anything you type reaches the model at its next step, and overrides its pick for the next cycle.
State lives in <folder>/.rsi/, the model's files in <folder>/workspace/. Settings: `rsi settings`.
"""
import json, os, shutil, signal, subprocess, sys, time, traceback
from . import llm, gguf_io, settings, tests
from .tools import Toolbox, tool_list, ACTIVITY
from .jobs import Jobs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYBOOK = os.path.join(ROOT, "rsi", "playbook.md")
PLANS = {"type": "object", "properties": {"summary": {"type": "string"}, "plans": {"type": "array", "items": {"type": "string"}},
                                          "pick": {"type": "integer"}, "reason": {"type": "string"}}, "required": ["summary", "plans", "pick", "reason"]}


# ------------------------------------------------------------------------------------------------ session (a folder)
class Session:
    def __init__(self, folder):
        self.folder = os.path.abspath(folder); self.dir = os.path.join(self.folder, ".rsi")

    def p(self, *a):
        return os.path.join(self.dir, *a)

    def exists(self):
        return os.path.exists(self.p("config.json"))

    @property
    def cfg(self):
        return json.load(open(self.p("config.json")))

    def read(self, name, default=None):
        try:
            return json.load(open(self.p(name)))
        except (OSError, ValueError):
            return default

    def write(self, name, obj):
        tmp = self.p("." + os.path.basename(name) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1, default=str)
        os.replace(tmp, self.p(name))

    def history(self):
        try:
            return [json.loads(l) for l in open(self.p("history.jsonl")) if l.strip()]
        except OSError:
            return []

    def post(self, text):
        """a message from the human: '' = run the next cycle, text = run it with this direction"""
        with open(self.p("inbox.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "text": text}) + "\n")

    def pid(self):
        try:
            pid = int(open(self.p("loop.pid")).read()); os.kill(pid, 0)
            return pid if "rsi.loop" in open(f"/proc/{pid}/cmdline").read() else None
        except (OSError, ValueError):
            return None

    def stopped(self):
        return os.path.exists(self.p("STOPPED"))

    def start(self):
        if os.path.exists(self.p("STOPPED")):
            os.unlink(self.p("STOPPED"))
        if self.pid():
            return self.pid()
        cmd = [sys.executable, "-u", "-m", "rsi.loop", self.folder]
        lim = sizing(hardware(), self.cfg.get("small", True))   # decided for this machine, now
        props = (["-p", f"MemoryMax={lim['memory_cap']}", "-p", "MemorySwapMax=0"] if lim.get("memory_cap") else []) + \
                (["-p", f"CPUQuota={lim['cpu_quota']}"] if lim.get("cpu_quota") else [])
        if props and shutil.which("systemd-run"):
            cmd = ["systemd-run", "--user", "--scope", "-q"] + props + cmd
        subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=open(self.p("loop.log"), "a"), stderr=subprocess.STDOUT)
        for _ in range(40):
            time.sleep(0.25)
            if self.pid():
                break
        return self.pid()

    def stop(self, grace=10):
        open(self.p("STOPPED"), "w").write(time.strftime("%Y-%m-%d %H:%M"))   # keeps open windows from restarting it
        pid = self.pid()
        if pid:
            os.kill(pid, signal.SIGTERM)
            for _ in range(grace * 4):
                time.sleep(0.25)
                try:
                    os.kill(pid, 0)
                except OSError:
                    return pid
            os.kill(pid, signal.SIGKILL)
        return pid


def hardware():
    """what this machine has: CUDA GPUs (via torch, else nvidia-smi), CPU cores, RAM"""
    hw = {"cuda": False, "gpus": [], "cpus": os.cpu_count(),
          "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)}
    try:
        import torch
        if torch.cuda.is_available():
            hw["gpus"] = [{"name": torch.cuda.get_device_name(i), "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                          for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    if not hw["gpus"] and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            hw["gpus"] = [{"name": l.split(",")[0].strip(), "vram_gb": round(float(l.split(",")[1]) / 1024, 1)} for l in out.splitlines() if "," in l]
        except Exception:  # noqa: BLE001
            pass
    hw["cuda"] = bool(hw["gpus"])
    hw["label"] = (" + ".join(f"{g['name']} ({g['vram_gb']:.0f} GB)" for g in hw["gpus"]) if hw["gpus"]
                   else f"CPU only ({hw['cpus']} cores, {hw['ram_gb']:.0f} GB RAM)")
    return hw


def sizing(hw, small):
    """parallelism and limits for this machine (re-applied on every start, so a session moved to a GPU box speeds up)"""
    vram = sum(g["vram_gb"] for g in hw["gpus"])
    if hw["cuda"] and (small or vram >= 24):
        return {"device": "cuda", "workers": 8 if small else (4 if vram < 48 else 8), "memory_cap": None}
    if hw["cuda"]:   # a big model on a small GPU: ollama splits it with system RAM
        return {"device": "cuda", "workers": 1, "memory_cap": None}
    # CPU-only: cap memory on small-RAM machines and use at most half the cores, so a laptop doesn't overheat
    return {"device": "cpu", "workers": 1 if small else 2, "memory_cap": "3500M" if small and hw["ram_gb"] < 24 else None,
            "cpu_quota": f"{max(1, (hw['cpus'] or 2) // 2) * 100}%"}


def model_info(tag, host=None):
    env = dict(os.environ, OLLAMA_HOST=host) if host else None
    info = {}
    for l in subprocess.run(["ollama", "show", tag], capture_output=True, text=True, env=env).stdout.splitlines():
        p = l.split()
        if len(p) >= 2 and p[0] in ("architecture", "parameters", "quantization") and p[0] not in info:
            info[p[0]] = p[-1]   # first occurrence = the language model (later sections describe the vision projector)
    return info


def models(host=None):
    env = dict(os.environ, OLLAMA_HOST=host) if host else None
    rows = []
    for l in subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env).stdout.splitlines()[1:]:
        p = l.split()
        if len(p) >= 4:
            rows.append({"tag": p[0], "size": " ".join(p[2:4])})
    return rows


def plan(tag, host=None):
    """Any model ollama can run. Which GGUF is the editable master (Q8_0/F16/BF16/F32 can be overwritten in place),
    and which quantization the derived models run at. Quantized-only models still work: write_gguf re-encodes the
    tensors it changes, and the model can re-quantize or fetch a higher-precision build itself."""
    info = model_info(tag, host)
    if not info.get("architecture"):
        return None, f"ollama doesn't know a model called {tag}"
    q = info.get("quantization", "").upper(); base = {"params": info.get("parameters", ""), "arch": info["architecture"]}
    if q in ("Q8_0", "F16", "BF16", "F32"):
        return {"master_tag": tag, "run_quant": None, **base}, None
    name, size = tag.split(":")[0], (tag.split(":")[1].split("-")[0] if ":" in tag else "latest")
    for cand in (f"{name}:{size}-q8_0", f"{name}:{size}-fp16", f"{name}:{size}-bf16"):
        if model_info(cand, host).get("architecture") == info["architecture"]:
            return {"master_tag": cand, "run_quant": "Q4_K_M" if q.startswith("Q4") else q, **base}, None
    return {"master_tag": tag, "run_quant": None, **base}, None


def create(folder, model, goal):
    s = Session(folder)
    for d in ("state", "live"):
        os.makedirs(s.p(d), exist_ok=True)
    admin = settings.for_new_loop(True)["ollama_admin_url"]
    pl, problem = plan(model, admin.split("://")[-1])
    if problem:
        raise ValueError(problem)
    pc = pl["params"].upper()
    small = pc.endswith("M") or (pc.endswith("B") and pc[:-1].replace(".", "", 1).isdigit() and float(pc[:-1]) <= 4)
    cfg = {**settings.for_new_loop(small), "model": model, "goal": goal, "master_tag": pl["master_tag"], "run_quant": pl["run_quant"], "arch": pl["arch"],
           "name": f"s{abs(hash(s.folder)) % 10**6}", "small": small, "created": time.strftime("%Y-%m-%d %H:%M")}
    s.write("config.json", cfg)
    return s


# ------------------------------------------------------------------------------------------------ the loop
class Loop:
    def __init__(self, folder):
        self.s = Session(folder); self.state = self.s.p("state"); os.makedirs(self.state, exist_ok=True)
        self.hw = hardware(); self.sz = sizing(self.hw, self.s.cfg.get("small", True))   # this machine, this start
        self.reload()
        llm.LIVE["dir"] = self.s.p("live"); os.makedirs(self.s.p("live"), exist_ok=True)
        self.workspace = os.path.join(self.s.folder, "workspace"); os.makedirs(self.workspace, exist_ok=True)
        self.gate_dir = self.s.p("net")
        self.lineage = self.s.read("lineage.json") or {"generations": [], "current": None}
        self.chat = self.s.read("state/chat.json", []) or []
        self.transcript = (self.s.read("live/transcript.json", {}) or {}).get("entries", [])
        self.cycle_k = 0; self.probation = None
        self.jobs = Jobs(self)

    def reload(self):
        """settings are re-read at every step, so `rsi settings` changes apply to a running loop"""
        self.cfg = settings.effective(self.s.cfg, self.sz)
        self.admin = self.cfg["ollama_admin_url"].split("://")[-1]

    # -------- bookkeeping
    def status(self, cycle, step, detail="", **kw):
        self.s.write("status.json", {"cycle": cycle, "step": step, "detail": detail, "since": time.time(), "generation": self.lineage["current"],
                                     "model": self.tag if self.lineage["generations"] else self.cfg["master_tag"],
                                     "device": self.hw["label"], "pid": os.getpid(), **kw})

    def status_activity(self, activity, detail=""):
        self.status(self.cycle_k, activity, detail)

    def note(self, kind, text):
        """the transcript the app shows: model messages, tool calls, tool results, your messages"""
        self.transcript.append({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": str(text)[:4000], "cycle": self.cycle_k})
        self.transcript = self.transcript[-150:]
        try:
            self.s.write("live/transcript.json", {"cycle": self.cycle_k, "entries": self.transcript})
        except Exception:  # noqa: BLE001
            pass

    def record_test(self, k, summ):
        runs = self.s.read("state/test_runs.json", []) or []
        runs.append({"cycle": k, **{x: summ[x] for x in ("run_id", "suite", "n", "baseline", "models", "at")},
                     "settings": {x: summ["settings"].get(x) for x in ("think", "num_predict", "internet", "code")}})
        self.s.write("state/test_runs.json", runs[-300:])
        self.note("result", "test " + summ["suite"] + ": " + ", ".join(f"{m.split(':')[-1]} {r['score']:.0%}" for m, r in summ["models"].items()))

    @property
    def gen(self):
        return next(g for g in self.lineage["generations"] if g["gen"] == self.lineage["current"])

    @property
    def tag(self):
        return self.gen["tag"]

    def gguf_of(self, tag):
        return gguf_io.ollama_show(tag, self.admin)[0]

    def add_generation(self, tag, note):
        n = len(self.lineage["generations"])
        self.lineage["generations"].append({"gen": n, "tag": tag, "note": note, "at": time.strftime("%Y-%m-%d %H:%M")})
        self.lineage["current"] = n; self.s.write("lineage.json", self.lineage)

    def switch(self, tag, reason, k):
        prev = self.lineage["current"]
        self.add_generation(tag, f"cycle {k}: {reason}"[:300])
        self.probation = {"prev": prev, "replies": 0}
        self.note("note", f"switched to {tag}: {reason}")
        self.write_context(k)

    def revert(self, why):
        """the new model couldn't carry on the work: go back to the one before it"""
        bad = self.gen; prev = self.probation["prev"] if self.probation else 0
        bad["reverted"] = why
        self.lineage["current"] = prev; self.s.write("lineage.json", self.lineage); self.probation = None
        self.note("note", f"switched back to {self.tag}: {bad['tag']} {why}")
        self.chat.append({"role": "user", "content": f"[loop] Your switch to {bad['tag']} was undone because it {why}. You are "
                                                     f"{self.tag} again. Note this and carry on."})
        self.write_context(self.cycle_k)

    def setup(self):
        if not self.lineage["generations"]:
            self.status(0, "setup", "preparing generation 0")
            tag = self.cfg["master_tag"]
            if self.cfg.get("run_quant"):   # big models: run a quantized copy, keep the Q8_0 master for editing
                build = os.path.join(self.state, "build"); os.makedirs(build, exist_ok=True)
                out = os.path.join(build, "g0.gguf"); gguf_io.derive_run_quant(self.gguf_of(tag), out, self.cfg["run_quant"])
                tag = f"rsi-{self.cfg['name']}:g0"; gguf_io.register(tag, out, self.cfg["master_tag"], self.admin)
                shutil.rmtree(build, ignore_errors=True)
            self.add_generation(tag, "the model as downloaded")
        tests.write_library(self.workspace)

    def write_context(self, k):
        json.dump({"goal": self.cfg["goal"], "cycle": k, "generation": self.lineage["current"], "current_model": self.tag,
                   "current_gguf": self.gguf_of(self.tag), "editable_master_gguf": self.gguf_of(self.cfg["master_tag"]),
                   "arch": self.cfg.get("arch", "qwen35"), "ollama_host": "127.0.0.1:11434",
                   "tokenizer": os.path.join(ROOT, "data", "qwen35_tokenizer.json") if self.cfg.get("arch", "qwen35") == "qwen35" else None,
                   "device": "cuda" if self.hw["cuda"] else "cpu", "hardware": self.hw["label"],
                   "lineage": self.lineage["generations"]}, open(os.path.join(self.workspace, ".rsi_context.json"), "w"), indent=1)

    def system_prompt(self):
        c = self.cfg
        walls = ("isolated sandbox (only the workspace is writable, your home is hidden)" if c.get("sandbox_isolate", True)
                 else "no sandbox (your code runs as the user)")
        net = ("internet on" if c.get("sandbox_internet", True) else "internet off") + (", local network on" if c.get("sandbox_local") else "")
        lim = [f"{c['max_steps']} tool calls" if c.get("max_steps") else None,
               f"{c['max_cycle_minutes']} minutes" if c.get("max_cycle_minutes") else None]
        return (open(PLAYBOOK).read() + f"\n\n# Your goal (set by the human)\n\n{c['goal']}\n\n# Where you are now\n\n"
                f"- You are `{self.tag}` (generation {self.lineage['current']}; base model `{c['model']}`).\n"
                f"- Your weights: `{self.gguf_of(self.tag)}` (read-only). Editable master: `{self.gguf_of(c['master_tag'])}`.\n"
                f"- Your workspace is `/workspace`: your code starts there, and file tool paths are relative to it. "
                f"Test suites go in `tests/<name>/`; ready-made ones are in `tests/library/`.\n"
                f"- Hardware: {self.hw['label']}. Code runs in {walls}, {net}"
                + (f", memory cap {c['sandbox_mem']}" if c.get("sandbox_mem") else "") + ".\n"
                f"- Budget per cycle: {', '.join(x for x in lim if x) or 'no limit'}. Models you may keep at once: {c.get('max_models') or 'no limit'}.\n"
                f"- Test defaults: thinking {settings.think(c.get('test_think', 'on'))}, {c.get('test_num_predict')} tokens per reply.\n")

    def _inbox(self):
        p = self.s.p("inbox.jsonl")
        return [json.loads(l)["text"].strip() for l in open(p).read().splitlines() if l.strip()] if os.path.exists(p) else []

    def _trim(self, msgs):
        """fit the conversation in the context window: shorten old tool results, then drop the oldest turns"""
        # the window holds the system prompt, the tool schemas, the conversation AND the reply: leave room for all of
        # them, or ollama silently cuts the start of the prompt (the system prompt goes first). ~3 characters per token.
        fixed = len(self.system_prompt()) + len(json.dumps(tool_list(self.cfg)))
        room = int(self.cfg["num_ctx"]) - int(self.cfg["pilot_num_predict"]) - fixed // 3 - 512
        budget = max(room, 1500) * 3
        msgs = [dict(m) for m in msgs]
        size = lambda ms: sum(len(str(m.get("content", ""))) + len(json.dumps(m.get("tool_calls", ""))) for m in ms)
        for m in msgs[:-8]:
            if size(msgs) <= budget:
                return msgs
            if m["role"] == "tool" and len(m["content"]) > 600:
                m["content"] = m["content"][:300] + " …[shortened]… " + m["content"][-200:]
        if size(msgs) <= budget:
            return msgs
        head, rest = msgs[:1], msgs[1:]
        while rest and size(head + rest) > budget * 0.85:
            rest = rest[1:]
            while rest and rest[0]["role"] != "user":
                rest = rest[1:]
        hist = self.s.history()[-8:]
        note = {"role": "user", "content": "[loop] Older turns were removed to fit your context window. Earlier cycles: " +
                json.dumps([{"cycle": h.get("cycle"), "model_after": h.get("model_after"), "summary": (h.get("summary") or "")[:400]} for h in hist])}
        return head + [note] + rest

    def _save_chat(self):
        self.chat = self.chat[-1500:]
        self.s.write("state/chat.json", self.chat)

    # -------- one cycle: the model works with its tools until it calls finish
    def cycle(self, k, direction):
        t0 = time.time(); self.cycle_k = k
        tb = Toolbox(self, k)
        rec = {"cycle": k, "generation": self.lineage["current"], "model": self.tag, "direction": direction, "started": time.strftime("%H:%M")}
        self.write_context(k)
        hist = self.s.history()
        opening = (f"Cycle {k} begins." + (f" Goal for this cycle ({'from the human' if (direction or {}).get('from') == 'human' else 'your own pick'}): "
                                          f"{direction['text']}" if direction else " This is the first cycle.") +
                   f"\nMain goal: {self.cfg['goal']}\n")
        if not hist:
            opening += ("\nStart by deciding how to measure progress on the goal: write test suites in tests/ (see the playbook), "
                        "run them on yourself, and read what goes wrong. Ready-made suites are in tests/library/. Before you edit "
                        "weights, map your own layers (playbook: 'How to measure your own model's map'): what matters in one "
                        "model can be harmless in another.\n")
        if self.jobs.running():
            opening += f"\nBackground jobs still running: {', '.join(self.jobs.running())}.\n"
        opening += "\nWorkspace:\n" + tb.t_list_files()[:3000]
        self.chat.append({"role": "user", "content": opening}); self._save_chat()
        self.note("note", f"cycle {k} begins" + (f": {direction['text']}" if direction else ""))
        seen = len(self._inbox()); quiet = late = 0; step = 0
        while True:
            self.reload(); c = self.cfg
            if (c.get("max_steps") and step >= int(c["max_steps"])) or (c.get("max_cycle_minutes") and time.time() - t0 > int(c["max_cycle_minutes"]) * 60):
                if late >= 2:
                    break
                late += 1; self.chat.append({"role": "user", "content": "[loop] This cycle's budget is used up: call finish now."})
            box = self._inbox()
            for t in box[seen:]:
                if t:
                    self.chat.append({"role": "user", "content": f"Message from the human: {t}"}); self.note("human", t)
            seen = len(box)
            for name, j in self.jobs.poll():   # background jobs that ended since the last step
                self.chat.append({"role": "user", "content": "[loop] " + self.jobs.summary(name, j)}); self.note("note", f"job {name} {j['state']}")
            if step and step % 60 == 0 and not c.get("max_steps"):
                self.chat.append({"role": "user", "content": f"[loop] {step} tool calls so far this cycle. When you have tested and decided, call finish."})
            step += 1
            self.status(k, "thinking", f"step {step}")
            try:
                r = llm.chat(c, self.tag, [{"role": "system", "content": self.system_prompt()}] + self._trim(self.chat),
                             think=settings.think(c.get("pilot_think", "on")), tools=tool_list(c), num_predict=c["pilot_num_predict"],
                             options={"temperature": c["pilot_temperature"]} if c.get("pilot_temperature") is not None else None,
                             live={"slot": "pilot", "title": f"{self.tag.split(':')[-1]}: cycle {k}, step {step}"})
            except Exception as e:  # noqa: BLE001
                if self.probation:
                    self.revert(f"failed to run ({type(e).__name__}: {str(e)[:150]})"); continue
                raise
            m = r["message"]; calls = m.get("tool_calls") or []; content = (m.get("content") or "").strip()
            if self.probation:
                self.probation["replies"] += 1
                if not calls and not content:
                    self.revert("gave an empty reply"); continue
                if calls or self.probation["replies"] >= 3:
                    self.probation = None   # it can work: the switch stands
            self.chat.append({"role": "assistant", "content": m.get("content", ""), **({"tool_calls": calls} if calls else {})})
            if content:
                self.note("model", content)
            if not calls:
                quiet += 1
                if quiet >= 3:
                    break
                self.chat.append({"role": "user", "content": "[loop] Use your tools to make progress. When you are done with this cycle, call finish."})
                self._save_chat(); continue
            quiet = 0
            for call in calls[:4]:
                fn = call.get("function", {}); args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                name = fn.get("name", "")
                self.note("call", f"{name}({json.dumps(args)[:1500]})")
                self.status(k, ACTIVITY.get(name, "code"), f"{name} …")
                out = tb.call(name, args if isinstance(args, dict) else {})
                self.note("result", out[:3000])
                self.chat.append({"role": "tool", "content": out, "tool_name": name})
            self._save_chat()
            if tb.finished:
                break
        fin = tb.finished or self.plan_next(k, tb)
        self.chat.append({"role": "user", "content": f"[loop] Cycle {k} is over. Next: {fin['next_plans'][fin['picked']]}"}); self._save_chat()
        rec.update(summary=fin["summary"], next_plans=fin["next_plans"], picked=fin["picked"], pick_reason=fin.get("pick_reason", ""),
                   test_runs=[{x: t[x] for x in ("run_id", "suite", "n", "models")} for t in tb.test_runs], switches=tb.switches,
                   generation_after=self.lineage["current"], model_after=self.tag, tools_used=len(tb.log), tool_log=tb.log[-80:],
                   minutes=round((time.time() - t0) / 60, 1))
        return rec

    def plan_next(self, k, tb):
        """the model stopped without calling finish: ask it for a summary and plans directly"""
        msgs = [{"role": "system", "content": self.system_prompt()}] + self._trim(self.chat) + [
            {"role": "user", "content": "[loop] The cycle is over. Summarise what you did and learned, propose 2 to 4 goals for the next "
                                        "cycle, and pick one. Answer in JSON."}]
        try:
            r = llm.chat(self.cfg, self.tag, msgs, think=settings.think(self.cfg.get("pilot_think", "on")), fmt=PLANS,
                         num_predict=self.cfg["pilot_num_predict"], live={"slot": "pilot", "title": "planning the next cycle"})
            d = json.loads(r["message"]["content"])
            plans = [str(p) for p in d.get("plans", [])][:4]
            if plans:
                return {"summary": d.get("summary", ""), "next_plans": plans, "picked": min(max(int(d.get("pick", 0)), 0), len(plans) - 1),
                        "pick_reason": d.get("reason", "")}
        except Exception:  # noqa: BLE001
            pass
        return {"summary": "(the model ended the cycle without a summary)", "next_plans": ["continue toward the main goal"], "picked": 0, "pick_reason": "fallback"}

    def run(self):
        open(self.s.p("loop.pid"), "w").write(str(os.getpid()))

        def term(*_):
            self.jobs.stop_all()
            try:
                self.status(len(self.s.history()), "stopped", "the loop was stopped"); os.unlink(self.s.p("loop.pid"))
            except Exception:  # noqa: BLE001
                pass
            os._exit(0)
        signal.signal(signal.SIGTERM, term)
        self.setup()
        hist = self.s.history(); k = len(hist); direction = None
        if hist and hist[-1].get("next_plans"):   # resuming: carry on with the plan picked last time
            last = hist[-1]; direction = {"from": "model", "text": last["next_plans"][last.get("picked", 0)]}
        seen = len(self._inbox())
        while True:
            box = self._inbox(); texts = [t for t in box[seen:] if t]; seen = len(box)
            if texts:
                direction = {"from": "human", "text": texts[-1]}   # your direction beats the model's own pick
            try:
                rec = self.cycle(k, direction)
                direction = {"from": "model", "text": rec["next_plans"][rec["picked"]]}
            except Exception as e:  # noqa: BLE001
                rec = {"cycle": k, "generation": self.lineage["current"], "summary": f"error: {type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-2000:]}
                self.status(k, "error", rec["summary"][:150]); time.sleep(30)
            with open(self.s.p("history.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            k += 1; self.cycle_k = k
            self.reload()
            if not self.cfg.get("auto_continue", True):   # wait for the human before the next cycle
                self.status(k, "waiting", "waiting for your message to start the next cycle")
                while len(self._inbox()) <= seen:
                    time.sleep(2); self.reload()
                    if self.cfg.get("auto_continue", True):
                        break


if __name__ == "__main__":
    Loop(sys.argv[1]).run()
