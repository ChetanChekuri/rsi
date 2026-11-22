"""The RSI loop. You give one main prompt (the goal). Then, every cycle:

  1 practice  the current model solves fresh problems for the goal; the loop counts where it goes wrong
  2 work      the model works on itself with tools: it runs any code in its sandboxed workspace, reads its own
              weights, searches the web, builds new models (GGUF files), registers them with ollama, tests them on
              practice problems, deletes the ones it doesn't need, and may submit one as its successor
  3 test      a submitted model is tested against the current one on fresh held-out problems (goal + general); it
              becomes the new current model only if it is better without getting worse at general problems
  4 plan      the model ends the cycle with a summary and plans for the next one, and picks one

Anything you type reaches the model during the cycle, and overrides its pick for the next one.
Everything lives in <folder>/.rsi/ (state) and <folder>/workspace/ (the model's sandbox).
"""
import json, math, os, shutil, signal, subprocess, sys, time, traceback
import numpy as np
from . import llm, detectors, gguf_io, research
from .factory import Factory, BUILTIN, design
from .probes import general_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYBOOK = os.path.join(ROOT, "rsi", "playbook.md")
PLANS = {"type": "object", "properties": {"plans": {"type": "array", "items": {"type": "string"}}, "pick": {"type": "integer"},
                                          "reason": {"type": "string"}}, "required": ["plans", "pick", "reason"]}



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
    """which GGUF gets edited (needs Q8_0/F16) and which quantization the candidates run at"""
    info = model_info(tag, host)
    if info.get("architecture") != "qwen35":
        return None, f"{tag} is {info.get('architecture', '?')}; the weight editor supports the Qwen3.5 / Qwen3.8 family (qwen35)"
    q = info.get("quantization", "").upper()
    if q in ("Q8_0", "F16", "BF16", "F32"):
        return {"master_tag": tag, "run_quant": None, "params": info.get("parameters", "")}, None
    base, size = tag.split(":")[0], (tag.split(":")[1].split("-")[0] if ":" in tag else "latest")
    master = f"{base}:{size}-q8_0"
    if model_info(master, host).get("architecture") != "qwen35":
        return None, f"{tag} is {q}; editing needs a Q8_0 build. Pull it first:  ollama pull {master}"
    return {"master_tag": master, "run_quant": "Q4_K_M" if q.startswith("Q4") else q, "params": info.get("parameters", "")}, None


def create(folder, model, goal):
    s = Session(folder)
    for d in ("state", "live"):
        os.makedirs(s.p(d), exist_ok=True)
    url = os.environ.get("RSI_OLLAMA_URL", "http://127.0.0.1:11434")
    admin = os.environ.get("RSI_OLLAMA_ADMIN_URL", url)
    pl, problem = plan(model, admin.split("://")[-1])
    if problem:
        raise ValueError(problem)
    pc = pl["params"].upper()
    small = pc.endswith("M") or (pc.endswith("B") and pc[:-1].replace(".", "", 1).isdigit() and float(pc[:-1]) <= 4)
    cfg = {"model": model, "goal": goal, "master_tag": pl["master_tag"], "run_quant": pl["run_quant"], "name": f"s{abs(hash(s.folder)) % 10**6}",
           "ollama_url": url, "ollama_admin_url": admin, "research": True, "small": small,
           "n_dev_per_family": 3, "n_test_per_family": 5, "n_general": 8, "min_gain": 2, "max_general_loss": 1,
           "num_predict": 3072 if small else 16384, "num_ctx": 16384 if small else 32768, "pilot_num_predict": 4096 if small else 12000,
           "max_steps": 40, "max_cycle_minutes": 120, "max_run_s": 1800, "sandbox_mem": "4G" if small else "24G", "sandbox_network": False,
           "pilot_think": True, "created": time.strftime("%Y-%m-%d %H:%M"), **sizing(hardware(), small)}
    s.write("config.json", cfg)
    return s


# ------------------------------------------------------------------------------------------------ the loop
def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def compare(a, b, ids):
    A = {r["id"]: r for r in a}; B = {r["id"]: r for r in b}; ids = [i for i in ids if i in A and i in B]
    fixed = sum(B[i]["correct"] and not A[i]["correct"] for i in ids); broken = sum(A[i]["correct"] and not B[i]["correct"] for i in ids)
    fin = lambda R: round(float(np.mean([R[i]["done_reason"] != "length" for i in ids])), 2) if ids else None
    return {"n": len(ids), "before": round(float(np.mean([A[i]["correct"] for i in ids])), 3) if ids else None,
            "after": round(float(np.mean([B[i]["correct"] for i in ids])), 3) if ids else None, "fixed": fixed, "broken": broken,
            "p": round(sign_p(fixed, broken), 3), "finished_before": fin(A), "finished_after": fin(B)}


class Asker:
    """lets the factory ask the model for structured answers (problem generators, reviews)"""
    def __init__(self, loop):
        self.loop = loop

    def ask(self, phase, brief, schema, check=None):
        cfg = self.loop.cfg
        msgs = [{"role": "system", "content": open(PLAYBOOK).read()},
                {"role": "user", "content": f"{phase}\n\n{json.dumps(brief, indent=1)}\n\nAnswer with JSON matching: {json.dumps(schema)}"}]
        for _ in range(2):
            r = llm.chat(cfg, self.loop.tag, msgs, think=cfg["pilot_think"], fmt=schema, num_predict=cfg["pilot_num_predict"],
                         live={"slot": "pilot", "title": f"writing practice problems for the goal ({phase.lower()})"})
            try:
                return json.loads(r["message"]["content"]), r["message"].get("thinking", "")
            except ValueError:
                msgs += [{"role": "assistant", "content": r["message"]["content"]}, {"role": "user", "content": "That was not valid JSON. Try again."}]
        raise ValueError("no valid JSON")




class Loop:
    def __init__(self, folder):
        self.s = Session(folder); self.cfg = self.s.cfg; self.state = self.s.p("state")
        small = self.cfg.get("small", True)
        for k, v in {"max_steps": 40, "max_cycle_minutes": 120, "max_run_s": 1800, "sandbox_mem": "4G" if small else "24G",
                     "sandbox_network": False, "num_ctx": 16384 if small else 32768}.items():
            self.cfg.setdefault(k, v)   # loops created before these settings existed
        self.hw = hardware(); self.cfg.update(sizing(self.hw, small))   # this machine, this start
        llm.LIVE["dir"] = self.s.p("live"); os.makedirs(self.s.p("live"), exist_ok=True)
        self.workspace = os.path.join(self.s.folder, "workspace"); os.makedirs(self.workspace, exist_ok=True)
        self.admin = self.cfg["ollama_admin_url"].split("://")[-1]
        self.fac = Factory(self.state)
        self.ref = [json.loads(l) for l in open(os.path.join(ROOT, "data", "reference_traces.jsonl"))]
        self.general = general_pool(exclude={r["id"] for r in self.ref})
        self.lineage = self.s.read("lineage.json") or {"generations": [], "current": None}
        self.transcript = []

    # -------- bookkeeping
    def status(self, cycle, step, detail="", **kw):
        self.s.write("status.json", {"cycle": cycle, "step": step, "detail": detail, "since": time.time(), "generation": self.lineage["current"],
                                     "model": self.tag if self.lineage["generations"] else self.cfg["master_tag"],
                                     "device": self.hw["label"], "pid": os.getpid(), **kw})

    def note(self, kind, text):
        """the agent transcript the app shows: model messages, tool calls, tool results, your messages"""
        self.transcript.append({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": str(text)[:4000]})
        self.transcript = self.transcript[-120:]
        try:
            self.s.write("live/transcript.json", {"cycle": self.cycle_k, "entries": self.transcript})
        except Exception:  # noqa: BLE001
            pass

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

    def setup(self):
        if not self.lineage["generations"]:
            self.status(0, "setup", "preparing generation 0")
            tag = self.cfg["master_tag"]
            if self.cfg.get("run_quant"):   # big models: run a quantized copy, edit the Q8_0 master
                build = os.path.join(self.state, "build"); os.makedirs(build, exist_ok=True)
                out = os.path.join(build, "g0.gguf"); gguf_io.derive_run_quant(self.gguf_of(tag), out, self.cfg["run_quant"])
                tag = f"rsi-{self.cfg['name']}:g0"; gguf_io.register(tag, out, self.cfg["master_tag"], self.admin)
                shutil.rmtree(build, ignore_errors=True)
            self.add_generation(tag, "the model as downloaded")
        if not self.fac.meta:
            self.status(0, "setup", "the model is writing practice-problem generators for the goal")
            kept = []
            try:
                kept = design(Asker(self), self.fac, self.cfg["goal"], 4, frozen=True, origin="gen0")
            except Exception:  # noqa: BLE001
                pass
            if len(kept) < 2:   # never stall: fall back to the built-in generators
                for n, (sk, code) in BUILTIN.items():
                    if n not in self.fac.meta:
                        self.fac.add(n, sk, code, frozen=True, origin="built-in")

    def write_context(self, k, practice_file):
        json.dump({"goal": self.cfg["goal"], "cycle": k, "generation": self.lineage["current"], "current_model": self.tag,
                   "current_gguf": self.gguf_of(self.tag), "editable_master_gguf": self.gguf_of(self.cfg["master_tag"]),
                   "tokenizer": os.path.join(ROOT, "data", "qwen35_tokenizer.json"), "latest_practice_traces": practice_file,
                   "device": "cuda" if self.hw["cuda"] else "cpu", "hardware": self.hw["label"],
                   "lineage": self.lineage["generations"]}, open(os.path.join(self.workspace, ".rsi_context.json"), "w"), indent=1)

    # -------- the judge: only this decides whether a model replaces the current one
    def held_out_test(self, k, cand):
        cfg = self.cfg
        n0 = cfg["n_general"]; start = (k * n0 + self.fac.secret) % max(1, len(self.general) - n0)
        focus = self.fac.sample("heldout", k, cfg["n_test_per_family"]); general = self.general[start:start + n0]
        self.status(k, "test", f"held-out test: current model on {len(focus) + len(general)} fresh problems")
        a = llm.solve(cfg, self.tag, focus + general, os.path.join(self.state, "tests", f"c{k}-current.jsonl"), who="held-out test, current model")
        self.status(k, "test", f"held-out test: {cand} on the same problems")
        b = llm.solve(cfg, cand, focus + general, os.path.join(self.state, "tests", f"c{k}-candidate.jsonl"), who=f"held-out test, {cand}")
        fp = compare(a, b, [it["id"] for it in focus]); gp = compare(a, b, [it["id"] for it in general])
        keep = fp["fixed"] - fp["broken"] >= cfg["min_gain"] and gp["broken"] - gp["fixed"] <= cfg["max_general_loss"]
        verdict = (f"goal {fp['before']:.0%} → {fp['after']:.0%} (+{fp['fixed']}/−{fp['broken']}), general {gp['before']:.0%} → "
                   f"{gp['after']:.0%} (+{gp['fixed']}/−{gp['broken']}): " + ("KEPT, it is now the current model" if keep else
                   f"not kept (needs +{cfg['min_gain']} net on the goal and at most −{cfg['max_general_loss']} net on general problems)"))
        if keep:
            self.add_generation(cand, f"cycle {k}")
        self.note("result", "held-out test: " + verdict)
        return {"kept": keep, "goal": fp, "general": gp, "verdict": verdict}

    # -------- one cycle: practice, then the model works on itself with tools
    def cycle(self, k, direction):
        from .tools import TOOLS, Toolbox
        cfg = self.cfg; t0 = time.time(); self.cycle_k = k; self.transcript = []
        rec = {"cycle": k, "generation": self.lineage["current"], "model": self.tag, "direction": direction, "started": time.strftime("%H:%M")}
        # 1 practice (the loop does this, so every cycle starts from fresh evidence)
        dev = self.fac.sample("dev", k * 100, self.cfg["n_dev_per_family"])
        self.status(k, "practice", f"solving {len(dev)} fresh practice problems")
        pfile = os.path.join(self.workspace, "practice", f"current-c{k}.jsonl")
        recs = llm.solve(cfg, self.tag, dev, pfile, who="practice")
        summ, D = detectors.summarize(recs); self.fac.update_levels(recs)
        rec["practice"] = {"accuracy": round(float(np.mean([r["correct"] for r in recs])), 3), "n": len(recs),
                           "unfinished": summ["counts"]["unfinished"], "doubt_after_answer": summ["doubt_events_after_answer"]}
        self.write_context(k, os.path.relpath(pfile, self.workspace))
        # 2 work: a free tool-using episode
        tb = Toolbox(self, k)
        brief = {"goal": cfg["goal"], "cycle": k, "you_are": f"{self.tag} (generation {self.lineage['current']})",
                 "plan_for_this_cycle": (direction or {}).get("text") or "(your call)", "plan_came_from": (direction or {}).get("from", ""),
                 "practice_just_now": rec["practice"], "accuracy_by_problem_type": summ["accuracy_by_suite"],
                 "failure_patterns_by_outcome": summ["detectors_mean_by_outcome"], "failing_excerpts": detectors.excerpts(recs, D, k=2, width=500),
                 "practice_traces_file": os.path.relpath(pfile, self.workspace),
                 "earlier_cycles": [{"cycle": h.get("cycle"), "summary": h.get("summary"), "submitted": (h.get("submitted") or {}).get("model"),
                                     "verdict": (h.get("submitted") or {}).get("verdict")} for h in self.s.history()[-6:]],
                 "your_models": tb._owned(), "workspace_files": tb.t_list_files()[:3000],
                 "budget": f"up to {cfg['max_steps']} tool calls and about {cfg['max_cycle_minutes']} minutes this cycle"}
        msgs = [{"role": "system", "content": open(PLAYBOOK).read() + f"\n\n# Your goal (set by the human)\n\n{cfg['goal']}"},
                {"role": "user", "content": "A new cycle. Here is where you stand:\n\n" + json.dumps(brief, indent=1, default=str)}]
        self.status(k, "work", "working on itself"); self.note("note", f"cycle {k} begins; practice accuracy {rec['practice']['accuracy']:.0%}")
        seen = len(self._inbox()); nudges = late = 0
        for step in range(cfg["max_steps"]):
            if time.time() - t0 > cfg["max_cycle_minutes"] * 60:
                if late > 2:
                    break
                late += 1; msgs.append({"role": "user", "content": "Time is up for this cycle: call finish now."})
            for t in self._inbox()[seen:]:   # you can talk to it while it works
                if t:
                    msgs.append({"role": "user", "content": f"Message from the human: {t}"}); self.note("human", t)
            seen = len(self._inbox())
            r = llm.chat(cfg, self.tag, self._trim(msgs), think=cfg["pilot_think"], tools=TOOLS, num_predict=cfg["pilot_num_predict"],
                         live={"slot": "pilot", "title": f"working on itself (step {step + 1})"})
            m = r["message"]; calls = m.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": m.get("content", ""), **({"tool_calls": calls} if calls else {})})
            if m.get("content", "").strip():
                self.note("model", m["content"])
            if not calls:
                nudges += 1
                if nudges > 2:
                    break
                msgs.append({"role": "user", "content": "Use your tools to make progress on the goal. When you are done, call finish."}); continue
            for c in calls[:3]:
                fn = c.get("function", {}); args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                self.note("call", f"{fn.get('name')}({json.dumps(args)[:1500]})")
                self.status(k, "work", f"{fn.get('name')} …")
                out = tb.call(fn.get("name", ""), args if isinstance(args, dict) else {})
                self.note("result", out[:3000])
                msgs.append({"role": "tool", "content": out, "tool_name": fn.get("name", "")})
            if tb.finished:
                break
        # 3 test happened inside submit (if the model submitted); 4 plan comes from finish
        fin = tb.finished or {}
        if not fin:
            plans, pick, reason = self.plan_next(k, rec, tb)
            fin = {"summary": "(the model did not call finish)", "next_plans": plans, "picked": pick, "pick_reason": reason}
        rec.update(summary=fin["summary"], next_plans=fin["next_plans"], picked=fin["picked"], pick_reason=fin.get("pick_reason", ""),
                   submitted=tb.submitted, kept=bool(tb.submitted and tb.submitted.get("kept")), tools_used=len(tb.log),
                   tool_log=tb.log[-60:], generation_after=self.lineage["current"], minutes=round((time.time() - t0) / 60, 1))
        return rec

    def _inbox(self):
        p = self.s.p("inbox.jsonl")
        return [json.loads(l)["text"].strip() for l in open(p).read().splitlines()] if os.path.exists(p) else []

    def _trim(self, msgs, budget=None):
        """keep the conversation inside the context window: shorten old tool results first"""
        budget = budget or self.cfg["num_ctx"] * 3
        size = lambda: sum(len(str(m.get("content", ""))) for m in msgs)
        for m in msgs[2:-6]:
            if size() <= budget:
                break
            if m["role"] == "tool" and len(m["content"]) > 400:
                m["content"] = m["content"][:200] + " …[shortened]… " + m["content"][-150:]
        return msgs

    def plan_next(self, k, rec, tb):
        """fallback when the model never called finish: ask it directly for plans"""
        msgs = [{"role": "system", "content": open(PLAYBOOK).read()},
                {"role": "user", "content": f"Cycle {k} ended. What you did: " + json.dumps(tb.log[-12:], default=str)[:6000] +
                 "\n\nPropose 2 to 4 plans for the next cycle and pick one."}]
        try:
            r = llm.chat(self.cfg, self.tag, msgs, think=self.cfg["pilot_think"], fmt=PLANS, num_predict=self.cfg["pilot_num_predict"],
                         live={"slot": "pilot", "title": "planning the next cycle"})
            d = json.loads(r["message"]["content"])
            plans = [p.get("plan") if isinstance(p, dict) else str(p) for p in d.get("plans", [])][:4]
            if plans:
                return plans, min(max(int(d.get("pick", 0)), 0), len(plans) - 1), d.get("reason", "")
        except Exception:  # noqa: BLE001
            pass
        return ["look at the practice traces and try one concrete change to my weights"], 0, "fallback"

    def run(self):
        open(self.s.p("loop.pid"), "w").write(str(os.getpid())); self.cycle_k = 0

        def term(*_):
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
            texts = [t for t in self._inbox()[seen:] if t]; seen = len(self._inbox())
            if texts:
                direction = {"from": "human", "text": texts[-1]}   # your direction beats the model's own pick
            try:
                rec = self.cycle(k, direction)
                direction = {"from": "model", "text": rec["next_plans"][rec["picked"]]}
            except Exception as e:  # noqa: BLE001
                rec = {"cycle": k, "generation": self.lineage["current"], "kept": False, "summary": f"error: {type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-2000:]}
                self.status(k, "error", rec["summary"][:150]); time.sleep(30)
            with open(self.s.p("history.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            k += 1


if __name__ == "__main__":
    Loop(sys.argv[1]).run()
