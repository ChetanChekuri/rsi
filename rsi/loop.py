"""The RSI loop. Every cycle:

  1 observe  the model solves fresh problems for the goal; detectors count where it goes wrong
  2 think    the model reads its own failing traces (and may search the web), then picks ONE change to its weights
  3 test     the change is baked into a copy of the weights; both versions solve the same fresh held-out problems
  4 keep     the copy becomes the model if it solves more of them without getting worse at general problems

Then it waits for you: press Enter for the next cycle, or type what it should work on.
Everything lives in <folder>/.rsi/. The app starts this process (python -m rsi.loop <folder>); it keeps running
when the app is closed.
"""
import json, math, os, shutil, signal, subprocess, sys, time, traceback
import numpy as np
from . import llm, detectors, directions, gguf_io, research
from .factory import Factory, BUILTIN, design
from .probes import general_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYBOOK = os.path.join(ROOT, "rsi", "playbook.md")

# the menu of changes the model can make to itself: name -> (operator, min strength, max strength)
EDITS = {"stop_second_guessing": ("ablate", 0.1, 0.9), "be_more_decisive": ("sharpen", 1.05, 1.8),
         "be_less_decisive": ("sharpen", 0.6, 0.95), "reason_like_reference": ("amplify", 0.05, 0.6)}
DECISION = {"type": "object", "properties": {
    "diagnosis": {"type": "string"}, "edit": {"type": "string", "enum": list(EDITS)}, "strength": {"type": "number"},
    "why": {"type": "string"}, "expect": {"type": "string"}}, "required": ["diagnosis", "edit", "strength", "why", "expect"]}


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

    def start(self):
        if self.pid():
            return self.pid()
        cmd = [sys.executable, "-u", "-m", "rsi.loop", self.folder]
        cap = self.cfg.get("memory_cap")
        if cap and shutil.which("systemd-run"):
            cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0"] + cmd
        subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=open(self.p("loop.log"), "a"), stderr=subprocess.STDOUT)
        for _ in range(40):
            time.sleep(0.25)
            if self.pid():
                break
        return self.pid()

    def stop(self, grace=10):
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
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    cfg = {"model": model, "goal": goal, "master_tag": pl["master_tag"], "run_quant": pl["run_quant"], "name": f"s{abs(hash(s.folder)) % 10**6}",
           "ollama_url": url, "ollama_admin_url": admin, "research": True, "auto": False,
           "n_dev_per_family": 3, "n_test_per_family": 5, "n_general": 8, "min_gain": 2, "max_general_loss": 1,
           "num_predict": 3072 if small else 16384, "num_ctx": 6144 if small else 24576, "pilot_num_predict": 3072 if small else 12000,
           "workers": 1 if small else 4, "memory_cap": "3500M" if small and ram < 24 else None, "pilot_think": True,
           "created": time.strftime("%Y-%m-%d %H:%M")}
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
        llm.LIVE["dir"] = self.s.p("live")
        self.admin = self.cfg["ollama_admin_url"].split("://")[-1]
        self.master_blob, _ = gguf_io.ollama_show(self.cfg["master_tag"], self.admin)
        self.fac = Factory(self.state)
        self.ref = [json.loads(l) for l in open(os.path.join(ROOT, "data", "reference_traces.jsonl"))]
        self.general = general_pool(exclude={r["id"] for r in self.ref})
        self.lineage = self.s.read("lineage.json") or {"generations": [], "current": None}
        self._S = None

    # -------- bookkeeping
    def status(self, cycle, step, detail="", **kw):
        self.s.write("status.json", {"cycle": cycle, "step": step, "detail": detail, "since": time.time(), "generation": self.lineage["current"], **kw})

    @property
    def gen(self):
        return next(g for g in self.lineage["generations"] if g["gen"] == self.lineage["current"])

    @property
    def tag(self):
        return self.gen["tag"]

    def traces(self, tag, name):
        return os.path.join(self.state, "traces", tag.replace(":", "_").replace("/", "_"), name + ".jsonl")

    def stream(self):
        if self._S is None:
            from .arch_qwen35 import Qwen35Stream
            self._S = Qwen35Stream(self.master_blob, os.path.join(ROOT, "data", "qwen35_tokenizer.json"), base_dir=self.state)
        return self._S

    def materialize(self, specs, tag):
        """bake the edits into a copy of the editable master and register it with ollama"""
        if not specs and not self.cfg.get("run_quant"):
            return self.cfg["master_tag"]
        build = os.path.join(self.state, "build"); os.makedirs(build, exist_ok=True)
        src = os.path.join(build, "edit.gguf")
        if specs:
            from . import edits
            gguf_io.bake(self.master_blob, src, specs, edits.bake_fn(self.state))
        else:
            src = self.master_blob
        out = src
        if self.cfg.get("run_quant"):
            out = os.path.join(build, "run.gguf"); gguf_io.derive_run_quant(src, out, self.cfg["run_quant"])
        gguf_io.register(tag, out, self.cfg["master_tag"], self.admin)
        shutil.rmtree(build, ignore_errors=True)
        return tag

    def add_generation(self, tag, specs, note):
        n = len(self.lineage["generations"])
        self.lineage["generations"].append({"gen": n, "tag": tag, "specs": specs, "note": note, "at": time.strftime("%Y-%m-%d %H:%M")})
        self.lineage["current"] = n; self.s.write("lineage.json", self.lineage)

    def setup(self):
        if not self.lineage["generations"]:
            self.status(0, "setup", "preparing generation 0")
            self.add_generation(self.materialize([], f"rsi-{self.cfg['name']}:g0"), [], "the model as downloaded")
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

    # -------- step 2: the model decides what to change about itself
    def think(self, k, brief, rec):
        cfg = self.cfg
        msgs = [{"role": "system", "content": open(PLAYBOOK).read() + f"\n\n# Your goal (set by the human)\n\n{cfg['goal']}"},
                {"role": "user", "content": "You just solved fresh practice problems. Here is what happened:\n\n" + json.dumps(brief, indent=1) +
                 ("\n\nIf useful, first search the web or arXiv for techniques that fit what you see (a few tool calls at most). " if cfg["research"] else "\n\n") +
                 "Then decide on ONE change to your weights from the menu."}]
        rec["searches"] = []
        if cfg["research"]:
            for _ in range(4):
                r = llm.chat(cfg, self.tag, msgs, think=cfg["pilot_think"], tools=research.TOOLS, num_predict=cfg["pilot_num_predict"],
                             live={"slot": "pilot", "title": "reading its traces, deciding what to change"})
                m = r["message"]; calls = m.get("tool_calls") or []
                msgs.append({"role": "assistant", "content": m.get("content", ""), **({"tool_calls": calls} if calls else {})})
                if not calls:
                    break
                for c in calls[:2]:
                    fn = c.get("function", {}); args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    self.status(k, "think", f"searching: {fn.get('name')} {json.dumps(args)[:80]}")
                    out = research.execute(fn.get("name", ""), args)
                    rec["searches"].append({"tool": fn.get("name"), "args": args, "result": out[:600]})
                    msgs.append({"role": "tool", "content": out[:8000], "tool_name": fn.get("name", "")})
                self.status(k, "think", "reading its traces, deciding what to change")
        msgs.append({"role": "user", "content": "Now give your decision as JSON: diagnosis, edit (one of " + ", ".join(EDITS) +
                                                "), strength (inside that edit's range), why, expect."})
        for _ in range(3):
            r = llm.chat(cfg, self.tag, msgs, think=cfg["pilot_think"], fmt=DECISION, num_predict=cfg["pilot_num_predict"],
                         live={"slot": "pilot", "title": "deciding on one change"})
            try:
                d = json.loads(r["message"]["content"])
                if d.get("edit") in EDITS and isinstance(d.get("strength"), (int, float)):
                    _, lo, hi = EDITS[d["edit"]]; d["strength"] = round(min(hi, max(lo, float(d["strength"]))), 3)
                    rec["thinking"] = (r["message"].get("thinking") or "")[-3000:]
                    return d
            except ValueError:
                pass
            msgs += [{"role": "assistant", "content": r["message"].get("content", "")},
                     {"role": "user", "content": f"Not valid. Answer with JSON; edit must be one of {list(EDITS)}."}]
        return None

    # -------- step 3a: turn the decision into a weight edit
    def make_edit(self, k, d, dev_recs, dev):
        op, _, _ = EDITS[d["edit"]]
        if op == "sharpen":
            return {"op": "sharpen", "strength": d["strength"]}, {}
        S = self.stream(); specs = self.gen["specs"]
        if d["edit"] == "stop_second_guessing":
            byid = {it["id"]: it for it in dev}
            seqs, caps, y, g = directions.decision_points(S, dev_recs, byid, post_answer_only=True)
            if (y == 1).sum() < 6:
                seqs, caps, y, g = directions.decision_points(S, dev_recs, byid, post_answer_only=False)
            if (y == 1).sum() < 6 or (y == 0).sum() < 6:
                return None, {"why": f"only {(y == 1).sum()} 'Wait…' moments in its traces: too few to find the direction"}
            per_layer = S.run(seqs, cands=[specs], capture=caps)[0]["pos"]; key = "doubt"
        else:
            rtrain = [r for r in self.ref if r["split"] == "train"][:12]
            self.status(k, "test", "solving the reference problems to compare its style with the reference")
            own = llm.solve(self.cfg, self.tag, rtrain, self.traces(self.tag, "reference"), who="reference problems")
            seqs, caps, y, g = directions.reference_vs_own(S, rtrain, own)
            per_layer = S.run(seqs, cands=[specs], capture=caps)[0]["span"]; key = "style"
        rows, D = directions.stats(per_layer, y, g)
        L = S.L; good = [r for r in rows if r["layer"] >= L // 3 and r["heldout_auc"] is not None]
        best = max(good, key=lambda r: r["heldout_auc"]) if good else rows[2 * L // 3]
        os.makedirs(os.path.join(self.state, "dirs"), exist_ok=True)
        dfile = f"dirs/{key}-c{k}.npz"; np.savez(os.path.join(self.state, dfile), d=D)
        return ({"op": op, "dir": dfile, "dir_layer": best["layer"], "layers": [L // 2, L - 1], "strength": d["strength"]},
                {"layer": best["layer"], "auc": best["heldout_auc"], "samples": f"{int((y == 1).sum())}+{int((y == 0).sum())}"})

    # -------- one cycle
    def cycle(self, k, direction):
        cfg = self.cfg; t0 = time.time(); gen0 = self.gen
        rec = {"cycle": k, "generation": gen0["gen"], "direction": direction, "started": time.strftime("%H:%M")}
        # 1 observe
        dev = self.fac.sample("dev", k, cfg["n_dev_per_family"])
        self.status(k, "observe", f"solving {len(dev)} fresh practice problems")
        recs = llm.solve(cfg, self.tag, dev, self.traces(self.tag, f"practice-c{k}"), who="practice")
        summ, D = detectors.summarize(recs)
        self.fac.update_levels(recs)
        rec["practice"] = {"n": len(recs), "accuracy": round(float(np.mean([r["correct"] for r in recs])), 3),
                           "unfinished": summ["counts"]["unfinished"], "doubt_after_answer": summ["doubt_events_after_answer"],
                           "doubt_total": summ["doubt_events_total"]}
        brief = {"goal": cfg["goal"], "human_direction": direction or "(none: your call)", "practice_results": rec["practice"],
                 "accuracy_by_problem_type": summ["accuracy_by_suite"], "patterns_by_outcome": summ["detectors_mean_by_outcome"],
                 "failing_traces": detectors.excerpts(recs, D, k=3, width=600),
                 "earlier_cycles": [{"edit": h.get("edit"), "strength": h.get("strength"), "kept": h.get("kept"), "result": h.get("summary")}
                                    for h in self.s.history()[-6:]],
                 "menu": {n: f"{op}, strength {lo}-{hi}" for n, (op, lo, hi) in EDITS.items()}}
        # 2 think
        self.status(k, "think", "reading its traces, deciding what to change")
        d = self.think(k, brief, rec)
        if d is None:
            if gen0["gen"] > 0:   # the edited model can no longer run the loop: undo the last edit
                self.lineage["current"] = gen0["gen"] - 1; self.s.write("lineage.json", self.lineage)
                rec.update(kept=False, summary=f"generation {gen0['gen']} could not produce a valid decision; reverted to generation {gen0['gen'] - 1}")
                return rec
            d = {"diagnosis": "(no valid decision from the model)", "edit": "stop_second_guessing", "strength": 0.5,
                 "why": "fallback: the model did not produce a valid decision", "expect": "fewer doubt loops"}
        rec.update(diagnosis=d["diagnosis"], edit=d["edit"], strength=d["strength"], why=d["why"], expect=d["expect"])
        # 3 edit + test
        self.status(k, "test", f"finding where '{d['edit']}' lives in its weights")
        spec, info = self.make_edit(k, d, recs, dev)
        rec["edit_info"] = info
        if spec is None:
            rec.update(kept=False, summary=f"skipped: {info.get('why')}"); return rec
        ctag = f"rsi-{cfg['name']}:g{len(self.lineage['generations'])}"
        self.status(k, "test", f"baking {d['edit']} ({d['strength']}) into a copy of its weights")
        self.materialize(gen0["specs"] + [spec], ctag)
        n0 = cfg["n_general"]; start = (k * n0) % max(1, len(self.general) - n0)
        focus = self.fac.sample("heldout", k, cfg["n_test_per_family"]); general = self.general[start:start + n0]
        self.status(k, "test", f"current model on {len(focus) + len(general)} fresh test problems")
        a = llm.solve(cfg, self.tag, focus + general, self.traces(self.tag, f"test-c{k}"), who="test, current model")
        self.status(k, "test", f"edited model on the same {len(focus) + len(general)} problems")
        b = llm.solve(cfg, ctag, focus + general, self.traces(ctag, f"test-c{k}"), who="test, edited model")
        fp = compare(a, b, [it["id"] for it in focus]); gp = compare(a, b, [it["id"] for it in general])
        keep = fp["fixed"] - fp["broken"] >= cfg["min_gain"] and gp["broken"] - gp["fixed"] <= cfg["max_general_loss"]
        # 4 keep or revert
        if keep:
            self.add_generation(ctag, gen0["specs"] + [spec], f"cycle {k}: {d['edit']} {d['strength']}")
        else:
            gguf_io.unregister(ctag, self.admin)
        rec.update(test={"goal": fp, "general": gp}, kept=keep, generation_after=self.lineage["current"], minutes=round((time.time() - t0) / 60, 1),
                   summary=f"{d['edit']} {d['strength']}: goal {fp['before']:.0%} → {fp['after']:.0%} (+{fp['fixed']}/−{fp['broken']}), "
                           f"general {gp['before']:.0%} → {gp['after']:.0%} (+{gp['fixed']}/−{gp['broken']}): {'KEPT' if keep else 'reverted'}")
        return rec

    def run(self):
        open(self.s.p("loop.pid"), "w").write(str(os.getpid()))

        def term(*_):
            try:
                self.status(len(self.s.history()), "stopped", "the loop was stopped"); os.unlink(self.s.p("loop.pid"))
            except Exception:  # noqa: BLE001
                pass
            os._exit(0)
        signal.signal(signal.SIGTERM, term)
        self.setup()
        k = len(self.s.history()); go = k == 0; direction = None
        seen = sum(1 for _ in open(self.s.p("inbox.jsonl"))) if os.path.exists(self.s.p("inbox.jsonl")) else 0
        while True:
            lines = open(self.s.p("inbox.jsonl")).read().splitlines() if os.path.exists(self.s.p("inbox.jsonl")) else []
            for l in lines[seen:]:
                go = True; t = json.loads(l)["text"].strip()
                direction = t or direction
            seen = len(lines)
            if not (go or self.cfg.get("auto")):
                self.status(k, "waiting", "press Enter for the next cycle, or type what it should work on"); time.sleep(1); continue
            go = False
            try:
                rec = self.cycle(k, direction)
            except Exception as e:  # noqa: BLE001
                rec = {"cycle": k, "generation": self.lineage["current"], "kept": False, "summary": f"error: {type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-2000:]}
            with open(self.s.p("history.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            k += 1; direction = None


if __name__ == "__main__":
    Loop(sys.argv[1]).run()
