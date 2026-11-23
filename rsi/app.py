"""rsi: one screen for a loop.

    rsi                open the loop for this folder (in a new folder: pick a model and a goal first); resumes it
    rsi PATH           the same for another folder
    rsi watch [PATH]   look without starting or resuming the loop
    rsi stop [PATH]    stop the loop (it stays stopped until you open rsi there again)
    rsi settings       ollama, thinking budget, limits, sandbox (see rsi/settings.py)

Keys: F1-F6 switch tabs (Live, Conversation, Tests, Models, Jobs, Log); ctrl+l type a message; ctrl+s stop or resume
the loop; ctrl+o settings; ctrl+q close (the loop keeps running).
"""
import glob, json, os, sys, time
from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Input, Static, TabbedContent, TabPane, TextArea
from .loop import Session, create, models
from . import settings

BG, PANEL, BORDER, TEXT, DIM = "#13161b", "#191e25", "#2b343e", "#d6dee5", "#6f7d89"
ACC, BLUE, VIOLET, ORANGE, WARN, GOOD, BAD, PINK = "#5fd7d7", "#87afff", "#c3a6ff", "#ffaf5f", "#ffd75f", "#87d787", "#ff8787", "#ff87af"
STEPS = [("thinking", BLUE), ("research", VIOLET), ("code", ACC), ("build", ORANGE), ("test", WARN), ("waiting", DIM), ("switch", GOOD), ("plan", PINK)]
STEP_COLOR = dict(STEPS) | {"error": BAD, "setup": BLUE, "stopped": BAD, "starting": DIM}
TABS = ["live", "conversation", "tests", "models", "jobs", "log"]

CSS = f"""
Screen {{ background: {BG}; color: {TEXT}; }}
#header {{ height: 2; padding: 0 1; background: {PANEL}; }}
#body {{ height: 1fr; }}
#tabs {{ width: 1fr; }}
TabbedContent {{ height: 1fr; }}
TabPane {{ padding: 0; }}
Tabs {{ background: {BG}; }}
Tab {{ color: {DIM}; }}
Tab.-active {{ color: {ACC}; text-style: bold; }}
Underline > .underline--bar {{ color: {ACC}; background: {BORDER}; }}
.pane {{ height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }}
#side {{ width: 46; padding: 0 0 0 1; }}
.card {{ height: auto; border: round {BORDER}; border-title-color: {DIM}; padding: 0 1; margin-bottom: 0; }}
#cycles-card {{ height: 1fr; }}
DataTable {{ height: 1fr; background: {BG}; }}
DataTable > .datatable--header {{ background: {PANEL}; color: {ACC}; text-style: bold; }}
DataTable > .datatable--cursor {{ background: #24303b; }}
#prompt {{ border: round {ACC}; background: {BG}; margin: 0 0; }}
#prompt:focus {{ border: round {ACC}; }}
#prompt.waiting {{ border: round {WARN}; }}
Footer {{ background: {PANEL}; }}
#new {{ padding: 1 3; }}
#new .h {{ color: {ACC}; text-style: bold; margin-top: 1; }}
#new .sub {{ color: {DIM}; }}
#model-table {{ height: 14; margin-top: 1; border: round {BORDER}; }}
#goal {{ height: 6; border: round {BORDER}; }}
#goal:focus {{ border: round {ACC}; }}
#pick {{ margin-top: 1; height: auto; }}
#err {{ color: {BAD}; margin-top: 1; }}
#start {{ margin-top: 1; }}
ConfirmStop {{ align: center middle; }}
#dialog {{ width: 64; height: auto; border: round {BAD}; background: {PANEL}; padding: 1 2; }}
#dialog .wrap {{ width: 100%; margin-top: 1; }}
#dialog Horizontal {{ height: auto; margin-top: 1; align-horizontal: right; }}
#tests-pane {{ height: 1fr; }}
#tests-table {{ height: 1fr; }}
#tests-detail {{ height: 45%; border-top: solid {BORDER}; padding: 0 1; }}
#dialog Button {{ margin-left: 2; }}
"""


# ------------------------------------------------------------------------------------------------ helpers
def ago(t):
    s = max(0, int(time.time() - (t or time.time())))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def short(tag):
    return (tag or "").split(":")[-1] if (tag or "").startswith("rsi-") else (tag or "")


def bar(frac, width=12, color=ACC):
    frac = max(0.0, min(1.0, float(frac or 0))); n = int(round(frac * width))
    return Text.assemble(("━" * n, color), ("━" * (width - n), BORDER))


def read_json(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def tail(path, n=150):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 200_000))
            return f.read().decode(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def render_call(text):
    """a tool call as the model made it: highlighted code for run_python / run_shell, compact arguments otherwise"""
    name, _, rest = text.partition("(")
    try:
        args = json.loads(rest[:-1]) if rest.endswith(")") else None
    except ValueError:
        args = None
    head = Text.assemble(("→ ", ACC), (name, f"bold {ACC}"))
    if not isinstance(args, dict):
        return Group(head, Text(rest[:600], style=DIM))
    if name in ("run_python", "run_shell") and isinstance(args.get("code") or args.get("command"), str):
        src = args.get("code") or args.get("command")
        return Group(head, Padding(Syntax(src[:3000], "python" if name == "run_python" else "bash", theme="monokai", background_color=PANEL,
                                          word_wrap=True), (0, 0, 0, 2)))
    if name == "write_file":
        body = str(args.get("content", ""))
        return Group(Text.assemble(head, (f"  {args.get('path', '')}", TEXT), (f"  ({len(body)} chars)", DIM)),
                     Padding(Text(body[:400] + ("…" if len(body) > 400 else ""), style=DIM), (0, 0, 0, 2)))
    parts = [f"{k}={json.dumps(v)[:120]}" for k, v in args.items()]
    return Group(Text.assemble(head, ("  " + "  ".join(parts), DIM)))


def render_result(text, lines=10):
    ls = text.splitlines() or [""]
    body = Text("\n".join(ls[:lines]), style="#8d9aa5")
    if len(ls) > lines:
        body.append(f"\n… {len(ls) - lines} more lines", style=f"italic {DIM}")
    return Padding(body, (0, 0, 0, 2))


def conversation(sd, limit=60):
    tr = read_json(os.path.join(sd, "live", "transcript.json"), {}) or {}
    out = []
    for e in tr.get("entries", [])[-limit:]:
        k, t = e.get("kind"), e.get("text", "")
        stamp = Text(e.get("t", "") + "  ", style=DIM)
        if k == "model":
            out += [Text.assemble(stamp, ("model", f"bold {TEXT}")), Padding(Markdown(t[:4000]), (0, 0, 0, 2))]
        elif k == "call":
            out += [Text.assemble(stamp, ("tool call", DIM)), Padding(render_call(t), (0, 0, 0, 2))]
        elif k == "result":
            out += [render_result(t)]
        elif k == "human":
            out += [Text.assemble(stamp, (" you ", f"bold {BG} on {WARN}"), ("  " + t, WARN))]
        else:
            out += [Text.assemble(stamp, ("● ", GOOD), (t, GOOD))]
        out.append(Text(""))
    return Group(*out) if out else Text("nothing yet", style=DIM)


def live(sd):
    """what is being generated right now (the most recently updated stream)"""
    fs = [f for f in glob.glob(os.path.join(sd, "live", "*.json")) if not f.endswith("transcript.json")]
    if not fs:
        return "live", Text("nothing is being generated yet", style=DIM), False
    d = read_json(max(fs, key=os.path.getmtime), {}) or {}
    on_air = not d.get("done") and time.time() - d.get("updated", 0) < 15
    parts = [Text.assemble((" ● LIVE " if on_air else " ■ last ", f"bold {BG} on {GOOD if on_air else DIM}"), ("  " + d.get("title", ""), "bold"),
                           (f"   {d.get('model', '')}", DIM))]
    if d.get("prompt"):
        parts += [Text(""), Text("problem", style=f"bold {DIM}"), Padding(Text(d["prompt"], style="#c3ced6"), (0, 0, 0, 2))]
        if d.get("gold"):
            parts.append(Padding(Text(f"answer key: {d['gold']}", style=DIM), (0, 0, 0, 2)))
    if d.get("thinking"):
        th = d["thinking"]
        parts += [Text(""), Text.assemble(("thinking", f"bold {BLUE}"), (f"   {len(th):,} chars", DIM)),
                  Padding(Text(("…" if len(th) > 6000 else "") + th[-6000:], style=f"italic #93a4b3"), (0, 0, 0, 2))]
    if d.get("content"):
        parts += [Text(""), Text("reply", style=f"bold {GOOD}"), Padding(Markdown(d["content"][-4000:]), (0, 0, 0, 2))]
    for c in d.get("tool_calls") or []:
        f = c.get("function", {}); parts.append(Padding(render_call(f"{f.get('name', '')}({json.dumps(f.get('arguments', {}))})"), (0, 0, 0, 2)))
    return d.get("title", "live"), Group(*parts), on_air


# ------------------------------------------------------------------------------------------------ new loop
class New(Screen):
    BINDINGS = [Binding("ctrl+q", "app.quit", "Quit")]

    def __init__(self, folder):
        super().__init__(); self.folder = folder; self.rows = []

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="new"):
            yield Static(Text.assemble((" ◆ rsi ", f"bold {BG} on {ACC}"), ("  a model that works on itself", "bold")))
            yield Static(Text(f"new loop in {self.folder}", style=DIM), classes="sub")
            yield Static("1 · Choose the model that will work on itself", classes="h")
            yield Static(Text("It becomes the pilot: it writes its own tests, edits its weights and decides whether to switch to a new "
                              "version. Bigger, tool-using models do far better; a 1B model can run the tools but loses track.", style=DIM), classes="sub")
            yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
            yield Static("", id="pick")
            yield Static("2 · The main prompt: what should it get better at?", classes="h")
            yield Static(Text("Every cycle after this one it plans itself, e.g. \"Get better at multi-step arithmetic word problems "
                              "without getting worse at anything else.\"", style=DIM), classes="sub")
            yield TextArea("", id="goal")
            yield Static("", id="err")
            yield Button("Start the loop", variant="success", id="start")
        yield Footer()

    def on_mount(self):
        from . import llm
        cfg = settings.for_new_loop(True); host = cfg["ollama_admin_url"].split("://")[-1]
        t = self.query_one("#model-table", DataTable)
        t.add_columns("model", "size", "can", "notes")
        try:
            ms = models(host)
        except Exception:  # noqa: BLE001
            ms = []
        for m in ms:
            caps = llm.model_caps({"ollama_url": cfg["ollama_url"]}, m["tag"])["caps"] or set()
            if caps and "completion" not in caps:
                continue   # embedding models can't be the pilot
            can = Text(" ".join(c for c in ("thinking", "tools", "vision") if c in caps) or "chat", style=GOOD if "tools" in caps else WARN)
            notes = [] if "tools" in caps else ["tools as text"]
            if m["size"].split()[-1] == "MB" or (m["size"].split()[-1] == "GB" and float(m["size"].split()[0]) < 2.5):
                notes.append("small: a weak pilot")
            self.rows.append(m["tag"]); t.add_row(m["tag"], m["size"], can, Text(", ".join(notes), style=DIM))
        if not ms:
            self.query_one("#err", Static).update(f"ollama at {host} lists no models: start ollama, or pull one first")
        t.focus()

    def chosen(self):
        t = self.query_one("#model-table", DataTable)
        return self.rows[t.cursor_row] if self.rows and t.cursor_row is not None and t.cursor_row < len(self.rows) else None

    @on(DataTable.RowHighlighted)
    def picked(self, _):
        m = self.chosen()
        self.query_one("#pick", Static).update(Text.assemble(("model  ", DIM), (m or "", f"bold {ACC}")) if m else "")

    @on(DataTable.RowSelected)
    def to_goal(self, _):
        self.query_one("#goal", TextArea).focus()

    @on(Button.Pressed, "#start")
    def start(self):
        model, goal = self.chosen(), self.query_one("#goal", TextArea).text.strip()
        err = self.query_one("#err", Static)
        if not model or len(goal) < 8:
            err.update("Choose a model and describe the goal in a sentence."); return
        err.update(Text("preparing… (checking the model, picking an editable master)", style=DIM)); self.refresh()
        try:
            create(self.folder, model, goal).start()
        except Exception as e:  # noqa: BLE001
            err.update(str(e)); return
        self.app.switch_screen(Main(self.folder))


# ------------------------------------------------------------------------------------------------ the loop's screen
class ConfirmStop(ModalScreen):
    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(Text("Stop the loop?", style=f"bold {BAD}"))
            yield Static(Text("Its current step is interrupted and background jobs end. Everything so far is kept; "
                              "ctrl+s (or opening rsi here) resumes from the last finished cycle.", style=TEXT), classes="wrap")
            with Horizontal():
                yield Button("Keep running", id="no")
                yield Button("Stop", variant="error", id="yes")

    @on(Button.Pressed)
    def done(self, ev):
        self.dismiss(ev.button.id == "yes")


class Main(Screen):
    BINDINGS = [Binding("ctrl+q", "app.quit", "Close"), Binding("ctrl+l", "focus_input", "Message"),
                Binding("ctrl+s", "toggle_loop", "Stop/resume"), Binding("ctrl+o", "settings", "Settings")] + \
               [Binding(f"f{i + 1}", f"tab('{t}')", t.capitalize(), show=i < 3) for i, t in enumerate(TABS)]

    def __init__(self, folder, autostart=True):
        super().__init__(); self.s = Session(folder); self.autostart = autostart; self.restarted = 0.0; self.sigs = {}

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Horizontal(id="body"):
            with TabbedContent(id="tabs", initial="live"):
                for t in TABS:
                    with TabPane(t.capitalize(), id=t):
                        if t == "tests":
                            with Vertical(id="tests-pane"):
                                yield DataTable(id="tests-table", zebra_stripes=True, cursor_type="row")
                                with VerticalScroll(id="tests-detail"):
                                    yield Static(Text("highlight a run to see its settings and failures", style=DIM), id="tests-detail-view")
                        else:
                            with VerticalScroll(classes="pane", id=f"{t}-box"):
                                yield Static(id=f"{t}-view")
            with Vertical(id="side"):
                yield Static(id="now", classes="card")
                yield Static(id="scores", classes="card")
                yield Static(id="plans", classes="card")
                with VerticalScroll(id="cycles-card", classes="card"):
                    yield Static(id="cycles")
        yield Input(placeholder="talk to the model: it reads this at its next step, and it sets the next cycle's goal", id="prompt")
        yield Footer()

    def on_mount(self):
        for cid, title in (("now", "now"), ("scores", "latest tests"), ("plans", "next cycle"), ("cycles-card", "cycles")):
            self.query_one(f"#{cid}").border_title = title
        t = self.query_one("#tests-table", DataTable)
        t.add_columns("cycle", "suite", "n", "settings", "scores", "vs baseline", "p", "when")
        n = len(self.s.history())
        if not self.autostart:
            self.notify("watching only: the loop is not started from this window", timeout=4)
        elif self.s.pid():
            self.notify(f"reconnected: the loop is running ({n} cycles so far)", timeout=4)
        else:
            self.s.start(); self.restarted = time.time()
            self.notify(f"resumed the loop from cycle {n}" if n else "started the loop", timeout=4)
        self.tick(); self.set_interval(0.5, self.tick)

    # -------- actions
    def action_tab(self, name):
        self.query_one("#tabs", TabbedContent).active = name

    def action_focus_input(self):
        self.query_one("#prompt", Input).focus()

    def action_settings(self):
        p = os.path.join(self.s.dir, "config.json")
        self.app.push_screen(settings.SettingsScreen(p, self.s.cfg.get("small", True), f"this loop ({self.s.folder})"))

    def action_toggle_loop(self):
        if self.s.pid():
            def go(yes):
                if yes:
                    self.s.stop(); self.notify("stopped; ctrl+s resumes", timeout=4)
            self.app.push_screen(ConfirmStop(), go)
        else:
            self.s.start(); self.restarted = time.time(); self.autostart = True
            self.notify("resumed the loop", timeout=4)

    @on(Input.Submitted, "#prompt")
    def send(self, ev):
        text = ev.value.strip(); ev.input.value = ""
        if not text:
            return
        self.s.post(text)
        self.notify("sent: the model reads it at its next step, and the next cycle follows it", timeout=4)

    # -------- rendering
    def _update(self, wid, key, make, follow=None):
        """re-render a widget only when its data changed; keep a scrolled-up view where it is"""
        if self.sigs.get(wid) == key:
            return
        self.sigs[wid] = key
        box = self.query_one(f"#{follow}", VerticalScroll) if follow else None
        at_end = box is not None and box.scroll_y >= box.max_scroll_y - 2
        self.query_one(f"#{wid}", Static).update(make())
        if box is not None and at_end:
            self.call_after_refresh(box.scroll_end, animate=False)

    def tick(self):
        cfg = self.s.cfg; st = self.s.read("status.json", {}) or {}; lin = self.s.read("lineage.json", {}) or {}
        hist = self.s.history(); running = bool(self.s.pid()); sd = self.s.dir
        if self.autostart and not running and not self.s.stopped() and time.time() - self.restarted > 60:   # died: resume it
            self.s.start(); self.restarted = time.time()
        step = st.get("step", "starting") if running else ("stopped" if self.s.stopped() else st.get("step", "starting"))
        self._header(cfg, st, lin, running, step)
        self._now(st, running, step)
        self._sidebar(hist)
        # tabs
        title, body, on_air = live(sd)
        lf = [f for f in glob.glob(os.path.join(sd, "live", "*.json")) if not f.endswith("transcript.json")]
        self._update("live-view", (title, max((os.path.getmtime(f) for f in lf), default=0)), lambda: body, follow="live-box")
        tp = os.path.join(sd, "live", "transcript.json")
        self._update("conversation-view", os.path.getmtime(tp) if os.path.exists(tp) else 0, lambda: conversation(sd), follow="conversation-box")
        self._tests()
        self._update("models-view", (json.dumps(lin, default=str), os.path.getmtime(self.s.p("state", "models.json")) if os.path.exists(self.s.p("state", "models.json")) else 0),
                     lambda: self._models(lin))
        jobs = self.s.read("state/jobs.json", {}) or {}
        self._update("jobs-view", (json.dumps(jobs, default=str), int(time.time() // 3)), lambda: self._jobs(jobs))
        lp = self.s.p("loop.log")
        self._update("log-view", os.path.getmtime(lp) if os.path.exists(lp) else 0, lambda: self._log(lp), follow="log-box")
        waiting = step == "waiting"
        inp = self.query_one("#prompt", Input); inp.set_class(waiting, "waiting")
        inp.placeholder = ("the loop is waiting for you: type the next cycle's goal and press enter" if waiting else
                           "talk to the model: it reads this at its next step, and it sets the next cycle's goal")

    def _header(self, cfg, st, lin, running, step):
        gen = lin.get("current", 0); cur = st.get("model") or cfg["model"]
        color = STEP_COLOR.get(step, ACC) if running else BAD
        pill = f" ● {step.upper()} " if running else (" ■ STOPPED " if self.s.stopped() else " ○ NOT RUNNING ")
        line1 = Text.assemble((" ◆ rsi ", f"bold {BG} on {ACC}"), ("  ", ""), (cfg["model"], "bold"),
                              *(((" → ", DIM), (short(cur), f"bold {ACC}")) if cur != cfg["model"] else ()),
                              ("   generation ", DIM), (str(gen), "bold"), ("  cycle ", DIM), (str(st.get("cycle", 0)), "bold"), ("   ", ""),
                              (pill, f"bold {BG} on {color}"), (f" {ago(st.get('since'))}" if running and st.get("since") else "", DIM),
                              (f"   {st.get('device', '')}", DIM))
        line2 = Text.assemble((" goal ", DIM), (cfg["goal"][:220] + ("…" if len(cfg["goal"]) > 220 else ""), f"italic {WARN}"))
        self.query_one("#header", Static).update(Group(line1, line2))

    def _now(self, st, running, step):
        chips = Text()
        for name, color in STEPS:
            if name == step and running:
                chips.append(f" {name} ", f"bold {BG} on {color}")
            else:
                chips.append(f" {name} ", DIM)
        lines = [chips, Text("")]
        if running:
            lines += [Text(st.get("detail", "") or "…", style="bold"), Text.assemble(("for ", DIM), (ago(st.get("since")), TEXT), ("  as ", DIM), (short(st.get("model", "")), ACC))]
        elif self.s.stopped():
            lines += [Text("stopped", style=f"bold {BAD}"), Text("ctrl+s resumes from the last finished cycle", style=DIM)]
        else:
            lines += [Text("not running", style=f"bold {BAD}"), Text("watching only: ctrl+s starts it" if not self.autostart else "restarting…", style=DIM)]
        jobs = {n: j for n, j in (self.s.read("state/jobs.json", {}) or {}).items() if j.get("state") == "running"}
        if jobs:
            lines += [Text("")] + [Text.assemble(("⚙ ", WARN), (n, WARN), (f"  {ago(j.get('t0'))}", DIM)) for n, j in list(jobs.items())[:3]]
        self.query_one("#now", Static).update(Group(*lines))

    def _sidebar(self, hist):
        runs = (self.s.read("state/test_runs.json", []) or [])[-3:]
        rows = []
        for r in reversed(runs):
            stt = r.get("settings") or {}
            flags = " ".join(x for x, v in (("think", stt.get("think") not in (False, "off", None)), ("web", stt.get("internet")), ("code", stt.get("code"))) if v)
            rows.append(Text.assemble((r["suite"].replace("tests/", ""), f"bold {ACC}"), (f"  n={r['n']}  {flags}", DIM)))
            for m, v in r["models"].items():
                vs = v.get("vs_baseline")
                rows.append(Text.assemble(("  ", ""), (f"{short(m)[:14]:14} ", TEXT), bar(v["score"], 10, GOOD if v["score"] >= 0.5 else WARN),
                                          (f" {v['score']:4.0%}", "bold"),
                                          *(((f"  +{vs['fixed']}", GOOD), (f"/−{vs['broken']}", BAD if vs["broken"] else DIM)) if vs else ())))
        self.query_one("#scores", Static).update(Group(*rows) if rows else Text("no tests yet", style=DIM))
        plans = []
        if hist and hist[-1].get("next_plans"):
            h = hist[-1]
            for i, p in enumerate(h["next_plans"]):
                plans.append(Text(("▶ " if i == h.get("picked", 0) else "  ") + str(p), style=f"bold {ACC}" if i == h.get("picked", 0) else DIM))
        self.query_one("#plans", Static).update(Group(*plans) if plans else Text("decided at the end of the cycle", style=DIM))
        key = len(hist), (hist[-1].get("summary") if hist else "")
        if self.sigs.get("cycles") != key:
            self.sigs["cycles"] = key
            rows = []
            for h in reversed(hist[-40:]):
                err = (h.get("summary") or "").startswith("error"); sw = bool(h.get("switches"))
                rows.append(Text.assemble((f"{h['cycle']:>2} ", "bold"), (short(h.get("model_after") or h.get("model") or ""), GOOD if sw else TEXT),
                                          ("  ⇄ switched" if sw else "", GOOD),
                                          (f"  {len(h.get('test_runs') or [])} tests · {round(h['minutes']) if isinstance(h.get('minutes'), (int, float)) else '?'}m", DIM)))
                summ = (h.get("summary") or "").replace("\n", " ")
                rows.append(Padding(Text(summ[:130] + ("…" if len(summ) > 130 else ""), style=BAD if err else DIM), (0, 0, 1, 3)))
            self.query_one("#cycles-card").border_title = f"cycles ({len(hist)})"
            self.query_one("#cycles", Static).update(Group(*rows) if hist else Text("the first cycle is running", style=DIM))

    def _tests(self):
        runs = self.s.read("state/test_runs.json", []) or []
        if self.sigs.get("tests") == len(runs):
            return
        self.sigs["tests"] = len(runs)
        t = self.query_one("#tests-table", DataTable); t.clear()
        self.run_ids = []
        for r in reversed(runs):
            self.run_ids.append(r.get("run_id"))
            stt = r.get("settings") or {}
            flags = " ".join(x for x, v in (("think", stt.get("think") not in (False, "off", None)), ("web", stt.get("internet")), ("code", stt.get("code"))) if v)
            scores = Text()
            vs_txt, ps = Text(), []
            for m, v in r["models"].items():
                scores.append(f"{short(m)[:12]} ", TEXT); scores.append(f"{v['score']:.0%}  ", "bold")
                if v.get("vs_baseline"):
                    b = v["vs_baseline"]; vs_txt.append(f"+{b['fixed']}", GOOD); vs_txt.append(f"/−{b['broken']} ", BAD if b["broken"] else DIM); ps.append(b["p"])
            t.add_row(str(r.get("cycle", "")), r["suite"].replace("tests/", ""), str(r["n"]), Text(flags, style=DIM), scores, vs_txt,
                      Text(f"{min(ps):.3f}" if ps else "", style=GOOD if ps and min(ps) < 0.05 else DIM), Text(str(r.get("at", ""))[5:], style=DIM))

    @on(DataTable.RowHighlighted, "#tests-table")
    def test_detail(self, ev):
        rid = (getattr(self, "run_ids", []) or [None])[ev.cursor_row] if ev.cursor_row is not None and ev.cursor_row < len(getattr(self, "run_ids", [])) else None
        self.query_one("#tests-detail-view", Static).update(self._test_detail(rid))

    def _test_detail(self, rid):
        d = os.path.join(self.s.folder, "workspace", "test_runs", str(rid))
        summ = read_json(os.path.join(d, "summary.json"))
        if not summ:
            return Text(f"no saved files for run {rid}", style=DIM)
        st = summ.get("settings", {})
        out = [Text.assemble((f"{summ.get('suite')}", f"bold {ACC}"), (f"   run {rid}   graded by {summ.get('graded_by', '?')}   {summ.get('files', '')}", DIM)),
               Text("settings  " + "  ".join(f"{k}={v}" for k, v in st.items() if v not in (None, False, "")), style=DIM), Text("")]
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            try:
                recs = [json.loads(l) for l in open(f) if l.strip()]
            except (OSError, ValueError):
                continue
            bad = [r for r in recs if r.get("score", 1) < 0.5]
            out.append(Text.assemble((os.path.basename(f)[:-6], "bold"), (f"   {len(recs) - len(bad)}/{len(recs)} right", GOOD if not bad else TEXT)))
            for r in bad[:3]:
                reply = (r.get("content") or "").strip().replace("\n", " ")
                out.append(Padding(Text.assemble((f"{r.get('id')}  ", WARN), (f"wanted {str(r.get('answer'))[:30]}  ", DIM),
                                                 (("cut off  " if r.get("done_reason") == "length" else ""), BAD),
                                                 ("reply: …" + reply[-160:] if reply else "(empty reply)", "#8d9aa5")), (0, 0, 0, 2)))
            out.append(Text(""))
        return Group(*out)

    def _models(self, lin):
        cur = lin.get("current"); tb = Table(box=None, expand=True, header_style=f"bold {ACC}", pad_edge=False, title_justify="left",
                                             title=Text("lineage: what it has been", style=f"bold {TEXT}"))
        for c in ("gen", "model", "how it got here", "when"):
            tb.add_column(c)
        for g in lin.get("generations", []):
            mark = ("▶ " if g["gen"] == cur else "  ") + str(g["gen"])
            style = f"bold {ACC}" if g["gen"] == cur else (BAD if g.get("reverted") else TEXT)
            tb.add_row(Text(mark, style=style), Text(short(g["tag"]), style=style),
                       Text((g.get("note") or "") + (f"  (undone: {g['reverted']})" if g.get("reverted") else ""), style=BAD if g.get("reverted") else DIM),
                       Text(str(g.get("at", "")), style=DIM))
        built = self.s.read("state/models.json", []) or []
        mt = Table(box=None, expand=True, header_style=f"bold {ACC}", pad_edge=False, title_justify="left",
                   title=Text(f"models it has built ({len(built)})", style=f"bold {TEXT}"))
        for c in ("model", "from", "Modelfile changes", "cycle"):
            mt.add_column(c)
        for m in reversed(built):
            mt.add_row(short(m["tag"]), Text(str(m.get("from", ""))[:40], style=DIM), Text((m.get("modelfile") or "").strip().replace("\n", "; ")[:60], style=DIM),
                       str(m.get("cycle", "")))
        return Group(tb, Text(""), mt if built else Text("none yet", style=DIM))

    def _jobs(self, jobs):
        if not jobs:
            return Text("no background jobs yet (the model starts them with start_job for long work)", style=DIM)
        out = []
        color = {"running": WARN, "done": GOOD, "failed": BAD, "stopped": DIM, "lost": BAD}
        for n, j in sorted(jobs.items(), key=lambda kv: kv[1].get("t0", 0), reverse=True):
            st = j.get("state", "")
            out += [Text.assemble((f" {st} ", f"bold {BG} on {color.get(st, DIM)}"), (f"  {n}", "bold"),
                                  (f"   {ago(j.get('t0')) if st == 'running' else str(j.get('minutes', '')) + ' min'}", DIM),
                                  (f"   exit {j['exit_code']}" if j.get("exit_code") is not None else "", DIM)),
                    Padding(Syntax(j.get("command", "")[:400], "bash", theme="monokai", background_color=PANEL, word_wrap=True), (0, 0, 0, 2))]
            if st == "running":
                log = tail(os.path.join(self.s.folder, "workspace", j.get("log", "")), 8)
                out.append(Padding(Text("\n".join(log), style="#8d9aa5"), (0, 0, 0, 2)))
            out.append(Text(""))
        return Group(*out)

    def _log(self, lp):
        out = Text(); tb = False
        for l in tail(lp, 200):
            if l.startswith("Traceback"):
                tb = True
            elif tb and not l.startswith(" "):
                tb = False; out.append(l + "\n", style=f"bold {BAD}"); continue
            bad = tb or any(k in l for k in ("Error", "error:", "Exception"))
            out.append(l + "\n", style=BAD if bad else "#8d9aa5")
        return out if out.plain else Text("the loop has written nothing yet", style=DIM)


class RSI(App):
    CSS = CSS
    TITLE = "rsi"

    def __init__(self, folder, autostart=True):
        super().__init__(); self.folder = folder; self.autostart = autostart

    def on_mount(self):
        self.push_screen(Main(self.folder, self.autostart) if Session(self.folder).exists() else New(self.folder))


def main():
    args = sys.argv[1:]
    if args[:1] in (["-h"], ["--help"], ["help"]):
        print(__doc__); return
    if args[:1] == ["settings"]:
        return settings.main(args[1:], os.getcwd())
    if args[:1] == ["stop"]:
        s = Session(args[1] if len(args) > 1 else os.getcwd())
        print("stopped" if s.stop() else "no loop is running here"); return
    watch = args[:1] == ["watch"]
    if watch:
        args = args[1:]
        if not Session(os.path.abspath(args[0] if args else os.getcwd())).exists():
            sys.exit("no loop in this folder")
    RSI(os.path.abspath(args[0] if args else os.getcwd()), autostart=not watch).run()


if __name__ == "__main__":
    main()
