"""rsi: one screen. Left: what the model is producing right now (scrolls as it streams). Right: what it is doing, its
latest test results, the plans it made for the next cycle, and every cycle so far. Bottom: talk to it.

    rsi            open the loop for this folder (in a new folder: pick a model and a goal first)
    rsi stop       stop this folder's loop (until you open rsi there again)
    rsi settings   ollama, thinking budget, limits, sandbox (see rsi/settings.py)
"""
import glob, json, os, sys, time
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Select, Static, TextArea
from .loop import Session, create, models
from . import settings

ACC, WARN, GOOD, BAD, DIM = "#5fd7d7", "#ffd75f", "#87d787", "#ff8787", "grey58"
STEPS = ["thinking", "research", "code", "build", "test", "waiting", "switch", "plan"]
KIND = {"call": (ACC, "→ "), "result": ("#8a98a3", "  "), "model": ("#e6edf2", ""), "human": (WARN, "you: "), "note": (GOOD, "· ")}
CSS = """
Screen { background: #1b1e23; }
#top { height: 3; padding: 0 1; background: #22272e; }
#main { height: 1fr; }
#live-box { width: 3fr; border: round #5fd7d7; }
#side { width: 2fr; }
#now { height: auto; border: round #3c4a55; padding: 0 1; }
#cycles-box { height: 1fr; border: round #3c4a55; }
#prompt { dock: bottom; border: tall #5fd7d7; }
#prompt.waiting { border: tall #ffd75f; }
#new { padding: 1 4; }
#new Label { margin-top: 1; color: #9fb3bf; }
#goal { height: 5; }
#err { color: #ff8787; margin-top: 1; }
"""


class New(Screen):
    def __init__(self, folder):
        super().__init__(); self.folder = folder

    def compose(self) -> ComposeResult:
        host = settings.for_new_loop(True)["ollama_admin_url"].split("://")[-1]
        try:
            ms = models(host)
        except Exception:  # noqa: BLE001
            ms = []
        with VerticalScroll(id="new"):
            yield Static(Text.assemble((" rsi ", f"bold black on {ACC}"), ("  a model that works on itself: it writes its own tests, changes itself, and decides whether to switch", "bold")))
            yield Static(Text(f"new loop in {self.folder}   (settings for new loops: rsi settings --global)", style=DIM))
            yield Label("Model")
            yield Select([(f"{m['tag']}   ({m['size']})", m["tag"]) for m in ms], prompt="choose a model", id="model")
            yield Label("Main prompt: what should it get better at?  (it plans every cycle after this on its own)")
            yield TextArea("", id="goal")
            yield Static("", id="err")
            yield Button("Start", variant="success", id="start")
        yield Footer()

    @on(Button.Pressed, "#start")
    def start(self):
        model, goal = self.query_one("#model", Select).value, self.query_one("#goal", TextArea).text.strip()
        err = self.query_one("#err", Static)
        if model in (None, Select.BLANK) or len(goal) < 8:
            err.update("Choose a model and describe the goal in a sentence."); return
        try:
            create(self.folder, model, goal).start()
        except Exception as e:  # noqa: BLE001
            err.update(str(e)); return
        self.app.switch_screen(Main(self.folder))


def live_view(sd):
    """the agent's transcript (its messages, tool calls and results, your messages) and, below it, what it is
    generating right now"""
    parts = []
    try:
        tr = json.load(open(os.path.join(sd, "live", "transcript.json")))
        for e in tr.get("entries", [])[-40:]:
            color, prefix = KIND.get(e["kind"], (DIM, ""))
            text = e["text"] if e["kind"] != "result" else e["text"][:1200]
            parts.append(Text.assemble((e["t"] + " ", DIM), (prefix + text, color)))
    except (OSError, ValueError):
        pass
    fs = [f for f in glob.glob(os.path.join(sd, "live", "*.json")) if not f.endswith("transcript.json")]
    title = "live"
    if fs:
        f = max(fs, key=os.path.getmtime)
        try:
            d = json.load(open(f))
            live = not d.get("done") and time.time() - d.get("updated", 0) < 15
            title = d.get("title", "") + ("   ● live" if live else "")
            if live or not parts:
                parts.append(Text(""))
                if d.get("prompt"):
                    parts += [Text(d["prompt"], style="#c8d3da")] + ([Text(f"answer key: {d['gold']}", style=DIM)] if d.get("gold") else [])
                if d.get("thinking"):
                    parts += [Text("thinking", style=f"bold {WARN}"), Text(d["thinking"][-6000:], style="italic #aebbc4")]
                if d.get("content"):
                    parts += [Text("answer", style=f"bold {GOOD}"), Text(d["content"][-3000:])]
        except (OSError, ValueError):
            pass
    return title, Group(*parts) if parts else Text("nothing yet", style=DIM)


class Main(Screen):
    BINDINGS = [("ctrl+q", "app.quit", "quit (the loop keeps running)")]

    def __init__(self, folder):
        super().__init__(); self.s = Session(folder); self.sig = None; self.restarted = 0.0

    def compose(self) -> ComposeResult:
        yield Static(id="top")
        with Horizontal(id="main"):
            with VerticalScroll(id="live-box"):
                yield Static(id="live")
            with Vertical(id="side"):
                yield Static(id="now")
                with VerticalScroll(id="cycles-box"):
                    yield Static(id="cycles")
        yield Input(placeholder="talk to the model while it works; your message also sets the next cycle's direction", id="prompt")
        yield Footer()

    def on_mount(self):
        n = len(self.s.history())
        if self.s.pid():
            self.notify(f"reconnected: the loop is running ({n} cycles so far)", timeout=4)
        else:
            self.s.start(); self.restarted = time.time()
            self.notify(f"resumed the loop from cycle {n}" if n else "started the loop", timeout=4)
        self.tick(); self.set_interval(0.5, self.tick)
        self.query_one("#prompt", Input).focus()

    def tick(self):
        cfg, st = self.s.cfg, self.s.read("status.json", {}) or {}
        lin = self.s.read("lineage.json", {}) or {}; hist = self.s.history(); running = bool(self.s.pid())
        if not running and not self.s.stopped() and time.time() - self.restarted > 60:   # died (crash, reboot): resume it
            self.s.start(); self.restarted = time.time()
        step = st.get("step", "starting")
        top = Text.assemble((" rsi ", f"bold black on {ACC}"), (f"  {cfg['model']}", "bold"), (f"   generation {lin.get('current', 0)}", ACC),
                            (f"   cycle {st.get('cycle', 0)}", DIM), ("   ", ""),
                            (f" {step.upper()} ", f"bold black on {BAD if (not running or step == 'error') else ACC}"),
                            ("" if running else ("  stopped (rsi stop); open rsi here again to resume" if self.s.stopped() else "  loop not running: restarting"), BAD),
                            (f"   {st.get('device', '')}", DIM),
                            (f"\n goal: {cfg['goal'][:150]}", f"italic {WARN}"))
        self.query_one("#top", Static).update(top)

        # live pane: re-render when it changed; follow the bottom unless the user scrolled up
        title, body = live_view(self.s.dir)
        box = self.query_one("#live-box", VerticalScroll)
        sig = (title, len(str(body.renderables[-1])) if isinstance(body, Group) and body.renderables else 0)
        if sig != self.sig:
            at_bottom = box.scroll_y >= box.max_scroll_y - 2
            box.border_title = title or "live"
            self.query_one("#live", Static).update(body)
            if at_bottom or self.sig is None or sig[0] != self.sig[0]:
                self.call_after_refresh(box.scroll_end, animate=False)
            self.sig = sig
        # what it is doing now
        line = Text()
        for s_ in STEPS:
            on_ = s_ == step
            line.append("● " if on_ else "○ ", ACC if on_ else DIM); line.append(s_ + "   ", f"bold {ACC}" if on_ else DIM)
        now = [line, Text(st.get("detail", ""), style="bold"), Text(f"now running as {st.get('model', '')}", style=DIM)]
        jobs = self.s.read("state/jobs.json", {}) or {}
        live_jobs = {n: j for n, j in jobs.items() if j.get("state") == "running"}
        if live_jobs:
            now += [Text(""), Text("background jobs", style=DIM)] + [
                Text.assemble((f"● {n}", WARN), (f"  {int((time.time() - j.get('t0', time.time())) / 60)} min  {j['command'][:70]}", DIM)) for n, j in live_jobs.items()]
        runs = (self.s.read("state/test_runs.json", []) or [])[-4:]
        if runs:
            now += [Text(""), Text("latest tests", style=DIM)]
            for r in reversed(runs):
                st_ = r.get("settings") or {}
                opts = ", ".join(x for x, on in (("think", st_.get("think") not in (False, "off")), ("internet", st_.get("internet")), ("code", st_.get("code"))) if on)
                scores = "   ".join(f"{m.split(':')[-1]} {v['score']:.0%}" + (f" (+{v['vs_baseline']['fixed']}/−{v['vs_baseline']['broken']})" if v.get("vs_baseline") else "")
                                   for m, v in r["models"].items())
                now += [Text.assemble((f"{r['suite']}", ACC), (f"  n={r['n']}  {opts}", DIM)), Text("  " + scores)]
        if hist:
            h = hist[-1]
            now += [Text(""), Text(f"last cycle ({h.get('tools_used', 0)} tool calls, {h.get('minutes', '?')} min)", style=DIM),
                    Text(h.get("summary", ""), style="italic")]
            for sw in h.get("switches") or []:
                now += [Text(f"switched {sw['from'].split(':')[-1]} → {sw['to'].split(':')[-1]}: {sw.get('reason', '')}", style=GOOD)]
            if h.get("next_plans"):
                now += [Text(""), Text("its plans for the next cycle", style=DIM)]
                for i, p in enumerate(h["next_plans"]):
                    chosen = i == h.get("picked", 0)
                    now += [Text(("▶ " if chosen else "  ") + str(p), style=f"bold {ACC}" if chosen else DIM)]
                if h.get("pick_reason"):
                    now += [Text(f"picked because: {h['pick_reason']}", style="italic")]
        ms = self.s.read("state/models.json", []) or []
        if ms:
            now += [Text(""), Text("models it has built: " + ", ".join(m["tag"].split(":")[-1] for m in ms[-8:]), style=DIM)]
        self.query_one("#now", Static).update(Group(*now))
        # all cycles
        tb = Table(box=None, expand=True, header_style=f"bold {ACC}", pad_edge=False)
        for c in ("#", "model after", "goal from", "tests", "switched", "summary"):
            tb.add_column(c)
        for h in reversed(hist):
            src = (h.get("direction") or {}).get("from", "goal") if isinstance(h.get("direction"), dict) else "goal"
            sw = h.get("switches") or []
            err = (h.get("summary") or "").startswith("error")
            tb.add_row(str(h["cycle"]), (h.get("model_after") or h.get("model") or "").split(":")[-1], {"human": "you", "model": "itself"}.get(src, src),
                       str(len(h.get("test_runs") or [])), Text("yes" if sw else "no", style=GOOD if sw else DIM),
                       Text((h.get("summary") or "")[:60], style=BAD if err else ""))
        self.query_one("#cycles-box", VerticalScroll).border_title = f"cycles ({len(hist)})"
        self.query_one("#cycles", Static).update(tb if hist else Text("the first cycle is running", style=DIM))

    @on(Input.Submitted, "#prompt")
    def send(self, ev):
        text = ev.value.strip(); ev.input.value = ""
        if not text:
            return
        self.s.post(text)
        self.notify("sent: the model reads it at its next step, and the next cycle follows it instead of its own pick", timeout=4)


class RSI(App):
    CSS = CSS
    TITLE = "rsi"

    def __init__(self, folder):
        super().__init__(); self.folder = folder

    def on_mount(self):
        self.push_screen(Main(self.folder) if Session(self.folder).exists() else New(self.folder))


def main():
    args = sys.argv[1:]
    if args[:1] == ["settings"]:
        return settings.main(args[1:], os.getcwd())
    if args[:1] == ["stop"]:
        s = Session(args[1] if len(args) > 1 else os.getcwd())
        print("stopped" if s.stop() else "no loop is running here"); return
    RSI(os.path.abspath(args[0] if args else os.getcwd())).run()


if __name__ == "__main__":
    main()
