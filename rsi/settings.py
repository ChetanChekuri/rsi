"""rsi settings: every knob in one place.

    rsi settings              edit this folder's loop (or, in a folder without one, the defaults for new loops)
    rsi settings --global     edit the defaults for new loops (~/.config/rsi/settings.json)
    rsi settings list         print them
    rsi settings KEY VALUE    set one (e.g.  rsi settings pilot_num_predict 8000)

A running loop picks up changes at its next step.
"""
import json, os, sys
from .network import local_url

GLOBAL = os.path.expanduser("~/.config/rsi/settings.json")
THINK = ["on", "off", "low", "medium", "high"]

# key, section, label, type, default (or (small, large)), help
SCHEMA = [
    ("ollama_url", "Ollama", "Inference server", "str", "http://127.0.0.1:11434", "where models run (the main model and tests)"),
    ("ollama_admin_url", "Ollama", "Admin server", "str", "http://127.0.0.1:11434", "where models are created and deleted"),
    ("keep_alive", "Ollama", "Keep loaded", "str", "5m", "how long a model stays in memory after use (0 = unload at once)"),
    ("pilot_think", "Main model", "Thinking", "choice", "on", "thinking mode for the model working on itself"),
    ("pilot_num_predict", "Main model", "Thinking budget", "int", 16384, "max tokens per reply, thinking included"),
    ("num_ctx", "Main model", "Context window", "int", (32768, 65536), "holds the instructions, the conversation and the reply; older turns are shortened to fit"),
    ("pilot_temperature", "Main model", "Temperature", "float", None, "empty = the model's default"),
    ("max_steps", "Main model", "Tool calls per cycle", "int", 0, "0 = no limit"),
    ("max_cycle_minutes", "Main model", "Minutes per cycle", "int", 0, "0 = no limit"),
    ("auto_continue", "Main model", "Continue on its own", "bool", True, "start the next cycle on its own plan; off = wait for your message"),
    ("research", "Main model", "Web research tools", "bool", True, "Tinyfish only: web_search, arxiv_search, fetch_page"),
    ("sandbox_isolate", "Sandbox", "Isolation", "bool", True, "code runs in a sandbox: only the workspace is writable, your home is hidden"),
    ("sandbox_internet", "Sandbox", "Internet", "bool", False, "must stay off; use Tinyfish research tools"),
    ("sandbox_local", "Sandbox", "Local network", "bool", False, "must stay off; only local inference is exposed"),
    ("sandbox_mem", "Sandbox", "Memory cap", "str", ("4G", "48G"), "per code run; empty = no cap"),
    ("sandbox_cpu", "Sandbox", "CPU cap", "str", "", "per code run, e.g. 400% = 4 cores; empty = half the cores on CPU-only machines"),
    ("max_run_s", "Sandbox", "Time limit per run (s)", "int", 3600, "for run_python / run_shell (background jobs have none); 0 = no limit"),
    ("max_jobs", "Sandbox", "Background jobs at once", "int", (2, 4), "0 = no limit"),
    ("job_mem", "Sandbox", "Memory cap per job", "str", "", "empty = same as the memory cap per run"),
    ("test_think", "Tests (defaults)", "Thinking", "choice", "on", "the model can choose differently for each test run"),
    ("test_num_predict", "Tests (defaults)", "Token budget", "int", (4096, 16384), "per reply of the tested model"),
    ("test_max_tool_steps", "Tests (defaults)", "Tool rounds", "int", 6, "per problem, when a test gives the tested model tools"),
    ("judge_model", "Tests (defaults)", "Rubric judge", "str", "", "who grades rubric suites; empty = the model in charge when a rubric is first used"),
    ("test_workers", "Tests (defaults)", "Parallel problems", "int", 0, "0 = decided from the hardware"),
    ("max_models", "Models", "Models kept at once", "int", (8, 3), "how many models it may have registered (each is a full copy); 0 = no limit"),
]
KEYS = {k for k, *_ in SCHEMA}


def default(key, small=True):
    d = next(s for s in SCHEMA if s[0] == key)[4]
    return (d[0] if small else d[1]) if isinstance(d, tuple) else d


def defaults(small=True):
    out = {k: default(k, small) for k in KEYS}
    for k, env in (("ollama_url", "RSI_OLLAMA_URL"), ("ollama_admin_url", "RSI_OLLAMA_ADMIN_URL")):
        if os.environ.get(env):
            out[k] = os.environ[env]
    if not os.environ.get("RSI_OLLAMA_ADMIN_URL"):
        out["ollama_admin_url"] = out["ollama_url"]
    return out


def read_global():
    try:
        return json.load(open(GLOBAL))
    except (OSError, ValueError):
        return {}


def for_new_loop(small):
    return policy({**defaults(small), **{k: v for k, v in read_global().items() if k in KEYS}})


def policy(cfg):
    # Old sessions are migrated to the mandatory network boundary in memory.
    cfg = dict(cfg, sandbox_isolate=True, sandbox_internet=False, sandbox_local=False)
    for key in ("ollama_url", "ollama_admin_url"):
        cfg[key] = local_url(cfg[key])
    return cfg


def effective(cfg, sizing):
    """a loop's settings as used: defaults <- the loop's config <- what the hardware decides for 'auto' values"""
    small = cfg.get("small", True)
    out = {**defaults(small), **cfg}
    out.update(device=sizing.get("device", "cpu"), workers=int(out.get("test_workers") or sizing.get("workers") or 1),
               cpu_quota=out.get("sandbox_cpu") or sizing.get("cpu_quota"), memory_cap=sizing.get("memory_cap"))
    out["num_predict"] = out.get("test_num_predict") or out.get("num_predict", 4096)
    return policy(out)


def think(v):
    v = "on" if v is True else ("off" if v is False else str(v).lower())
    return True if v == "on" else (False if v == "off" else v)


def parse(key, text):
    typ = next(s for s in SCHEMA if s[0] == key)[3]
    text = str(text).strip()
    if key in ("ollama_url", "ollama_admin_url"):
        return local_url(text)
    if key == "sandbox_isolate" and text.lower() not in ("1", "true", "on", "yes"):
        raise ValueError("isolation is required")
    if key in ("sandbox_internet", "sandbox_local") and text.lower() not in ("0", "false", "off", "no"):
        raise ValueError("code network access is disabled; use Tinyfish research tools")
    if typ == "bool":
        if text.lower() in ("1", "true", "on", "yes"):
            return True
        if text.lower() in ("0", "false", "off", "no"):
            return False
        raise ValueError("on or off")
    try:
        if typ == "int":
            return int(text)
        if typ == "float":
            return None if text in ("", "default", "none") else float(text)
    except ValueError:
        raise ValueError("a whole number" if typ == "int" else "a number (or empty for the default)") from None
    if typ == "choice":
        if text.lower() not in THINK:
            raise ValueError(" / ".join(THINK))
        return text.lower()
    return text


def target(folder, force_global=False):
    """(path of the settings file, small?, label)"""
    p = os.path.join(folder, ".rsi", "config.json")
    if not force_global and os.path.exists(p):
        return p, json.load(open(p)).get("small", True), f"this loop ({folder})"
    return GLOBAL, True, "defaults for new loops"


def load(path, small):
    try:
        cur = json.load(open(path))
    except (OSError, ValueError):
        cur = {}
    return policy({**(defaults(small) if path != GLOBAL else for_new_loop(small)), **{k: v for k, v in cur.items() if k in KEYS}}), cur


def save(path, key, value):
    try:
        cur = json.load(open(path))
    except (OSError, ValueError):
        cur = {}
    cur[key] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=1)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ the screen
@__import__('functools').lru_cache(None)
def _screen_classes():
    from rich.text import Text
    from textual import on
    from textual.app import App
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.screen import Screen
    from textual.widgets import Footer, Input, Select, Static, Switch

    ACC, DIM, GOOD, BAD, WARN = "#5fd7d7", "#6f7d89", "#87d787", "#ff8787", "#ffd75f"

    class SettingsScreen(Screen):
        """every setting, grouped; switches and choices save at once, text fields when you press enter or leave them"""
        CSS = """
        SettingsScreen { background: #13161b; }
        #shead { padding: 1 2 0 2; height: auto; }
        #sbody { padding: 0 2; }
        .section { color: #5fd7d7; text-style: bold; margin-top: 1; border-bottom: solid #2b343e; }
        .row { height: auto; padding: 0 0 0 1; margin-top: 1; }
        .text { width: 1fr; height: auto; }
        .help { color: #6f7d89; }
        .row Input, .row Select { width: 36; }
        .row Switch { width: 10; }
        .row Input.-bad { border: tall #ff8787; }
        .row Input.-dirty { border: tall #ffd75f; }
        #smsg { dock: bottom; height: 1; padding: 0 2; background: #191e25; }
        """
        BINDINGS = [Binding("escape", "close", "Back"), Binding("ctrl+q", "close", "Back")]

        def __init__(self, path, small, label):
            super().__init__(); self.path, self.small, self.label = path, small, label
            self.vals, _ = load(path, small); self.ready = False

        def compose(self):
            yield Static(Text.assemble((" rsi settings ", f"bold #13161b on {ACC}"), (f"  {self.label}", "bold"),
                                       ("\n switches save at once; text fields when you press enter or leave them. A running loop "
                                        "uses them from its next step.", DIM)), id="shead")
            with VerticalScroll(id="sbody"):
                sec = None
                for key, section, lab, typ, _, hlp in SCHEMA:
                    if section != sec:
                        sec = section; yield Static(section, classes="section")
                    with Horizontal(classes="row"):
                        with Vertical(classes="text"):
                            yield Static(Text(lab, style="bold"))
                            dflt = default(key, self.small)
                            yield Static(Text.assemble((hlp, DIM), (f"   {key}", "#4b5864"),
                                                       (f"   default {dflt if dflt not in (None, '') else 'empty'}", "#4b5864")), classes="help")
                        v = self.vals.get(key)
                        if typ == "bool":
                            yield Switch(value=bool(v), id=f"s-{key}", disabled=key in ("sandbox_isolate", "sandbox_internet", "sandbox_local"))
                        elif typ == "choice":
                            yield Select([(c, c) for c in THINK], value=think_label(v), allow_blank=False, id=f"s-{key}")
                        else:
                            yield Input("" if v is None else str(v), id=f"s-{key}")
            yield Static("", id="smsg")
            yield Footer()

        def on_mount(self):
            self.call_after_refresh(lambda: setattr(self, "ready", True))   # widgets report their initial values: ignore those

        def action_close(self):
            if len(self.app.screen_stack) > 2:
                self.app.pop_screen()
            else:
                self.app.exit()

        def _msg(self, text, color=GOOD):
            self.query_one("#smsg", Static).update(Text(text, style=color))

        def _save(self, key, raw):
            if not self.ready:
                return False
            try:
                v = parse(key, raw)
            except ValueError as e:
                self._msg(f"{key}: expected {e}", BAD); return False
            if v != self.vals.get(key):
                save(self.path, key, v); self.vals[key] = v; self._msg(f"saved {key} = {v}")
            return True

        @on(Switch.Changed)
        def sw(self, ev):
            self._save(ev.switch.id[2:], "on" if ev.value else "off")

        @on(Select.Changed)
        def sel(self, ev):
            if ev.value is not Select.BLANK:
                self._save(ev.select.id[2:], ev.value)

        @on(Input.Changed)
        def typing(self, ev):   # validate as you type, save later
            if not self.ready:
                return
            key = ev.input.id[2:]
            try:
                v = parse(key, ev.value); ok = True
            except ValueError:
                ok = False; v = None
            ev.input.set_class(not ok, "-bad"); ev.input.set_class(ok and v != self.vals.get(key), "-dirty")

        @on(Input.Submitted)
        def submit(self, ev):
            if self._save(ev.input.id[2:], ev.value):
                ev.input.remove_class("-dirty")

        @on(Input.Blurred)
        def leave(self, ev):
            if ev.input.has_class("-dirty") and self._save(ev.input.id[2:], ev.input.value):
                ev.input.remove_class("-dirty")

    class SettingsApp(App):
        def __init__(self, path, small, label):
            super().__init__(); self.args = (path, small, label)

        def on_mount(self):
            self.push_screen(SettingsScreen(*self.args))

    return SettingsScreen, SettingsApp


def __getattr__(name):   # settings.SettingsScreen without importing textual for the command line
    if name == "SettingsScreen":
        return _screen_classes()[0]
    raise AttributeError(name)


def app(path, small, label):
    return _screen_classes()[1](path, small, label)


def think_label(v):
    return "on" if v is True or v == "on" else ("off" if v is False or v == "off" else (v if v in THINK else "on"))


def main(args, folder):
    force = "--global" in args; args = [a for a in args if a != "--global"]
    path, small, label = target(folder, force)
    if not args:
        app(path, small, label).run(); return
    vals, _ = load(path, small)
    if args[0] == "list":
        print(f"{label}  ({path})")
        sec = None
        for key, section, lab, *_ in SCHEMA:
            if section != sec:
                sec = section; print(f"\n{section}")
            print(f"  {key:30} {vals.get(key)!s:30} {lab}")
        return
    if len(args) == 2 and args[0] in KEYS:
        try:
            v = parse(args[0], args[1])
        except ValueError as e:
            sys.exit(f"{args[0]}: expected {e}")
        save(path, args[0], v); print(f"{args[0]} = {v}   ({label})"); return
    sys.exit(__doc__)
