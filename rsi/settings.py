"""rsi settings: every knob in one place.

    rsi settings              edit this folder's loop (or, in a folder without one, the defaults for new loops)
    rsi settings --global     edit the defaults for new loops (~/.config/rsi/settings.json)
    rsi settings list         print them
    rsi settings KEY VALUE    set one (e.g.  rsi settings pilot_num_predict 8000)

A running loop picks up changes at its next step.
"""
import json, os, sys

GLOBAL = os.path.expanduser("~/.config/rsi/settings.json")
THINK = ["on", "off", "low", "medium", "high"]

# key, section, label, type, default (or (small, large)), help
SCHEMA = [
    ("ollama_url", "Ollama", "Inference server", "str", "http://127.0.0.1:11434", "where models run (the main model and tests)"),
    ("ollama_admin_url", "Ollama", "Admin server", "str", "http://127.0.0.1:11434", "where models are created and deleted"),
    ("keep_alive", "Ollama", "Keep loaded", "str", "5m", "how long a model stays in memory after use (0 = unload at once)"),
    ("pilot_think", "Main model", "Thinking", "choice", "on", "thinking mode for the model working on itself"),
    ("pilot_num_predict", "Main model", "Thinking budget", "int", (8192, 16384), "max tokens per reply, thinking included"),
    ("num_ctx", "Main model", "Context window", "int", (32768, 65536), "tokens of conversation it sees (older turns are shortened)"),
    ("pilot_temperature", "Main model", "Temperature", "float", None, "empty = the model's default"),
    ("max_steps", "Main model", "Tool calls per cycle", "int", 0, "0 = no limit"),
    ("max_cycle_minutes", "Main model", "Minutes per cycle", "int", 0, "0 = no limit"),
    ("auto_continue", "Main model", "Continue on its own", "bool", True, "start the next cycle on its own plan; off = wait for your message"),
    ("require_tests_before_switch", "Main model", "Test before switching", "bool", True, "it may only switch to a model it has tested"),
    ("research", "Main model", "Web research tools", "bool", True, "web_search, arxiv_search, fetch_page"),
    ("sandbox_isolate", "Sandbox", "Isolation", "bool", True, "code runs in a sandbox: only the workspace is writable, your home is hidden"),
    ("sandbox_internet", "Sandbox", "Internet", "bool", True, "code can reach the public internet (pip, downloads, APIs)"),
    ("sandbox_local", "Sandbox", "Local network", "bool", False, "code can also reach this machine's services, including ollama's admin API"),
    ("sandbox_mem", "Sandbox", "Memory cap", "str", ("4G", "48G"), "per code run; empty = no cap"),
    ("sandbox_cpu", "Sandbox", "CPU cap", "str", "", "per code run, e.g. 400% = 4 cores; empty = half the cores on CPU-only machines"),
    ("max_run_s", "Sandbox", "Time limit per run (s)", "int", 3600, "0 = no limit"),
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
    return {**defaults(small), **{k: v for k, v in read_global().items() if k in KEYS}}


def effective(cfg, sizing):
    """a loop's settings as used: defaults <- the loop's config <- what the hardware decides for 'auto' values"""
    small = cfg.get("small", True)
    out = {**defaults(small), **cfg}
    out.update(device=sizing.get("device", "cpu"), workers=int(out.get("test_workers") or sizing.get("workers") or 1),
               cpu_quota=out.get("sandbox_cpu") or sizing.get("cpu_quota"), memory_cap=sizing.get("memory_cap"))
    out["num_predict"] = out.get("test_num_predict") or out.get("num_predict", 4096)
    return out


def think(v):
    v = "on" if v is True else ("off" if v is False else str(v).lower())
    return True if v == "on" else (False if v == "off" else v)


def parse(key, text):
    typ = next(s for s in SCHEMA if s[0] == key)[3]
    text = str(text).strip()
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
    return {**(defaults(small) if path != GLOBAL else for_new_loop(small)), **{k: v for k, v in cur.items() if k in KEYS}}, cur


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
def app(path, small, label):
    from rich.text import Text
    from textual import on
    from textual.app import App
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Footer, Input, Select, Static, Switch

    vals, _ = load(path, small)

    class Settings(App):
        CSS = """
        Screen { background: #1b1e23; }
        #head { padding: 1 2 0 2; }
        VerticalScroll { padding: 0 2; }
        .section { color: #5fd7d7; text-style: bold; margin-top: 1; }
        .row { height: auto; margin-bottom: 1; }
        .text { width: 1fr; }
        .help { color: grey; }
        .row Input, .row Select { width: 34; }
        .row Switch { width: 12; }
        #msg { dock: bottom; height: 1; padding: 0 2; color: #87d787; }
        """
        BINDINGS = [("ctrl+q", "quit", "close"), ("escape", "quit", "close")]

        def compose(self):
            yield Static(Text.assemble((" rsi settings ", "bold black on #5fd7d7"), (f"  {label}", "bold"),
                                       ("\n changes are saved at once; a running loop uses them from its next step", "grey58")), id="head")
            with VerticalScroll():
                sec = None
                for key, section, lab, typ, _, hlp in SCHEMA:
                    if section != sec:
                        sec = section; yield Static(section, classes="section")
                    with Horizontal(classes="row"):
                        with Vertical(classes="text"):
                            yield Static(lab)
                            yield Static(f"{hlp}   [{key}]", classes="help")
                        v = vals.get(key)
                        if typ == "bool":
                            yield Switch(value=bool(v), id=f"s-{key}")
                        elif typ == "choice":
                            yield Select([(c, c) for c in THINK], value=think_label(v), allow_blank=False, id=f"s-{key}")
                        else:
                            yield Input("" if v is None else str(v), id=f"s-{key}")
            yield Static("", id="msg")
            yield Footer()

        def on_mount(self):
            self.ready = False
            self.call_after_refresh(lambda: setattr(self, "ready", True))   # widgets report their initial values: ignore those

        def _set(self, key, raw):
            if not getattr(self, "ready", False):
                return
            try:
                v = parse(key, raw)
            except ValueError as e:
                self.query_one("#msg", Static).update(Text(f"{key}: expected {e}", style="#ff8787"))
                return
            if v == vals.get(key):
                return
            save(path, key, v); vals[key] = v
            self.query_one("#msg", Static).update(f"saved {key} = {v}")

        @on(Switch.Changed)
        def sw(self, ev):
            self._set(ev.switch.id[2:], "on" if ev.value else "off")

        @on(Select.Changed)
        def sel(self, ev):
            if ev.value is not Select.BLANK:
                self._set(ev.select.id[2:], ev.value)

        @on(Input.Changed)
        def inp(self, ev):
            self._set(ev.input.id[2:], ev.value)

    return Settings()


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
