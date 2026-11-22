"""Running model-written Python safely enough for problem generators and code grading.

Layers of protection (none is perfect alone):
  1. static check for generator code: AST import allowlist + no file/OS/network/introspection builtins
  2. a fresh `python -I` subprocess in an empty temp directory with a minimal environment
  3. resource limits: CPU seconds, address space, file size, process count
  4. no network: `unshare -rn` (new user + network namespace) when the kernel allows it
"""
import ast, json, os, resource, shutil, subprocess, sys, tempfile

ALLOWED_IMPORTS = {"random", "math", "itertools", "fractions", "string", "datetime", "collections", "functools", "decimal",
                   "statistics", "re", "json", "calendar", "operator", "heapq", "bisect"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr",
                "input", "breakpoint", "memoryview", "help", "exit", "quit"}
_UNSHARE = None


def _can_unshare():
    global _UNSHARE
    if _UNSHARE is None:
        try:
            _UNSHARE = shutil.which("unshare") is not None and subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
        except Exception:  # noqa: BLE001
            _UNSHARE = False
    return _UNSHARE


def static_check(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import {a.name} not allowed (allowed: {sorted(ALLOWED_IMPORTS)})"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from {node.module} not allowed"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"use of {node.id} not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"dunder attribute {node.attr} not allowed"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "generate" for n in tree.body):
        return "must define generate(rng, level)"
    return None


def _limits(cpu, mem_mb):
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    return f


def run(code, timeout=20, cpu=15, mem_mb=1024):
    """run python source; returns (returncode, stdout, stderr). returncode -9/-24 etc. on limits."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "main.py")
        with open(path, "w") as f:
            f.write(code)
        cmd = [sys.executable, "-I", path]
        if _can_unshare():
            cmd = ["unshare", "-rn"] + cmd
        try:
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout, preexec_fn=_limits(cpu, mem_mb),
                               env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
            return p.returncode, p.stdout, p.stderr[-2000:]
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"


def sample_family(code, seeds, level):
    """execute a generator family for the given seeds; returns (items, error)"""
    err = static_check(code)
    if err:
        return [], err
    driver = code + "\n\nimport random as _r, json as _j\nfor _s in " + json.dumps(list(seeds)) + ":\n" \
        f"    _it = generate(_r.Random(_s), {int(level)})\n    _it['seed'] = _s\n    print(_j.dumps(_it))\n"
    rc, out, errs = run(driver)
    if rc != 0:
        return [], f"generator failed (rc={rc}): {errs.strip().splitlines()[-1] if errs.strip() else 'no output'}"
    try:
        return [json.loads(l) for l in out.splitlines() if l.strip()], None
    except ValueError as e:
        return [], f"bad generator output: {e}"


def grade_code(solution, tests, entry_point, timeout=15):
    """run a candidate solution against python assert tests (HumanEval style)"""
    prog = solution + "\n\n" + tests + "\n"
    rc, _, _ = run(prog, timeout=timeout, cpu=10, mem_mb=1024)
    return rc == 0


# ------------------------------------------------------------------------------------------------ the agent's workspace
def _gpu_devices():
    return [d for d in ("/dev/nvidia0", "/dev/nvidia1", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset")
            if os.path.exists(d)]


def run_in_workspace(workspace, argv, timeout=600, mem="4G", cpu_quota=None, network=False, root=None, extra_ro=()):
    """Run argv inside a bubblewrap sandbox: the workspace is the only writable place; the system, the rsi code, the
    Python environment and ollama's model files are read-only; the home directory is hidden; the network is off
    unless network=True. Memory, CPU and time are limited. Returns (exit code, combined output)."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv = os.path.realpath(os.path.dirname(os.path.dirname(sys.executable)))
    pyroot = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
    ws = os.path.realpath(workspace); os.makedirs(ws, exist_ok=True)
    b = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-user-try",
         "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
         "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    if not network:
        b += ["--unshare-net"]
    for p in {venv, pyroot, os.path.realpath(root), *[os.path.realpath(x) for x in extra_ro]}:
        if os.path.exists(p) and not p.startswith("/usr/"):
            b += ["--ro-bind", p, p]
    blobs = "/usr/share/ollama/.ollama/models/blobs"
    if os.path.exists(blobs):
        b += ["--ro-bind", blobs, blobs]
    for d in _gpu_devices():
        b += ["--dev-bind", d, d]
    b += ["--bind", ws, ws, "--chdir", ws, "--setenv", "HOME", ws, "--setenv", "PYTHONPATH", root,
          "--setenv", "RSI_WORKSPACE", ws, "--setenv", "PATH", f"{venv}/bin:/usr/bin:/bin"] + list(argv)
    if shutil.which("systemd-run"):
        props = ["-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0"] + (["-p", f"CPUQuota={cpu_quota}"] if cpu_quota else [])
        b = ["systemd-run", "--user", "--scope", "-q"] + props + b
    try:
        p = subprocess.run(b, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else ""))
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return -1, out + f"\n[killed: exceeded the {timeout}s time limit]"
