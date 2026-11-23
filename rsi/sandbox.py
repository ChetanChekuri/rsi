"""Where the model's code runs.

By default (isolate on): a bubblewrap sandbox in its own namespaces, inside a systemd scope with memory and CPU caps.
  - the workspace is the only writable place and the only one that persists
  - the system, the rsi code, the Python environment and ollama's model files are read-only (the model reads its own
    weights from there); the home directory is hidden
  - internet: through the netgate proxy (public addresses only); this machine's local services are not reachable,
    except ollama for running and inspecting models (see netgate.py)
  - `pip install` works: packages go to <workspace>/.pkgs, which is on PYTHONPATH
With `local` on, the sandbox shares this machine's network (everything local is reachable too). With isolate off, the
code runs as you, in the workspace folder, with no walls at all.
"""
import os, shutil, subprocess, sys
from . import netgate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_dirs():
    """where ollama keeps model files on this machine (system service, user install, snap, or $OLLAMA_MODELS)"""
    roots = [os.environ.get("OLLAMA_MODELS", ""), "/usr/share/ollama/.ollama/models", os.path.expanduser("~/.ollama/models"),
             "/var/lib/ollama/models", "/var/snap/ollama/common/models"]
    return sorted({os.path.realpath(os.path.join(r, "blobs")) for r in roots if r and os.path.isdir(os.path.join(r, "blobs"))})


def _gpu_devices():
    return [d for d in ("/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2", "/dev/nvidia3", "/dev/nvidiactl", "/dev/nvidia-uvm",
                        "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/dri", "/dev/kfd") if os.path.exists(d)]


def command(workspace, argv, mem="4G", cpu_quota=None, internet=True, local=False, isolate=True,
            ollama_url="http://127.0.0.1:11434", gate_dir=None, env=None, ro=()):
    """-> (cmd, env for subprocess or None, cwd): argv wrapped in the sandbox with the workspace as its home.
    ro: extra folders to show read-only (e.g. where the model's own GGUF lives)"""
    ws = os.path.realpath(workspace); os.makedirs(os.path.join(ws, ".pkgs"), exist_ok=True)
    venv = os.path.realpath(os.path.dirname(os.path.dirname(sys.executable)))
    pyroot = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
    w = "/workspace" if isolate else ws   # inside the sandbox the workspace is /workspace (what models expect)
    base_env = {"HOME": w, "RSI_WORKSPACE": w, "PYTHONPATH": f"{ROOT}:{w}/.pkgs", "PIP_TARGET": f"{w}/.pkgs",
                "PIP_BREAK_SYSTEM_PACKAGES": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PATH": f"{venv}/bin:{ROOT}/rsi/sbin:{w}/.pkgs/bin:/usr/local/bin:/usr/bin:/bin",
                "LANG": "C.UTF-8", "TERM": "dumb", **(env or {})}
    if not isolate:
        base_env["OLLAMA_HOST"] = ollama_url.split("://")[-1].rstrip("/")
        cmd = list(argv); run_env = dict(os.environ, **base_env)
    else:
        b = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-user-try",
             "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
             "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for p in {venv, pyroot, ROOT}:
            if os.path.exists(p) and not p.startswith("/usr/"):
                b += ["--ro-bind", p, p]
        for d in sorted(set(model_dirs()) | {os.path.realpath(x) for x in ro if x and os.path.isdir(x)}):
            if not d.startswith("/usr/"):
                b += ["--ro-bind", d, d]
        for d in _gpu_devices():
            b += ["--dev-bind", d, d]
        b += ["--bind", ws, ws, "--bind", ws, "/workspace", "--chdir", "/workspace", "--clearenv"]   # also at its real path
        inner = list(argv)
        if local:
            base_env["OLLAMA_HOST"] = ollama_url.split("://")[-1].rstrip("/")
        else:
            b += ["--unshare-net"]
            doors = netgate.ensure(gate_dir or os.path.join(ws, ".net"), ollama_url)
            b += ["--ro-bind", os.path.dirname(doors["ollama"]), "/run/rsi"]
            pairs = ["/run/rsi/ollama.sock:11434"]
            base_env["OLLAMA_HOST"] = "127.0.0.1:11434"
            if internet:
                pairs.append("/run/rsi/proxy.sock:3128")
                for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
                    base_env[k] = "http://127.0.0.1:3128"
                base_env["NO_PROXY"] = base_env["no_proxy"] = "127.0.0.1,localhost"
            inner = [os.path.join(venv, "bin", "python"), "-m", "rsi.netgate", "bridge", *pairs, "--"] + inner
        for k, v in base_env.items():
            b += ["--setenv", k, v]
        cmd = b + inner; run_env = None
    if shutil.which("systemd-run"):
        props = (["-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0"] if mem else []) + (["-p", f"CPUQuota={cpu_quota}"] if cpu_quota else [])
        if props:
            cmd = ["systemd-run", "--user", "--scope", "-q"] + props + cmd
    return cmd, run_env, ws


def run_in_workspace(workspace, argv, timeout=3600, **kw):
    """Run argv in the sandbox and wait. Returns (exit code, combined output). timeout None/0 = no limit."""
    cmd, run_env, ws = command(workspace, argv, **kw)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout or None, stdin=subprocess.DEVNULL,
                           cwd=ws, env=run_env)
        return p.returncode, (p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else ""))
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return -1, out + f"\n[killed: exceeded the {timeout}s time limit]"


def start_in_workspace(workspace, argv, log_path, **kw):
    """Start argv in the sandbox without waiting (a background job); output goes to log_path. Returns the Popen.
    The job dies with the loop (bubblewrap's --die-with-parent; a new session otherwise, killed by the loop)."""
    cmd, run_env, ws = command(workspace, argv, **kw)
    log = open(log_path, "ab", buffering=0)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ws, env=run_env, start_new_session=True)
