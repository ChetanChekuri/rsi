"""Isolated, offline code execution. Web research goes through host-owned Tinyfish tools."""
import os, shutil, subprocess, sys
from . import netgate, network

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_dirs():
    """where ollama keeps model files on this machine (system service, user install, snap, or $OLLAMA_MODELS)"""
    roots = [os.environ.get("OLLAMA_MODELS", ""), "/usr/share/ollama/.ollama/models", os.path.expanduser("~/.ollama/models"),
             "/var/lib/ollama/models", "/var/snap/ollama/common/models"]
    return sorted({os.path.realpath(os.path.join(r, "blobs")) for r in roots if r and os.path.isdir(os.path.join(r, "blobs"))})


def _gpu_devices():
    return [d for d in ("/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2", "/dev/nvidia3", "/dev/nvidiactl", "/dev/nvidia-uvm",
                        "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/dri", "/dev/kfd") if os.path.exists(d)]


def command(workspace, argv, mem="4G", cpu_quota=None, internet=False, local=False, isolate=True,
            ollama_url="http://127.0.0.1:11434", gate_dir=None, env=None, ro=()):
    """-> (cmd, env for subprocess or None, cwd): argv wrapped in the sandbox with the workspace as its home.
    ro: extra folders to show read-only (e.g. where the model's own GGUF lives)"""
    if internet or local or not isolate:
        raise ValueError("code must remain isolated and offline; use Tinyfish research tools")
    ollama_url = network.local_url(ollama_url)
    ws = os.path.realpath(workspace); os.makedirs(os.path.join(ws, ".pkgs"), exist_ok=True)
    venv = os.path.realpath(os.path.dirname(os.path.dirname(sys.executable)))
    pyroot = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
    w = "/workspace" if isolate else ws   # inside the sandbox the workspace is /workspace (what models expect)
    base_env = {"HOME": w, "RSI_WORKSPACE": w, "PYTHONPATH": f"{ROOT}:{w}/.pkgs", "PIP_TARGET": f"{w}/.pkgs",
                "PIP_BREAK_SYSTEM_PACKAGES": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PATH": f"{venv}/bin:{ROOT}/rsi/sbin:{w}/.pkgs/bin:/usr/local/bin:/usr/bin:/bin",
                "LANG": "C.UTF-8", "TERM": "dumb", **(env or {})}
    b = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-user-try",
         "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
         "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    runtime = (os.path.join(ROOT, "rsi"), os.path.join(ROOT, "rsi_lib"),
               os.path.join(ROOT, "data", "qwen35_tokenizer.json"))
    for p in sorted({venv, pyroot, *runtime}):
        if os.path.exists(p) and not p.startswith("/usr/"):
            b += ["--ro-bind", p, p]
    for d in sorted(set(model_dirs()) | {os.path.realpath(x) for x in ro if x and os.path.isdir(x)}):
        if not d.startswith("/usr/"):
            b += ["--ro-bind", d, d]
    for d in _gpu_devices():
        b += ["--dev-bind", d, d]
    b += ["--bind", ws, ws, "--bind", ws, "/workspace", "--chdir", "/workspace", "--clearenv"]   # also at its real path
    inner = list(argv)
    b += ["--unshare-net"]
    doors = netgate.ensure(gate_dir or os.path.join(ws, ".net"), ollama_url)
    b += ["--dir", "/run/rsi", "--ro-bind", doors["ollama"], "/run/rsi/ollama.sock"]
    base_env["OLLAMA_HOST"] = "127.0.0.1:11434"
    inner = [os.path.join(venv, "bin", "python"), "-m", "rsi.netgate", "bridge", "/run/rsi/ollama.sock:11434", "--"] + inner
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
