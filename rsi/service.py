"""A persistent user service owns the loop, independently of the terminal UI."""
import hashlib
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

ROOT = str(Path(__file__).resolve().parent.parent)


def name(folder):
    return "rsi-" + hashlib.sha256(os.path.realpath(folder).encode()).hexdigest()[:16] + ".service"


def _quote(value, command=False):
    value = str(value)
    if any(c in value for c in "\n\r\0"):
        raise ValueError("service paths and environment values must not contain newlines or NUL")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def _path(value):
    # Single-path directives consume the full value; quotes would become part of the path.
    value = str(value)
    if any(c in value for c in "\n\r\0"):
        raise ValueError("service paths must not contain newlines or NUL")
    return value.replace("%", "%%")


def unit(folder, limits):
    state = str(Path(folder).absolute() / ".rsi")
    props = []
    if limits.get("memory_cap"):
        props += ["MemoryMax=" + str(limits["memory_cap"]), "MemorySwapMax=0"]
    if limits.get("cpu_quota"):
        props.append("CPUQuota=" + str(limits["cpu_quota"]))
    return "\n".join([
        "[Unit]", "Description=Autonomous RSI loop", "StartLimitIntervalSec=0",
        "ConditionPathExists=!" + _path(os.path.join(state, "STOPPED")),
        "[Service]", "Type=simple", "WorkingDirectory=" + _path(ROOT),
        "Environment=" + _quote("PYTHONPATH=" + ROOT),
        "EnvironmentFile=-" + _path(os.path.join(ROOT, ".env")),
        "EnvironmentFile=-" + _path(os.path.join(state, "service.env")),
        "ExecStart=" + " ".join(_quote(x, command=True) for x in (sys.executable, "-u", "-m", "rsi.loop", str(Path(folder).absolute()))),
        "Restart=always", "RestartSec=10", "TimeoutStopSec=20", "KillMode=control-group",
        "StandardOutput=append:" + _path(os.path.join(state, "loop.log")), "StandardError=inherit",
        *props, "[Install]", "WantedBy=default.target", "",
    ])


def _run(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"service setup failed: {' '.join(args[:3])}: {(result.stderr or result.stdout).strip()[:500]}")
    return result.stdout.strip()


def start(folder, limits):
    if not shutil.which("systemctl") or not shutil.which("loginctl"):
        raise RuntimeError("unattended RSI requires systemd user services and loginctl")
    # Lingering starts the user manager on boot and keeps it alive after logout.
    uid = str(os.getuid())
    if _run(["loginctl", "show-user", uid, "-p", "Linger", "--value"]) != "yes":
        _run(["loginctl", "--no-ask-password", "enable-linger", uid])
    state = Path(folder).absolute() / ".rsi"
    path = state / name(folder)
    path.write_text(unit(folder, limits))
    # Carry only explicit runtime settings, never the entire login environment.
    envpath = state / "service.env"
    permitted = ("TINYFISH_API_KEY", "OLLAMA_MODELS", "RSI_LLAMA_QUANTIZE", "CUDA_VISIBLE_DEVICES")
    values = {}
    for line in envpath.read_text().splitlines() if envpath.exists() else []:
        fields = shlex.split(line)
        if len(fields) == 1 and "=" in fields[0]:
            key, value = fields[0].split("=", 1)
            if key in permitted:
                values[key] = value
    values.update({key: os.environ[key] for key in permitted if key in os.environ})
    fd = os.open(envpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as target:
        target.write("\n".join(key + "=" + _quote(value).replace("%%", "%") for key, value in values.items()) + "\n")
    os.chmod(envpath, 0o600)
    _run(["systemctl", "--user", "link", str(path)])
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", name(folder)])
    _run(["systemctl", "--user", "start", name(folder)])


def stop(folder):
    if (Path(folder) / ".rsi" / name(folder)).exists():
        _run(["systemctl", "--user", "stop", name(folder)])
