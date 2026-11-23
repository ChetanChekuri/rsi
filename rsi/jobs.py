"""Background jobs: long work (training, data generation, big evaluations) that runs in the sandbox while the model
keeps working. Output goes to workspace/jobs/<name>.log. When a job ends, the loop tells the model at its next step.
Jobs live as long as the loop: `rsi stop` (or a crash) ends them, and on restart they are reported as lost."""
import os, re, signal, time
from . import sandbox


class Jobs:
    def __init__(self, loop):
        self.L = loop; self.procs = {}
        self.dir = os.path.join(loop.workspace, "jobs"); os.makedirs(self.dir, exist_ok=True)
        table = self.table()
        for j in table.values():   # a previous loop process ran these; they died with it
            if j["state"] == "running":
                j.update(state="lost", ended=time.strftime("%Y-%m-%d %H:%M"), note="the loop restarted while it was running", reported=False)
        self.save(table)

    def table(self):
        return self.L.s.read("state/jobs.json", {}) or {}

    def save(self, table):
        self.L.s.write("state/jobs.json", table)

    def start(self, name, command, cycle):
        c = self.L.cfg; table = self.table()
        name = re.sub(r"[^\w.-]", "-", name)[:40] or "job"
        running = [n for n, j in table.items() if j["state"] == "running"]
        cap = int(c.get("max_jobs") or 0)
        if name in running:
            return f"a job named {name} is already running"
        if cap and len(running) >= cap:
            return f"refused: {len(running)} jobs are running (the limit is {cap}): {', '.join(running)}. Wait for one or stop one."
        log = os.path.join(self.dir, f"{name}.log")
        open(log, "a").write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {command}\n")
        p = sandbox.start_in_workspace(self.L.workspace, ["bash", "-c", command], log, mem=c.get("job_mem") or c.get("sandbox_mem") or None,
                                       cpu_quota=c.get("cpu_quota"), internet=c.get("sandbox_internet", True), local=c.get("sandbox_local", False),
                                       isolate=c.get("sandbox_isolate", True), ollama_url=c["ollama_url"], gate_dir=self.L.gate_dir)
        self.procs[name] = p
        table[name] = {"command": command[:2000], "state": "running", "pid": p.pid, "cycle": cycle, "started": time.strftime("%Y-%m-%d %H:%M"),
                       "t0": time.time(), "log": os.path.relpath(log, self.L.workspace)}
        self.save(table)
        return f"started job {name}; output goes to {table[name]['log']}. You'll be told when it ends (job_log to look before that)."

    def poll(self):
        """update states; returns the jobs that ended since the last poll (not yet reported)"""
        table = self.table(); ended = []
        for name, p in list(self.procs.items()):
            rc = p.poll()
            if rc is not None and table.get(name, {}).get("state") == "running":
                table[name].update(state="done" if rc == 0 else "failed", exit_code=rc, ended=time.strftime("%Y-%m-%d %H:%M"),
                                   minutes=round((time.time() - table[name]["t0"]) / 60, 1), reported=False)
                del self.procs[name]
        for name, j in table.items():
            if j["state"] != "running" and j.get("reported") is False:
                ended.append((name, j)); j["reported"] = True
        self.save(table)
        return ended

    def tail(self, name, lines=40):
        j = self.table().get(name)
        if not j:
            return None
        try:
            return "".join(open(os.path.join(self.L.workspace, j["log"]), errors="replace").readlines()[-int(lines):])
        except OSError:
            return ""

    def stop(self, name):
        p = self.procs.get(name)
        if not p:
            return f"no running job named {name}"
        try:
            os.killpg(p.pid, signal.SIGTERM)
            for _ in range(40):
                if p.poll() is not None:
                    break
                time.sleep(0.25)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        p.wait(timeout=10)
        table = self.table()
        table[name].update(state="stopped", exit_code=p.returncode, ended=time.strftime("%Y-%m-%d %H:%M"),
                           minutes=round((time.time() - table[name]["t0"]) / 60, 1), reported=True)
        del self.procs[name]; self.save(table)
        return f"stopped job {name}"

    def stop_all(self):
        for name in list(self.procs):
            try:
                os.killpg(self.procs[name].pid, signal.SIGKILL)
            except OSError:
                pass

    def running(self):
        return [n for n, j in self.table().items() if j["state"] == "running"]

    def summary(self, name, j):
        return (f"Job {name} {j['state']}" + (f" (exit code {j.get('exit_code')})" if j.get("exit_code") is not None else "")
                + (f" after {j['minutes']} min" if j.get("minutes") is not None else "") + (f": {j['note']}" if j.get("note") else "")
                + f". Last lines of {j['log']}:\n" + (self.tail(name, 25) or "(no output)"))
