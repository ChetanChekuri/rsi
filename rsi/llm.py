"""Talking to models through ollama. Every generation streams into <session>/.rsi/live/<slot>.json while it is
produced, which is what the app's live pane shows. A dead server raises instead of returning empty answers (empty
answers would be scored as wrong and look like a broken model)."""
import json, os, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from . import probes

LIVE = {"dir": None}


class ServerDown(RuntimeError):
    pass


def _live(slot, data):
    if not LIVE["dir"]:
        return
    try:
        tmp = os.path.join(LIVE["dir"], f".{slot}.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(LIVE["dir"], f"{slot}.json"))
    except OSError:
        pass


def _stream(req, timeout, live, model):
    th, ct, calls, last, final = [], [], [], 0.0, {}
    info = dict(live, model=model, started=time.time())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.strip():
                continue
            ch = json.loads(line); m = ch.get("message") or {}
            th.append(m.get("thinking") or ""); ct.append(m.get("content") or "")
            calls += m.get("tool_calls") or []
            if ch.get("done"):
                final = ch
            if time.time() - last > 0.3 or ch.get("done"):
                _live(live["slot"], dict(info, thinking="".join(th)[-60000:], content="".join(ct)[-20000:], updated=time.time(),
                                         done=bool(ch.get("done")), tool_calls=calls))
                last = time.time()
    msg = {"role": "assistant", "content": "".join(ct), "thinking": "".join(th)}
    if calls:
        msg["tool_calls"] = calls
    return {"message": msg, "done_reason": final.get("done_reason"), "eval_count": final.get("eval_count", 0)}


def chat(cfg, model, messages, think=True, tools=None, fmt=None, num_predict=None, seed=0, live=None, timeout=7200):
    """one chat call; with `live` ({"slot": ..., "title": ...}) it streams into the live dir"""
    stream = bool(live and LIVE["dir"])
    body = {"model": model, "stream": stream, "think": think, "messages": messages,
            "options": {"seed": seed, "num_predict": num_predict or cfg["num_predict"], "num_ctx": cfg["num_ctx"]}}
    if tools:
        body["tools"] = tools
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(cfg["ollama_url"] + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    err = None
    for attempt in range(4):
        try:
            if stream:
                return _stream(req, timeout, live, model)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(10 * (attempt + 1))
    raise ServerDown(f"ollama unreachable for {model}: {err}")


def unload(cfg, model):
    try:
        req = urllib.request.Request(cfg["ollama_url"] + "/api/generate", json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:  # noqa: BLE001
        pass


def solve(cfg, tag, items, path, who="solving"):
    """the model solves `items` in thinking mode; results are cached in `path` (jsonl) so nothing is solved twice"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]

    def one(it):
        r = chat(cfg, tag, [{"role": "user", "content": it["prompt"]}], think=True,
                 live={"slot": "subject-" + threading.current_thread().name.split("_")[-1], "title": f"{who}: {it['id']}",
                       "prompt": it["prompt"][:1500], "gold": it.get("gold", "")})
        m = r["message"]; content = m.get("content", "")
        if it["kind"] == "code":
            from .sandbox import grade_code
            blocks = re.findall(r"```(?:python)?\n(.*?)```", content, re.S)
            pred = blocks[-1] if blocks else content
            ok = bool(pred.strip()) and grade_code(pred, it["tests"], it["entry_point"])
        else:
            pred = probes.extract(content); ok = probes.grade(it, pred)
        return {"id": it["id"], "suite": it["suite"], "gold": it.get("gold", ""), "kind": it["kind"], "prompt": it["prompt"],
                "thinking": m.get("thinking", ""), "content": content, "pred": pred[:2000], "correct": ok,
                "done_reason": r.get("done_reason"), "tokens": r.get("eval_count", 0)}
    with open(path, "a") as f, ThreadPoolExecutor(cfg["workers"]) as ex:
        for rec in ex.map(one, todo):
            f.write(json.dumps(rec) + "\n"); f.flush(); done[rec["id"]] = rec
    unload(cfg, tag)
    return [done[it["id"]] for it in items]
