"""Talking to models through ollama. Every generation streams into <session>/.rsi/live/<slot>.json while it is
produced, which is what the app's live pane shows. A dead server raises instead of returning empty answers (empty
answers would be scored as wrong and look like a broken model).

Works with any model ollama serves: each call asks ollama what the model can do (/api/show capabilities) and adapts.
  - thinking: models that can't think are never asked to ("think" is left out instead of making ollama refuse)
  - tools: models without native tool calling get the tools as text, call them as <tool_call>{...}</tool_call>, and
    get results back as user messages; the caller sees ordinary tool_calls either way
  - context: num_ctx is capped at the model's trained context length
"""
import json, os, re, time, urllib.error, urllib.request

LIVE = {"dir": None}
_CAPS = {}


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


def model_caps(cfg, model, ttl=60):
    """{"caps": set of capabilities or None if ollama didn't say, "ctx": trained context length or None}"""
    key = (cfg["ollama_url"], model); now = time.time()
    if key in _CAPS and now - _CAPS[key][0] < ttl:
        return _CAPS[key][1]
    info = {"caps": None, "ctx": None}
    try:
        req = urllib.request.Request(cfg["ollama_url"] + "/api/show", json.dumps({"model": model}).encode(), {"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())
        info["caps"] = set(d.get("capabilities") or []) or None
        ctx = [v for k, v in (d.get("model_info") or {}).items() if k.endswith(".context_length")]
        info["ctx"] = int(ctx[0]) if ctx else None
    except Exception:  # noqa: BLE001
        pass
    _CAPS[key] = (now, info)
    return info


def think_value(think):
    """True / False / "low" / "medium" / "high" / None from anything the settings or a caller may hold"""
    if isinstance(think, str):
        t = think.strip().lower()
        return {"on": True, "true": True, "yes": True, "off": False, "false": False, "no": False, "": None}.get(t, t)
    return think


# ------------------------------------------------------------------------------------------------ tools as text
TOOL_PROTOCOL = """# Tools
You can use tools. To use one, reply with a line like this and nothing after it:
{example}
(several such lines are fine). Tools, as JSON schemas:
{schemas}"""


def text_tool_messages(messages, tools):
    """the conversation for a model without native tool calling: tool schemas in the system prompt, earlier calls
    written back as <tool_call> text, tool results as user messages (consecutive user messages merged)"""
    schemas = "\n".join(json.dumps(t.get("function", t)) for t in tools)
    out = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            m = {"role": "user", "content": f"Output of {m.get('tool_name', 'the tool')}:\n{m.get('content', '')}"}
        elif role == "assistant" and m.get("tool_calls"):
            calls = "\n".join("<tool_call>" + json.dumps({"name": c.get("function", {}).get("name", ""),
                                                          "arguments": c.get("function", {}).get("arguments", {})}) + "</tool_call>"
                              for c in m["tool_calls"])
            m = {"role": "assistant", "content": ((m.get("content") or "") + "\n" + calls).strip()}
        else:
            m = {"role": role, "content": m.get("content", "")}
        if out and out[-1]["role"] == m["role"] == "user":
            out[-1] = {"role": "user", "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(m)
    f0 = (tools[0].get("function", tools[0]) if tools else {}); props = (f0.get("parameters") or {}).get("properties") or {}
    req = (f0.get("parameters") or {}).get("required") or list(props)[:1]
    ex = {k: ("..." if (props.get(k) or {}).get("type", "string") == "string" else 1) for k in req}
    proto = TOOL_PROTOCOL.format(schemas=schemas, example="<tool_call>" + json.dumps({"name": f0.get("name", "TOOL"), "arguments": ex}) + "</tool_call>")
    if out and out[0]["role"] == "system":
        out[0] = {"role": "system", "content": out[0]["content"] + "\n\n" + proto}
    else:
        out.insert(0, {"role": "system", "content": proto})
    return out


def parse_text_calls(content):
    """(tool_calls in ollama's shape, the content without the call markup). Lenient: the JSON object after <tool_call>
    is read whether or not the closing tag follows (small models often stop before it)."""
    calls, cut, text, dec = [], [], content or "", json.JSONDecoder()
    for m in re.finditer(r"<tool_call>\s*", text):
        try:
            d, end = dec.raw_decode(text, m.end())
        except ValueError:
            continue
        close = re.match(r"\s*>?\s*</tool_call>|\s*>", text[end:])
        cut.append((m.start(), end + (close.end() if close else 0)))
        if isinstance(d, dict) and d.get("name"):
            args = d.get("arguments", d.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append({"function": {"name": str(d["name"]), "arguments": args if isinstance(args, dict) else {}}})
    for a_, b_ in reversed(cut):
        text = text[:a_] + text[b_:]
    return calls, text.strip()


def salvage_calls(content, names):
    """calls written as plain JSON text ({"name": <a known tool>, "parameters"/"arguments": {...}}), which some
    models (e.g. small Llama 3.x) emit instead of real tool calls; only well-formed objects naming a known tool count"""
    calls, cut, text, dec = [], [], content or "", json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            d, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1); continue
        if isinstance(d, dict):
            f = d.get("function") if isinstance(d.get("function"), dict) else d
            name, args = f.get("name"), f.get("arguments", f.get("parameters", {}))
            if name in names and isinstance(args, dict):
                calls.append({"function": {"name": name, "arguments": args}}); cut.append((i, end))
        i = text.find("{", end)
    for a_, b_ in reversed(cut):
        text = text[:a_] + text[b_:]
    return calls, re.sub(r"```(?:json)?\s*```", "", text).strip()


# ------------------------------------------------------------------------------------------------ calls
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


def chat(cfg, model, messages, think=True, tools=None, fmt=None, num_predict=None, seed=0, live=None, timeout=7200, options=None):
    """one chat call; with `live` ({"slot": ..., "title": ...}) it streams into the live dir. cfg needs ollama_url;
    num_ctx / num_predict / keep_alive are taken from it when present."""
    info = model_caps(cfg, model); caps = info["caps"]
    think = think_value(think)
    if caps is not None and "thinking" not in caps:
        think = None                                   # this model can't think: don't ask (ollama would refuse)
    text_tools = bool(tools) and (cfg.get("text_tools") or (caps is not None and "tools" not in caps))
    num_ctx = int(cfg.get("num_ctx") or 8192)
    if info["ctx"]:
        num_ctx = min(num_ctx, info["ctx"])
    stream = bool(live and LIVE["dir"])
    body = {"model": model, "stream": stream, "messages": text_tool_messages(messages, tools) if text_tools else messages,
            "options": {"seed": seed, "num_predict": num_predict or cfg.get("num_predict") or 4096, "num_ctx": num_ctx, **(options or {})}}
    if think is not None:
        body["think"] = think
    if cfg.get("keep_alive"):
        body["keep_alive"] = cfg["keep_alive"]
    if tools and not text_tools:
        body["tools"] = tools
    if fmt:
        body["format"] = fmt
    err = None
    for attempt in range(6):
        req = urllib.request.Request(cfg["ollama_url"] + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            r = _stream(req, timeout, live, model) if stream else json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:300]
            if 400 <= e.code < 500:
                if "think" in msg.lower() and "think" in body:   # a thinking level it doesn't take, or no thinking at all
                    if isinstance(body["think"], str):
                        body["think"] = True
                    else:
                        del body["think"]
                    continue
                if "tool" in msg.lower() and "tools" in body:   # no native tool calling: tools as text
                    text_tools = True; del body["tools"]; body["messages"] = text_tool_messages(messages, tools)
                    continue
                raise ValueError(f"ollama refused the request for {model}: {msg}")   # unknown model, bad option: retrying won't help
            err = e; time.sleep(10 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(10 * (attempt + 1))
    else:
        raise ServerDown(f"ollama unreachable for {model}: {err}")
    if tools and not r["message"].get("tool_calls"):
        calls, clean = parse_text_calls(r["message"].get("content", ""))
        if not calls:
            calls, clean = salvage_calls(r["message"].get("content", ""), {t.get("function", t).get("name") for t in tools})
        if calls:
            r["message"]["tool_calls"] = calls; r["message"]["content"] = clean
    return r


def unload(cfg, model):
    try:
        req = urllib.request.Request(cfg["ollama_url"] + "/api/generate", json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:  # noqa: BLE001
        pass
