"""Compact old conversation while preserving the latest complete tool exchanges."""
import copy
import json


def _size(messages):
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)


def _short(text, limit):
    text = str(text or "")
    return text if len(text) <= limit else text[:limit // 3] + " …[shortened]… " + text[-limit // 2:]


def groups(messages):
    out = []
    for message in messages:
        if message.get("role") == "tool":
            if out and out[-1][0].get("tool_calls"):
                out[-1].append(message)
            else:
                out.append([{"role": "user", "content": "[loop] Previous tool result: " + _short(message.get("content"), 1000)}])
        else:
            out.append([message])
    return out


def trim(messages, budget, history=()):
    if _size(messages) <= budget:
        return messages
    chunks = groups(copy.deepcopy(messages))
    # Tool output is data and can be shortened without changing the call arguments.
    for index, chunk in enumerate(chunks):
        for message in chunk:
            if message.get("role") == "tool":
                message["content"] = _short(message.get("content"), 2400 if index >= len(chunks) - 2 else 500)
    recent, used, cutoff = [], 0, len(chunks)
    reserve = min(1800, budget // 3)
    for index in range(len(chunks) - 1, -1, -1):
        chunk = chunks[index]
        if used + _size(chunk) > budget - reserve:
            if recent:
                break
            # One oversized call cannot fit intact. Keep its result as a summary, never orphan tool messages.
            chunk = [{"role": "user", "content": "[loop] Latest exchange (abbreviated): " + _short(
                "\n".join(json.dumps(m, ensure_ascii=False) for m in chunk), budget - reserve - 150)}]
        recent = chunk + recent; used += _size(chunk); cutoff = index
    older = [m for chunk in chunks[:cutoff] for m in chunk]
    direction = next((m.get("content", "") for m in reversed(older) if m.get("role") == "user"), "")
    events = [f"{m.get('tool_name') or m.get('role')}: {_short(m.get('content'), 220)}" for m in older[-6:]]
    past = [f"cycle {h.get('cycle')}: {_short(h.get('summary'), 150)}" for h in history[-3:]]
    note = "[loop] Older messages compacted. Current direction: " + _short(direction, 500)
    note += "\nPrevious cycles: " + " | ".join(past) + "\nEarlier actions/results:\n" + "\n".join(events)
    prefix = {"role": "user", "content": _short(note, max(100, budget - used - 100))}
    return [prefix] + recent
