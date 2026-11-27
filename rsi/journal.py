"""Locked JSONL journals; only an interrupted final record is recoverable."""
import fcntl
import json
import os
import tempfile


def _decode(data):
    rows, offset = [], 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip():
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise
                return rows, offset
            rows.append(row)
        offset += len(line)
    return rows, offset


def read(path):
    try:
        with open(path, "rb") as source:
            fcntl.flock(source, fcntl.LOCK_SH)
            return _decode(source.read())[0]
    except FileNotFoundError:
        return []


def append(path, row):
    record = json.dumps(row, default=str).encode() + b"\n"
    with open(path, "a+b") as target:
        fcntl.flock(target, fcntl.LOCK_EX)
        target.seek(0)
        data = target.read()
        _, end = _decode(data)
        if end != len(data):
            # Preserve the fragment for diagnosis before repairing the journal.
            with tempfile.NamedTemporaryFile(prefix=os.path.basename(path) + ".interrupted-",
                                             dir=os.path.dirname(path), delete=False) as backup:
                backup.write(data[end:]); backup.flush(); os.fsync(backup.fileno())
            target.truncate(end)
        target.seek(0, os.SEEK_END)
        if end and not data[:end].endswith(b"\n"):
            target.write(b"\n")
        target.write(record); target.flush(); os.fsync(target.fileno())
