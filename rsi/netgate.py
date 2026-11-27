"""An inference-only socket into local Ollama; no general internet proxy."""
import hashlib, http.client, json, os, socket, subprocess, sys, threading, urllib.parse

OLLAMA_OK = {"/api/chat", "/api/generate", "/api/embed", "/api/embeddings", "/api/show", "/api/tags", "/api/ps", "/api/version",
             "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/models", "/"}
from . import network

_started = {}
_targets = {}
_lock = threading.Lock()


def _pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    for s, how in ((b, socket.SHUT_WR), (a, socket.SHUT_RD)):
        try:
            s.shutdown(how)
        except OSError:
            pass


def _splice(a, b):
    t = threading.Thread(target=_pipe, args=(b, a), daemon=True); t.start()
    _pipe(a, b); t.join()
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass


def _head(conn, limit=65536):
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = conn.recv(8192)
        if not d:
            return None, b""
        buf += d
        if len(buf) > limit:
            return None, b""
    head, rest = buf.split(b"\r\n\r\n", 1)
    return head.decode("latin-1").split("\r\n"), rest


def _deny(conn, code, msg):
    body = msg.encode()
    try:
        conn.sendall(f"HTTP/1.1 {code}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    finally:
        conn.close()


def _ollama(conn, target):
    target = network.local_url(target)
    conn.settimeout(30)
    lines, rest = _head(conn)
    if not lines:
        return conn.close()
    try:
        method, path, _ = lines[0].split(" ", 2)
        url = urllib.parse.urlsplit(path)
    except ValueError:
        return _deny(conn, "400 Bad Request", "bad request")
    if url.scheme or url.netloc or url.fragment or not path.startswith("/") or (
            url.path.rstrip("/") not in {p.rstrip("/") for p in OLLAMA_OK} or method not in ("GET", "POST", "HEAD")):
        return _deny(conn, "403 Forbidden", f"rsi sandbox: {method} {path} is not available here. Use the ollama_create / "
                                            "ollama_delete tools to add or remove models.")
    headers = {}
    for line in lines[1:]:
        key, colon, value = line.partition(":")
        key = key.lower()
        if not colon or key.strip() != key or key == "transfer-encoding" or (key == "content-length" and key in headers):
            return _deny(conn, "400 Bad Request", "invalid request framing")
        headers[key] = value.strip()
    size = headers.get("content-length", "0")
    if not size.isascii() or not size.isdecimal() or len(size) > 9 or int(size) > 64 * 1024**2:
        return _deny(conn, "413 Content Too Large", "unsupported request size")
    size = int(size)
    if len(rest) > size:
        return _deny(conn, "400 Bad Request", "unexpected data after the request body")
    body = bytearray(rest)
    while len(body) < size:
        data = conn.recv(min(65536, size - len(body)))
        if not data:
            return _deny(conn, "400 Bad Request", "truncated request body")
        body.extend(data)
    # Reject cloud-backed models even though the API server itself is local.
    if body:
        try:
            request_data = json.loads(body)
            # Go's JSON decoder accepts case-insensitive field names. Apply the same rule here,
            # rejecting duplicate spellings instead of letting a second model bypass validation.
            fields = {}
            if not isinstance(request_data, dict):
                raise ValueError("expected a JSON object")
            for key, value in request_data.items():
                key = key.casefold()
                if key in fields:
                    raise ValueError("duplicate field")
                fields[key] = value
            model = fields.get("model") or fields.get("name")
            if model:
                from . import llm
                llm.model_caps({"ollama_url": target}, model, require_local=True)
        except Exception:
            return _deny(conn, "403 Forbidden", "cannot verify a local model for this request")
    # Rebuild a single allowlisted request. The response may stream, but the client connection carries only this request.
    dest = urllib.parse.urlsplit(target)
    client = http.client.HTTPSConnection if dest.scheme == "https" else http.client.HTTPConnection
    up = client(dest.hostname, dest.port, timeout=600)
    try:
        forward = {key: headers[key] for key in ("content-type", "accept", "authorization") if key in headers}
        up.request(method, path, body=bytes(body) if size else None, headers=forward)
        response = up.getresponse()
    except (OSError, http.client.HTTPException) as e:
        up.close()
        return _deny(conn, "502 Bad Gateway", f"ollama unreachable: {e}")
    try:
        permitted = ("content-type", "content-length", "content-encoding")
        response_lines = [f"HTTP/1.1 {response.status} {response.reason}"]
        response_lines += [f"{key}: {value}" for key, value in response.getheaders() if key.lower() in permitted]
        conn.sendall(("\r\n".join(response_lines) + "\r\nConnection: close\r\n\r\n").encode("latin-1"))
        while data := response.read1(65536):
            conn.sendall(data)
    finally:
        up.close()
        conn.close()


def _serve(path, handler):
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(path); os.chmod(path, 0o600); srv.listen(64)

    def loop():
        while True:
            c, _ = srv.accept()
            threading.Thread(target=_safe, args=(handler, c), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()


def _safe(handler, c):
    try:
        handler(c)
    except Exception:  # noqa: BLE001
        try:
            c.close()
        except OSError:
            pass


def ensure(sockdir, ollama_url):
    """start the inference door for this process (once per directory); returns {name: socket path}"""
    with _lock:
        _targets[sockdir] = network.local_url(ollama_url)
        if sockdir not in _started:
            d = sockdir
            if len(d) > 80:   # unix socket paths are limited to ~108 bytes: use the runtime dir instead
                base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/rsi-{os.getuid()}"
                d = os.path.join(base, "rsi", hashlib.sha1(sockdir.encode()).hexdigest()[:12])
            os.makedirs(d, mode=0o700, exist_ok=True)
            p = {"ollama": os.path.join(d, "ollama.sock")}
            _serve(p["ollama"], lambda c: _ollama(c, _targets[sockdir]))
            _started[sockdir] = p
        return _started[sockdir]


# ------------------------------------------------------------------------------------------------ inside the sandbox
def bridge(pairs, cmd):
    for sock, port in pairs:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", int(port))); srv.listen(64)

        def loop(srv=srv, sock=sock):
            while True:
                c, _ = srv.accept()

                def one(c=c):
                    u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    try:
                        u.connect(sock)
                    except OSError:
                        return c.close()
                    _splice(c, u)
                threading.Thread(target=one, daemon=True).start()
        threading.Thread(target=loop, daemon=True).start()
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__" and sys.argv[1:2] == ["bridge"]:
    a = sys.argv[2:]; i = a.index("--")
    sys.exit(bridge([x.rsplit(":", 1) for x in a[:i]], a[i + 1:]))
