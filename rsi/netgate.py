"""Internet for the sandbox without exposing this machine.

The sandbox has its own network namespace, so on its own it can reach nothing. Two doors are opened into it, each a
unix socket that the loop serves from outside:

  proxy   an HTTP(S) proxy (CONNECT and plain HTTP) to the public internet. It resolves names itself and refuses
          loopback, private, link-local and other non-public addresses, so code can't reach this machine's local
          services or the home network. Inside the sandbox HTTP_PROXY / HTTPS_PROXY point at it (pip, curl,
          requests, urllib, httpx all use them).
  ollama  the inference server, for running and inspecting models (chat, generate, embed, show, tags). Creating,
          copying, pulling and deleting models are refused: those go through the ollama_* tools, which track which
          models the loop made.

Inside the sandbox, `python -m rsi.netgate bridge SOCK:PORT ... -- CMD` listens on 127.0.0.1:PORT for each door,
forwards to the socket, and runs CMD.
"""
import hashlib, ipaddress, os, socket, subprocess, sys, threading, urllib.parse

OLLAMA_OK = {"/api/chat", "/api/generate", "/api/embed", "/api/embeddings", "/api/show", "/api/tags", "/api/ps", "/api/version",
             "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/models", "/"}
_started = {}
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


def _close_headers(lines, first):
    keep = [l for l in lines[1:] if l.split(":", 1)[0].strip().lower() not in ("connection", "proxy-connection", "keep-alive")]
    return ("\r\n".join([first] + keep + ["Connection: close"]) + "\r\n\r\n").encode("latin-1")


def _deny(conn, code, msg):
    body = msg.encode()
    try:
        conn.sendall(f"HTTP/1.1 {code}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    finally:
        conn.close()


def public_address(host, port):
    """resolve and return a public address to connect to, or None (never loopback / private / link-local / ...)"""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return None
    for fam, _, _, _, addr in infos:
        ip = ipaddress.ip_address(addr[0])
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if ip.is_global and not ip.is_multicast:
            return fam, addr
    return None


def _proxy(conn):
    lines, rest = _head(conn)
    if not lines:
        return conn.close()
    try:
        method, target, ver = lines[0].split(" ", 2)
    except ValueError:
        return _deny(conn, "400 Bad Request", "bad request")
    if method == "CONNECT":
        host, _, port = target.rpartition(":"); port = int(port or 443); first = None
    else:
        u = urllib.parse.urlsplit(target)
        if u.scheme != "http" or not u.hostname:
            return _deny(conn, "400 Bad Request", "use an absolute http:// URL (https goes through CONNECT)")
        host, port = u.hostname, u.port or 80
        first = f"{method} {urllib.parse.urlunsplit(('', '', u.path or '/', u.query, ''))} {ver}"
    host = host.strip("[]")
    dest = public_address(host, port)
    if not dest:
        return _deny(conn, "403 Forbidden", f"rsi sandbox: {host} is not a public internet address")
    try:
        up = socket.socket(dest[0], socket.SOCK_STREAM); up.settimeout(30); up.connect(dest[1]); up.settimeout(None)
    except OSError as e:
        return _deny(conn, "502 Bad Gateway", f"could not connect to {host}: {e}")
    if first is None:
        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        if rest:
            up.sendall(rest)
    else:
        up.sendall(_close_headers(lines, first) + rest)
    _splice(conn, up)


def _ollama(conn, target):
    lines, rest = _head(conn)
    if not lines:
        return conn.close()
    try:
        method, path, ver = lines[0].split(" ", 2)
    except ValueError:
        return _deny(conn, "400 Bad Request", "bad request")
    if urllib.parse.urlsplit(path).path.rstrip("/") not in {p.rstrip("/") for p in OLLAMA_OK} or method not in ("GET", "POST", "HEAD"):
        return _deny(conn, "403 Forbidden", f"rsi sandbox: {method} {path} is not available here. Use the ollama_create / "
                                            "ollama_delete tools to add or remove models.")
    host, _, port = target.rpartition(":")
    try:
        up = socket.create_connection((host, int(port)), timeout=30); up.settimeout(None)
    except OSError as e:
        return _deny(conn, "502 Bad Gateway", f"ollama unreachable: {e}")
    up.sendall(_close_headers(lines, lines[0]) + rest)
    _splice(conn, up)


def _serve(path, handler):
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(path); os.chmod(path, 0o600); srv.listen(64)

    def loop():
        while True:
            c, _ = srv.accept()
            threading.Thread(target=lambda: _safe(handler, c), daemon=True).start()
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
    """start the two doors for this process (once per directory); returns {name: socket path}"""
    with _lock:
        if sockdir not in _started:
            d = sockdir
            if len(d) > 80:   # unix socket paths are limited to ~108 bytes: use the runtime dir instead
                base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/rsi-{os.getuid()}"
                d = os.path.join(base, "rsi", hashlib.sha1(sockdir.encode()).hexdigest()[:12])
            os.makedirs(d, mode=0o700, exist_ok=True)
            p = {"proxy": os.path.join(d, "proxy.sock"), "ollama": os.path.join(d, "ollama.sock")}
            _serve(p["proxy"], _proxy)
            target = ollama_url.split("://")[-1].rstrip("/")
            _serve(p["ollama"], lambda c: _ollama(c, target))
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
