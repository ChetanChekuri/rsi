"""Network policy: local Ollama inference and Tinyfish research only."""
import ipaddress
import urllib.parse
import urllib.request


def local_url(value):
    value = value if "://" in value else "http://" + value
    url = urllib.parse.urlsplit(value)
    host = url.hostname
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "localhost"
    if not local or url.scheme not in ("http", "https") or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        raise ValueError("Ollama must use a loopback HTTP(S) address; Tinyfish is the only external API")
    # Do not let an environment proxy forward local inference externally.
    return value.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("API redirects are disabled by the network policy")


def open_local(url, **kw):
    address = url.full_url if isinstance(url, urllib.request.Request) else url
    parts = urllib.parse.urlsplit(address)
    local_url(urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", "")))
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(url, **kw)


def local_model(info):
    """Ollama can proxy cloud models even at a loopback address."""
    if info.get("remote_host") or info.get("remote_model"):
        raise ValueError("cloud-backed Ollama models are disabled; inference must stay local")
