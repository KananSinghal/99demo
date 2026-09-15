"""The zero-dependency HTTP server.

Runs on a stock Python 3.9+ with nothing installed at all - which is the point. An
air-gapped refinery node does not get to `pip install`, and neither does a hackathon
laptop at 3am when the venue wifi has gone.

    python -m samanvay.server

`samanvay.server_fastapi` serves exactly the same routes for teams who want uvicorn,
OpenAPI docs and async workers. Both import the same handlers, so they cannot drift.
"""

from __future__ import annotations

import json
import mimetypes
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__, config
from .api.router import dispatch
from .store import connect, init_db

_STATIC_ROOT = Path(config.WEB_DIR)


class Handler(BaseHTTPRequestHandler):
    server_version = f"Samanvay/{__version__}"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ helpers
    def _send(self, status: int, payload, content_type: str = "application/json") -> None:
        if isinstance(payload, (dict, list)):
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = payload or b""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {"records": parsed}
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    def _static(self, path: str) -> bool:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (_STATIC_ROOT / rel).resolve()
        try:
            target.relative_to(_STATIC_ROOT.resolve())
        except ValueError:
            self._send(403, {"error": "forbidden"})
            return True
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            return False
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)
        return True

    # ------------------------------------------------------------------ verbs
    def do_GET(self) -> None:           # noqa: N802
        self._handle("GET")

    def do_HEAD(self) -> None:          # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:          # noqa: N802
        self._handle("POST")

    def do_OPTIONS(self) -> None:       # noqa: N802
        self.send_response(204)
        self.send_header("Allow", "GET, POST, HEAD, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _handle(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if not path.startswith("/api/"):
                if self._static(path):
                    return
                if self._static("/index.html"):
                    return
                self._send(404, {"error": "not found"})
                return
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            body = self._body() if method == "POST" else {}
            # National material codes carry a "/" before the check digit, so a code in
            # a path arrives percent-encoded. Decode the segments, not the separators.
            decoded = "/".join(unquote(seg) for seg in path.split("/"))
            status, payload = dispatch(method, decoded, query, body)
            self._send(status, payload)
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except BrokenPipeError:
            pass
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": str(exc),
                             "traceback": traceback.format_exc().splitlines()[-5:]})

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")


def serve(host: str = None, port: int = None) -> None:
    host = host or config.HOST
    port = int(port or config.PORT)
    conn = connect()
    init_db(conn)
    conn.close()

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    banner = f"""
  Samanvay {__version__}  -  One Nation, One Material Code
  SIH26099  |  Ministry of Petroleum & Natural Gas

  console   http://{host}:{port}/
  api       http://{host}:{port}/api/routes
  database  {config.DB_PATH}
  node      {config.NODE_ROLE}{(' / ' + config.NODE_CPSE) if config.NODE_CPSE else ''}
  embedding {config.EMBEDDING_BACKEND}   rerank {config.RERANK_BACKEND}

  Similarity proposes. Standards decide. Stewards approve. The ledger remembers.
"""
    print(banner, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping", flush=True)
    finally:
        httpd.server_close()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Samanvay - zero-dependency server")
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    args = ap.parse_args()
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
