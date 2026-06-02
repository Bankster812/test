"""
server — stdlib HTTP dashboard for the company
==============================================

Serves the single-page UI and a tiny JSON API:

    GET  /              -> dashboard HTML
    GET  /api/state     -> full company snapshot (polled by the UI)
    POST /api/decide    -> {deal_id, decision: approved|rejected}  (CEO action)

No framework, no build — `BaseHTTPRequestHandler` like `brain_web.py`.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .page import PAGE

logger = logging.getLogger("wholesale.dashboard")

# The handler reads this module-level reference (set by serve()).
_company = None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence default access logging
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            body = json.dumps(_company.snapshot()).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self) -> None:
        if self.path != "/api/decide":
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            ok = _company.decide(int(data.get("deal_id")), str(data.get("decision")))
            self._send(200 if ok else 400,
                       json.dumps({"ok": ok}).encode(), "application/json")
        except Exception as e:  # noqa: BLE001
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")


def serve(company, host: str, port: int) -> ThreadingHTTPServer:
    """Start the dashboard server (blocking caller should run serve_forever)."""
    global _company
    _company = company
    httpd = ThreadingHTTPServer((host, port), _Handler)
    return httpd
