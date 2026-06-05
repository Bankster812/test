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
import urllib.parse
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
        path, _, query = self.path.partition("?")
        if path == "/" or path.startswith("/index"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/state":
            body = json.dumps(_company.snapshot()).encode()
            self._send(200, body, "application/json")
        elif path in ("/api/contract", "/api/dispo"):
            params = urllib.parse.parse_qs(query)
            try:
                deal_id = int(params.get("deal_id", ["0"])[0])
            except ValueError:
                deal_id = 0
            result = (_company.contract_packet(deal_id) if path == "/api/contract"
                      else _company.dispo_submission(deal_id))
            if result is None:
                self._send(404, b'{"error":"deal not found"}', "application/json")
            else:
                self._send(200, json.dumps(result).encode(), "application/json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    _POST_ROUTES = ("/api/decide", "/api/action", "/api/control",
                    "/api/tick", "/api/contact", "/api/legal", "/api/arm",
                    "/api/foreign", "/api/agent/message",
                    "/api/leads/import", "/api/leads/add")

    def do_POST(self) -> None:
        if self.path not in self._POST_ROUTES:
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            result: dict = {"ok": False}

            if self.path == "/api/decide":
                result["ok"] = _company.decide(int(data.get("deal_id")),
                                               str(data.get("decision")))
            elif self.path == "/api/action":
                result["ok"] = _company.do_action(str(data.get("action_id")),
                                                  data.get("decision"))
            elif self.path == "/api/tick":
                result["ok"] = _company.tick_now()
            elif self.path == "/api/control":
                cmd = str(data.get("cmd"))
                result["ok"] = (_company.pause() if cmd == "pause"
                                else _company.resume() if cmd == "resume" else False)
            elif self.path == "/api/contact":
                _company.add_contact(
                    name=str(data.get("name", "")).strip() or "New Contact",
                    kind=str(data.get("kind", "cashbuyer")),
                    market=str(data.get("market", "Dallas, TX")),
                    area=str(data.get("area", "75215")),
                    email=str(data.get("email", "")))
                result["ok"] = True
            elif self.path == "/api/legal":
                result = _company.legal_review(
                    str(data.get("state", "TX")),
                    pre_foreclosure=bool(data.get("pre_foreclosure", False)))
                result["ok"] = True
            elif self.path == "/api/arm":
                result["ok"] = True
                result["armed"] = _company.arm(bool(data.get("on", False)))
            elif self.path == "/api/foreign":
                result = _company.foreign_payee_readiness()
                result["ok"] = True
            elif self.path == "/api/agent/message":
                code = str(data.get("code", "")).upper()
                message = str(data.get("message", "")).strip()
                agent = _company.agents.get(code)
                if agent and message:
                    reply = agent.receive_message(message)
                    result = {"ok": True, "reply": reply, "agent": code,
                              "agent_name": agent.name}
                else:
                    result = {"ok": False, "error": "unknown agent or empty message"}
            elif self.path == "/api/leads/import":
                csv_text = str(data.get("csv", "")).strip()
                if csv_text:
                    count = _company.import_leads_csv(csv_text)
                    result = {"ok": True, "imported": count}
                else:
                    result = {"ok": False, "error": "no csv data provided"}
            elif self.path == "/api/leads/add":
                ok = _company.add_lead(
                    address=str(data.get("address", "")).strip(),
                    city=str(data.get("city", "")).strip(),
                    state=str(data.get("state", "TX")).strip(),
                    zip_code=str(data.get("zip", "00000")).strip(),
                    est_market_value=int(data.get("market_value", 0)),
                    seller_name=str(data.get("seller_name", "Unknown")).strip(),
                    asking_price=int(data.get("asking_price", 0)),
                    motivation=str(data.get("motivation", "distress")).strip(),
                    reachable_via=str(data.get("reachable_via", "mail")).strip(),
                )
                result = {"ok": ok}

            self._send(200 if result.get("ok") else 400,
                       json.dumps(result).encode(), "application/json")
        except Exception as e:  # noqa: BLE001
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")


def serve(company, host: str, port: int) -> ThreadingHTTPServer:
    """Start the dashboard server (blocking caller should run serve_forever)."""
    global _company
    _company = company
    httpd = ThreadingHTTPServer((host, port), _Handler)
    return httpd
