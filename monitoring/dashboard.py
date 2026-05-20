"""Lightweight HTTP dashboard — stdlib only."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from config.settings import DASHBOARD_PORT, STATE_FILE, STORAGE_DIR


def _html() -> str:
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Quant Singularity</title>
<meta http-equiv="refresh" content="5">
<style>body{font-family:system-ui;background:#0d1117;color:#e6edf3;margin:2rem}
.card{background:#161b22;padding:1rem;border-radius:8px;margin:1rem 0}
.green{color:#3fb950}.red{color:#f85149}</style></head><body>
<h1>Quant Singularity</h1><motion id="root">Loading...</motion>
<script>
fetch('/api/status').then(r=>r.json()).then(d=>{
  const p=d.portfolio||{},m=d.market||{},perf=d.performance||{};
  const eq=(p.equity||0).toFixed(2);
  const wr=((perf.winrate||0)*100).toFixed(1);
  document.getElementById('root').innerHTML=
    '<motion class="card"><h2>Portfolio</h2><p>Equity: <b>'+eq+'</b> USDT</p>'+
    '<p>Positions: '+(p.positions||0)+' | DD: '+((p.drawdown||0)*100).toFixed(2)+'%</p></motion>'+
    '<motion class="card"><h2>Market</h2><p>'+(m.regime||'-')+' | vol '+(m.volatility||'-')+'</p></motion>'+
    '<motion class="card"><h2>Performance</h2><p>Winrate: '+wr+'% | Trades: '+(perf.total_trades||0)+'</p></motion>';
}).catch(e=>{document.getElementById('root').textContent=String(e);});
</script></body></html>""".replace("motion", "div")


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html", _html().encode())
        elif path == "/api/status":
            self._send(200, "application/json", self._state())
        else:
            self._send(404, "text/plain", b"404")

    def _state(self) -> bytes:
        data = {}
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
            except json.JSONDecodeError:
                pass
        pf = STORAGE_DIR / "performance.json"
        if pf.exists():
            try:
                pm = json.loads(pf.read_text())
                t = max(pm.get("total_trades", 0), 1)
                data.setdefault("performance", {})
                data["performance"].update({
                    "winrate": pm.get("wins", 0) / t,
                    "total_trades": pm.get("total_trades", 0),
                    "pnl": pm.get("total_pnl", 0),
                })
            except json.JSONDecodeError:
                pass
        return json.dumps(data, default=str).encode()

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)


class DashboardServer:
    def start(self):
        srv = HTTPServer(("0.0.0.0", DASHBOARD_PORT), DashboardHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[Dashboard] http://127.0.0.1:{DASHBOARD_PORT}")
