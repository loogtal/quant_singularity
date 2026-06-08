"""Lightweight HTTP dashboard — stdlib only."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from config.settings import DASHBOARD_PORT, STATE_FILE, STORAGE_DIR
from monitoring.dual_dashboard import normalize_dual_state


def _html() -> str:
    return r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Quant Singularity</title>
<meta http-equiv="refresh" content="5">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:1.2rem}
h1{color:#58a6ff;font-size:1.4rem;margin-bottom:1rem;letter-spacing:.04em}
h2{color:#8b949e;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.6rem}
h3{color:#6e7681;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin:.6rem 0 .3rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.8rem;margin-bottom:.8rem}
.card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:.9rem}
.big{font-size:1.5rem;font-weight:700;color:#e6edf3}
.sub{font-size:.8rem;color:#6e7681;margin-top:.15rem}
.row{display:flex;justify-content:space-between;align-items:center;padding:.25rem 0;border-bottom:1px solid #21262d;font-size:.82rem}
.row:last-child{border-bottom:none}
.green{color:#3fb950}.red{color:#f85149}.yellow{color:#e3b341}.blue{color:#58a6ff}.orange{color:#e3793a}
.pill{display:inline-block;padding:.15rem .5rem;border-radius:12px;font-size:.7rem;font-weight:600}
.pill.bull{background:#0d3321;color:#3fb950}
.pill.bear{background:#3d1012;color:#f85149}
.pill.side{background:#1c2030;color:#8b949e}
.pill.vol{background:#2e1f00;color:#e3b341}
.pill.open{background:#0d3321;color:#3fb950}
.pill.closed{background:#3d1012;color:#f85149}
.ev-pos{color:#3fb950;font-weight:600}.ev-neg{color:#f85149;font-weight:600}
.log-item{font-size:.74rem;color:#6e7681;padding:.2rem 0;border-bottom:1px solid #21262d}
.log-item:last-child{border-bottom:none}
.fg-bar{height:6px;border-radius:3px;margin-top:.35rem;background:linear-gradient(to right,#f85149 0%,#e3b341 25%,#e3b341 50%,#3fb950 75%,#f85149 100%)}
.fg-dot{width:10px;height:10px;border-radius:50%;background:#fff;margin-top:-8px;position:relative}
</style></head><body>
<h1>&#9889; Quant Singularity</h1>
<div id="root">Loading&#8230;</div>
<script>
function fmt(n,d=2){return Number(n||0).toFixed(d)}
function pct(n){return fmt(n*100)+'%'}
function sign(n){return n>=0?'+':'';}
function evClass(n){return n>0?'ev-pos':'ev-neg'}
function regimePill(r){
  if(!r)return '';
  const cls={bull:'bull',bear:'bear',sideways:'side',volatile:'vol'}[r]||'side';
  return '<span class="pill '+cls+'">'+r.toUpperCase()+'</span>';
}
function fgColor(v){
  if(v<=22)return '#f85149';
  if(v<=40)return '#e3793a';
  if(v<=55)return '#e3b341';
  if(v<=75)return '#3fb950';
  return '#f85149';
}
function fgLabel(v){
  if(v<=22)return 'PANIC';
  if(v<=40)return 'FEAR';
  if(v<=55)return 'NEUTRAL';
  if(v<=75)return 'GREED';
  return 'EUPHORIA';
}
function btcRsColor(v){return v>65?'#f85149':v<35?'#3fb950':'#e3b341'}
function btcRsLabel(v){return v>65?'BTC DOMINANT':v<35?'ALT SEASON':'neutral'}
function trendPill(t){
  const m={bull:'bull',bear:'bear',mixed:'side'}[t]||'side';
  return '<span class="pill '+m+'">'+(t||'—').toUpperCase()+'</span>';
}
function gatePill(g){
  const open=(g||'OPEN')==='OPEN';
  return '<span class="pill '+(open?'open':'closed')+'">'+(open?'OPEN':'CLOSED')+'</span>';
}

function render(d){
  d=d||{};
  const p=d.portfolio||{}, m=d.market||{}, perf=d.performance||{}, dual=d.dual||{};
  const passive=dual.passive||{}, active=dual.active||{};
  const util=dual.capital_utilization||{}, alloc=dual.capital_allocation||{};
  const route=dual.regime_route||{}, evols=dual.evolution_log||[];
  const sstats=dual.strategy_stats||{};
  const pstats=sstats.passive||{}, astats=sstats.active||{};
  const evolved=dual.evolved_params||{};
  const intel=dual.market_intelligence||{};
  const dtarget=dual.daily_target||{};
  const aTarget=dtarget.active||{};

  // ── row 1 ─────────────────────────────────────────────────────────────────
  let html='<div class="grid">';

  // Portfolio
  const eq=p.equity||0, dd=(p.drawdown||0)*100, ddClass=dd>10?'red':dd>5?'yellow':'green';
  html+=`<div class="card">
    <h2>Portfolio</h2>
    <div class="big">${fmt(eq)} <span style="font-size:.9rem;color:#6e7681">USDT</span></div>
    <div class="sub">P&L: <span class="${eq>=35000?'green':'red'}">${sign(perf.pnl)}${fmt(perf.pnl)}</span>
      &nbsp;|&nbsp; DD: <span class="${ddClass}">${fmt(dd)}%</span></div>
    <div style="margin-top:.6rem">
      <div class="row"><span>Positions</span><span>${p.positions||0}</span></div>
      <div class="row"><span>Winrate</span><span>${pct(perf.winrate||0)}</span></div>
      <div class="row"><span>Total trades</span><span>${perf.total_trades||0}</span></div>
    </div>
  </div>`;

  // Market + Regime + Route
  html+=`<div class="card">
    <h2>Market &amp; Regime</h2>
    <div class="big">${regimePill(m.regime)}</div>
    <div class="sub">Vol: ${fmt(m.volatility||0,3)} &nbsp;|&nbsp; BTC 24h: ${sign(m.btc_change_24h)}${fmt((m.btc_change_24h||0)*100)}%</div>
    <div style="margin-top:.6rem">
      <h3>Route</h3>
      <div class="row"><span>Passive mode</span><span class="blue">${route.passive_mode||'—'}</span></div>
      <div class="row"><span>Active mode</span><span class="blue">${route.active_mode||'—'}</span></div>
      <div class="row"><span>Size mult</span><span>${fmt(route.size_mult||1,2)}&times;</span></div>
      <div class="row"><span>Reason</span><span style="font-size:.70rem;color:#6e7681">${route.reason||'—'}</span></div>
    </div>
  </div>`;

  // Passive strategy
  const pev=pstats.expected_value||0, pkf=(pstats.kelly_fraction||0)*100;
  html+=`<div class="card">
    <h2>Passive Strategy</h2>
    <div class="big green">${fmt(passive.equity||0)}</div>
    <div class="sub">P&L: ${sign(passive.realized_pnl)}${fmt(passive.realized_pnl||0)}
      &nbsp;|&nbsp; Unrealised: ${sign(passive.unrealized_pnl)}${fmt(passive.unrealized_pnl||0)}</div>
    <div style="margin-top:.6rem">
      <div class="row"><span>Positions</span><span>${passive.positions||0} / 3</span></div>
      <div class="row"><span>Utilization</span><span>${pct(util.passive||0)}</span></div>
      <div class="row"><span>Trades / Wins</span><span>${pstats.trades||0} / ${pstats.wins||0}</span></div>
      <div class="row"><span>EV/trade</span><span class="${evClass(pev)}">${sign(pev)}${fmt(pev*100,3)}%</span></div>
      <div class="row"><span>Kelly fraction</span><span>${pkf>0?fmt(pkf,1)+'%':'&lt; 10 trades'}</span></div>
    </div>
  </div>`;

  // Active strategy
  const aev=astats.expected_value||0, akf=(astats.kelly_fraction||0)*100;
  const gate=aTarget.daily_gate||'OPEN';
  html+=`<div class="card">
    <h2>Active Strategy</h2>
    <div class="big ${aev>0?'green':'yellow'}">${fmt(active.equity||0)}</div>
    <div class="sub">P&L: ${sign(active.realized_pnl)}${fmt(active.realized_pnl||0)}
      &nbsp;|&nbsp; Unrealised: ${sign(active.unrealized_pnl)}${fmt(active.unrealized_pnl||0)}</div>
    <div style="margin-top:.6rem">
      <div class="row"><span>Positions</span><span>${active.positions||0} / 2</span></div>
      <div class="row"><span>Utilization</span><span>${pct(util.active||0)}</span></div>
      <div class="row"><span>Trades / Wins</span><span>${astats.trades||0} / ${astats.wins||0}</span></div>
      <div class="row"><span>EV/trade</span><span class="${evClass(aev)}">${sign(aev)}${fmt(aev*100,3)}%</span></div>
      <div class="row"><span>Kelly fraction</span><span>${akf>0?fmt(akf,1)+'%':'&lt; 10 trades'}</span></div>
      <div class="row"><span>Daily gate</span><span>${gatePill(gate)}</span></div>
      <div class="row"><span>Today P&L</span><span class="${(aTarget.profit_today||0)>=0?'green':'red'}">${sign(aTarget.profit_today)}${fmt(aTarget.profit_today||0)}</span></div>
    </div>
  </div>`;

  html+='</div>';

  // ── row 2 ─────────────────────────────────────────────────────────────────
  html+='<div class="grid">';

  // Market Intelligence
  const fg=Number(intel.fear_greed||50), breadth=Number(intel.breadth||50);
  const fgPct=fg+'%';
  html+=`<div class="card">
    <h2>Market Intelligence</h2>
    <div class="row"><span>Dominant trend</span><span>${trendPill(intel.dominant_trend)}</span></div>
    <div class="row"><span>Bull breadth</span>
      <span class="${breadth>=60?'green':breadth<=35?'red':'yellow'}">${fmt(breadth,0)}% above EMA50</span></div>
    <div class="row"><span>Agg. funding</span>
      <span class="${(intel.aggregate_funding||0)>0.0003?'red':(intel.aggregate_funding||0)<-0.0002?'green':'yellow'}">${fmt((intel.aggregate_funding||0)*100,4)}%</span></div>
    <div style="margin-top:.6rem">
      <div style="display:flex;justify-content:space-between;font-size:.78rem">
        <span>Fear &amp; Greed</span>
        <span style="color:${fgColor(fg)};font-weight:600">${fgLabel(fg)} ${fmt(fg,0)}</span>
      </div>
      <div class="fg-bar"></div>
      <div class="fg-dot" style="margin-left:calc(${fgPct} - 5px);background:${fgColor(fg)}"></div>
    </div>
    <div class="row" style="margin-top:.4rem"><span>Panic signal</span>
      <span class="${intel.panic?'red':'green'}">${intel.panic?'YES':'no'}</span></div>
    <div class="row"><span>Euphoria signal</span>
      <span class="${intel.euphoria?'red':'green'}">${intel.euphoria?'YES':'no'}</span></div>
    <div style="border-top:1px solid #21262d;margin-top:.35rem;padding-top:.35rem">
      <div class="row"><span>BTC RS</span>
        <span style="color:${btcRsColor(Number(intel.btc_rs_score||50))};font-weight:600">${btcRsLabel(Number(intel.btc_rs_score||50))} (${fmt(intel.btc_rs_score||50,0)})</span></div>
      <div class="row"><span>Alt season</span>
        <span class="${intel.alt_season?'green':'yellow'}">${intel.alt_season?'YES':'—'}</span></div>
    </div>
  </div>`;

  // Capital Allocation
  html+=`<div class="card">
    <h2>Capital Allocation</h2>
    <div class="row"><span>Passive weight</span><span>${pct(alloc.passive_weight||0)}</span></div>
    <div class="row"><span>Active weight</span><span>${pct(alloc.active_weight||0)}</span></div>
    <div class="row"><span>Passive target</span><span>${fmt(alloc.passive_target_capital||0)}</span></div>
    <div class="row"><span>Active target</span><span>${fmt(alloc.active_target_capital||0)}</span></div>
    <div class="row"><span>Passive score</span><span>${fmt(alloc.passive_score||0,4)}</span></div>
    <div class="row"><span>Active score</span><span>${fmt(alloc.active_score||0,4)}</span></div>
  </div>`;

  // Evolved Params
  const ep=evolved.passive||{}, ea=evolved.active||{};
  html+=`<div class="card">
    <h2>Evolved Params</h2>
    <h3>Passive</h3>
    <div class="row"><span>TP / SL</span><span>${pct(ep.take_profit||.10)} / ${pct(ep.stop_loss||.05)}</span></div>
    <div class="row"><span>EMA fast/slow</span><span>${ep.ema_fast||50} / ${ep.ema_slow||200}</span></div>
    <div class="row"><span>Max hold (days)</span><span>${ep.max_hold_days||14}</span></div>
    <div class="row"><span>Sharpe</span><span class="blue">${fmt(ep.sharpe||0,3)}</span></div>
    <h3>Active</h3>
    <div class="row"><span>TP / SL</span><span>${pct(ea.take_profit||.015)} / ${pct(ea.stop_loss||.007)}</span></div>
    <div class="row"><span>EMA fast/slow</span><span>${ea.ema_fast||9} / ${ea.ema_slow||21}</span></div>
    <div class="row"><span>Sharpe</span><span class="blue">${fmt(ea.sharpe||0,3)}</span></div>
  </div>`;

  // Conflicts + Evolution log
  const conflicts=dual.conflicts||[];
  const cfHtml=conflicts.slice(0,5).map(c=>`<div class="log-item">${c.symbol||''}: ${c.reason||''}</div>`).join('')||'<div class="log-item" style="color:#3fb950">No conflicts</div>';
  const evolHtml=evols.slice(0,4).map(e=>`<div class="log-item">[${e.strategy}] sharpe ${fmt(e.baseline_sharpe,3)}&rarr;${fmt(e.new_sharpe,3)} (+${fmt(e.improvement,3)})</div>`).join('')||'<div class="log-item">No evolutions yet</div>';
  html+=`<div class="card">
    <h2>Conflict Log (last 5)</h2>
    ${cfHtml}
    <h3 style="margin-top:.7rem">Evolution Log (last 4)</h3>
    ${evolHtml}
  </div>`;

  html+='</div>';

  // ── row 3: Daily Profit + Bandit + LGBM ──────────────────────────────────
  html+='<div class="grid">';

  // Daily Profit Engine
  const adp=aTarget||{};
  const dpPhase=adp.phase||'HUNTING';
  const dpProg=Number(adp.progress_pct||0);
  const dpPnlUsd=Number(adp.pnl_today_usdt||0);
  const dpTargUsd=Number(adp.target_usdt||0);
  const dpPnlThb=Number(adp.pnl_today_thb||0);
  const dpTargThb=Number(adp.target_thb||0);
  const dpMult=Number(adp.aggression_mult||1);
  const dpPhaseColor={'HUNTING':'#58a6ff','ON_TRACK':'#3fb950','PROTECTING':'#e3b341','LOCKED':'#f85149'}[dpPhase]||'#8b949e';
  const dpBarWidth=Math.min(dpProg,100);
  html+=`<div class="card">
    <h2>&#127881; Daily Profit Engine</h2>
    <div class="big" style="color:${dpPhaseColor}">${dpPhase}</div>
    <div class="sub">Aggression: ${fmt(dpMult,2)}&times; size</div>
    <div style="margin:.5rem 0;background:#21262d;border-radius:4px;height:8px">
      <div style="background:${dpPhaseColor};width:${dpBarWidth}%;height:8px;border-radius:4px;transition:width .4s"></div>
    </div>
    <div class="row"><span>Progress</span><span style="color:${dpPhaseColor};font-weight:700">${fmt(dpProg,1)}%</span></div>
    <div class="row"><span>Today P&amp;L</span><span class="${dpPnlUsd>=0?'green':'red'}">${sign(dpPnlUsd)}${fmt(dpPnlUsd)} USDT</span></div>
    <div class="row"><span>Today P&amp;L (THB)</span><span class="${dpPnlThb>=0?'green':'red'}" style="font-weight:700">${sign(dpPnlThb)}฿${fmt(dpPnlThb,0)}</span></div>
    <div class="row"><span>Daily Target</span><span>${fmt(dpTargUsd)} USDT &nbsp;/&nbsp; ฿${fmt(dpTargThb,0)}</span></div>
  </div>`;

  // Strategy Bandit
  const bandit=dual.bandit||{};
  const topModes=dual.bandit_top_modes||{};
  const curRegime=(d.market||{}).regime||'sideways';
  const bandActive=(bandit.active||{})[curRegime]||{};
  const bandPassive=(bandit.passive||{})[curRegime]||{};
  function bandRows(obj){
    return Object.entries(obj).map(([arm,st])=>{
      const wr=st.winrate!=null?pct(st.winrate):'—';
      const cls=st.winrate>=0.5?'green':st.winrate<0.4?'red':'yellow';
      return `<div class="row"><span>${arm}</span><span class="${cls}">${wr} (${st.trials||0})</span></div>`;
    }).join('');
  }
  html+=`<div class="card">
    <h2>&#129302; Strategy Bandit <span style="font-size:.7rem;color:#6e7681">(${curRegime})</span></h2>
    <div class="row"><span>Best passive</span><span class="blue">${topModes.passive||'warming up'}</span></div>
    <div class="row"><span>Best active</span><span class="blue">${topModes.active||'warming up'}</span></div>
    <h3 style="margin-top:.5rem">Active arms (win rate / trials)</h3>
    ${bandRows(bandActive)}
    <h3 style="margin-top:.5rem">Passive arms</h3>
    ${bandRows(bandPassive)}
  </div>`;

  // LGBM Accuracy
  const lgbmAcc=dual.lgbm_accuracy||{};
  const lgbmRows=Object.entries(lgbmAcc).map(([r,a])=>{
    const cls=a>=0.55?'green':a>=0.50?'yellow':'red';
    return `<div class="row"><span>${r}</span><span class="${cls}">${pct(a)}</span></div>`;
  }).join('')||'<div class="log-item">Not trained yet</div>';
  html+=`<div class="card">
    <h2>&#129504; LightGBM Predictor</h2>
    <div class="sub">Accuracy per regime (20% holdout)</div>
    <div style="margin-top:.5rem">${lgbmRows}</div>
    <h3 style="margin-top:.6rem">Threshold to retrain: 52%</h3>
  </div>`;

  html+='</div>';
  document.getElementById('root').innerHTML=html;
}
fetch('/api/status').then(r=>r.json()).then(render).catch(e=>{
  document.getElementById('root').innerHTML='<div class="card red">API error: '+e+'</div>';
});
</script></body></html>"""


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
        dual_total = data.get("dual", {}).get("total_portfolio_value", 0)
        if pf.exists() and not dual_total:
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
        try:
            data = normalize_dual_state(data)
        except Exception:
            pass
        return json.dumps(data, default=str).encode()

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)


class DashboardServer:
    def start(self):
        # Try preferred port, then scan 8788-8798 for a free one
        for port in range(DASHBOARD_PORT, DASHBOARD_PORT + 10):
            try:
                srv = HTTPServer(("0.0.0.0", port), DashboardHandler)
                srv.allow_reuse_address = True
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                print(f"[Dashboard] http://127.0.0.1:{port}")
                return
            except OSError:
                continue
        print(f"[Dashboard] All ports {DASHBOARD_PORT}-{DASHBOARD_PORT+9} busy — disabled")
