# Edge Validation Findings

Honest, evidence-based status of whether the two engines have a real, deployable
edge. Updated as validation progresses. **Bottom line: do not deploy real capital
until `python scripts/edge_gate.py` returns GO for an engine.**

## What was tested (real strategy logic, realistic costs)

All tests replay the ACTUAL strategy code over historical bars via a point-in-time
market adapter (no lookahead), simulate the live `TradeManager` exits
(trailing / breakeven-lock / partial-TP), and charge realistic costs
(taker fee + slippage + funding; env-overridable via `QS_BT_FEE/SLIP/FUNDING`).

| Edge family | Tool | Verdict |
|---|---|---|
| Active TA signals (RSI/BB/MACD/Ichimoku/VWAP/momentum) | `backtest_real_active.py`, `walkforward_real.py` | **≈0 edge.** Gross edge at zero cost with realistic exits ≈ +0.5% (noise); negative after any cost. The earlier "+9% gross" was a churn artifact of an unrealistic fixed-TP/SL model. |
| Passive trend (EMA + pullback + ADX + Ichimoku) | `backtest_real_passive.py` | **Negative.** −20%/15mo, 38% peak DD; trend-follower whipsawed in chop. Raising ADX did not help. TradeManager raised win-rate but not total. |
| Funding carry (market-neutral) | `funding_carry_scan.py` | **Real but tiny (~1–4%/yr).** Timed harvest loses to rebalance cost. Static long-spot/short-perp on persistent-positive-funding alts ≈ +2%/yr. Safe base layer, not a money printer. |
| Donchian breakout trend-following | `backtest_donchian.py` | **Regime-dependent.** Net −2.4% over an 18mo CHOPPY window; positive skew (one +8.8% trend quarter). Plausibly positive over a full cycle (incl. bull runs), but Binance perp history (~600d) can't test that here. |
| Cross-sectional momentum (long strong / short weak, market-neutral) | `backtest_xsec_momentum.py` | **No reliable edge.** Sweep of lookback/rebal/k mostly negative; 30d momentum REVERSED (negative long-short spread = momentum crash). Best config +5% but Sharpe 0.25 / 29% DD (noise). Long-lookback spread ~1%/period but doesn't beat turnover cost. |
| Cross-exchange divergence / stat-arb (Binance vs Gate.io) | `backtest_cross_exchange.py` | **No retail-capturable edge.** Over 1000×1m bars the cross-venue spread NEVER exceeded an optimistic 0.12% round-trip cost (max divergence 0.024–0.053%). HFT keeps venues aligned to <0.03%. Capturable/day = 0.000%. |

L2 order-book microstructure and liquidation-cascade edges are NOT testable here — historical L2/tick/liquidation data is not available via REST (would need a stored dataset / live capture).

## The core lesson

Realism kills false positives. As the model got more honest the same momentum
strategy went **+15% → +3.4% → −17%**. Standard TA indicators on liquid majors
have no exploitable edge after costs — they are arbitraged away. "Profit every
day, the more the better" is **not achievable** with the retail-accessible edges
tested. A realistic system is: a small market-neutral funding base (~2%/yr) +
trend-following that wins over cycles (not daily) + strict capital preservation.

## The decision gate

`python scripts/edge_gate.py` re-runs the real walk-forwards and emits GO/NO-GO
per engine. GO requires: walk-forward total > 0, >55% of windows profitable, and
worst-window drawdown < 25%. **As of last run: NO-GO for both engines.**

## Is it "ready to trade"?

**Yes — to run; no — to risk real money on current logic.** The bot is a complete,
runnable system (paper mode is the safe default). But "ready to trade real money"
is gated honestly:

- `scripts/golive_check.py` now includes a **hard edge gate (Section 5)**: it runs
  the real-strategy walk-forward and FAILS go-live unless an engine is GO. Today it
  fails — by design, because deploying capital into a no-edge strategy loses money.
- To go live you must either (a) develop an edge that passes `edge_gate.py`, or
  (b) explicitly override with `QS_SKIP_EDGE_GATE=true` (NOT recommended — that is
  trading without validated edge).

**Go-live runbook:** `python scripts/edge_gate.py` → must show GO →
`python scripts/golive_check.py` → must pass all sections → set `QS_LIVE_MODE=true`.
Until edge_gate is GO, run in paper mode only.

This is the responsible definition of "ready to trade": the system is built, safe,
and will deploy capital the moment — and only the moment — a validated edge exists.

## Tools (reusable for any future edge hypothesis)

- `scripts/backtest_real_active.py` — real ActiveStrategy replay (+ `--min-conf`, `--cooldown`, TradeManager exits)
- `scripts/backtest_real_passive.py` — real PassiveStrategy replay (TradeManager exits, `--window-days`)
- `scripts/walkforward_real.py` — active walk-forward across regimes × modes
- `scripts/backtest_donchian.py` — Donchian breakout family (`--sweep`)
- `scripts/funding_carry_scan.py` — funding-carry opportunity scan
- `scripts/edge_gate.py` — GO/NO-GO deployment decision

## Live-code changes made from this research

- `strategy/active_strategy.py`: per-mode confidence floor `_MODE_MIN_CONF`
  (env-overridable); momentum raised 0.65→0.72 (fewer/higher-conviction trades).
- `strategy/passive_strategy.py`: `ADX_MIN_THRESHOLD` env-overridable (default
  kept 14 — raising it did not help in tests).
- Cost model in `scripts/backtest_dual.py` env-overridable (`QS_BT_FEE/SLIP/FUNDING`).
