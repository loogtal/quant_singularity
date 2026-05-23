# Quant Singularity

Autonomous AI crypto trader for **Binance USDT-M Futures**. Paper trading by default.

## Features

- Scans and ranks coins automatically
- Multi-strategy ensemble (trend, momentum, mean reversion, breakout, funding)
- ML direction predictor (logistic, trains on OHLCV)
- Risk: ATR stops, trailing stop, max drawdown, daily loss cap
- Self-tuning confidence + reflection
- WebSocket prices, optional dashboard & alerts

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

Dashboard: http://127.0.0.1:8787 (if `QS_DASHBOARD=true`)

## Project layout

```
quant_singularity/
├── main.py                 # Entry — start bot
├── config/                 # settings.py, constants.py
├── core/                   # Trading engine, state, logging
├── data/                   # Binance, OHLCV, WebSocket, funding
├── factors/                # Momentum, trend, volatility
├── alpha/                  # Signal generation (+ ML blend)
├── strategy/               # Sub-strategies + router
├── meta/                   # Regime, meta brain, memory
├── risk/                   # Position sizing, kill switch, live safety
├── portfolio/              # Positions & equity
├── execution/              # Paper & live brokers
├── models/                 # ML predictor, trainer, online learning
├── research/               # Coin scanner, trade log, performance
├── self_evolve/            # Reflection, auto-tuner
├── monitoring/             # Dashboard, Discord/Telegram alerts
├── backtest/               # Quick & walk-forward backtests
├── scripts/                # CLI utilities
└── storage/                # Runtime data (gitignored)
```

## Scripts

```bash
python scripts/validate.py          # preflight + backtest + readiness
python scripts/backtest.py --symbol BTC/USDT:USDT --bars 500
python scripts/walkforward.py --symbol BTC/USDT:USDT
python scripts/train_ml.py --symbol BTC/USDT:USDT
python scripts/daily_report.py        # daily stats + BTC/ETH backtest
python scripts/preflight.py           # before live
```

See [GOAL.md](GOAL.md) for mission and live checklist.

## Configuration

Copy `.env.example` → `.env`. Key variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `QS_LIVE_MODE` | false | Paper unless true |
| `QS_LIVE_CONFIRM` | false | Must be true to send live orders |
| `QS_INITIAL_CAPITAL` | 10000 | Paper capital |
| `QS_MIN_CONFIDENCE` | 0.55 | Min signal confidence |
| `QS_MAX_DRAWDOWN` | 0.15 | Halt at 15% drawdown |
| `QS_USE_ML` | true | Blend ML into alpha |
| `QS_USE_WEBSOCKET` | true | Fast price feed |
| `QS_DASHBOARD` | true | Web UI on port 8787 |

## Live trading (careful)

1. Paper trade for weeks; run backtests first.
2. Testnet: `BINANCE_TESTNET=true`, API keys in `.env`.
3. Live: `QS_LIVE_MODE=true`, `QS_LIVE_CONFIRM=true`, small `QS_MAX_LIVE_ORDER_USDT`.

**No system guarantees daily profit.** Use at your own risk.

## License

MIT — use responsibly; not financial advice.
