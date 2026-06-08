"""
Mode-vs-Regime Edge Backtest
============================
Tests four signal modes across three market regimes to find where genuine
edge exists. Runs self-contained simulations (no full strategy classes).

Usage:
    python scripts/backtest_modes.py
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import ccxt
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

SYMBOLS   = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]
MODES     = ["momentum", "mean_reversion", "vwap_reversal", "funding_arb"]
REGIMES   = ["bull", "bear", "sideways"]
TIMEFRAME = "15m"
LIMIT     = 1000
TAKER_FEE = 0.0004   # 0.04% per side

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"


# ── Data Fetching ─────────────────────────────────────────────────────────────

def build_exchange() -> ccxt.Exchange:
    ex = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30_000})
    ex.load_markets()
    return ex


def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, limit: int = LIMIT) -> np.ndarray | None:
    """Fetch raw OHLCV and return structured numpy array or None on failure."""
    try:
        raw = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        if not raw or len(raw) < 100:
            print(f"  [skip] {symbol}: insufficient bars ({len(raw) if raw else 0})")
            return None
        arr = np.array(raw, dtype=float)
        # columns: [ts, open, high, low, close, volume]
        return arr
    except Exception as e:
        print(f"  [skip] {symbol}: {e}")
        return None


def fetch_funding_history(ex: ccxt.Exchange, symbol: str) -> list[float]:
    """Return a list of funding rates aligned to 8h intervals."""
    rates = []
    try:
        history = ex.fetch_funding_rate_history(symbol, limit=200)
        rates = [float(r.get("fundingRate") or 0.0) for r in history]
    except Exception:
        pass
    return rates if rates else [0.0001] * 50


# ── Technical Indicators ──────────────────────────────────────────────────────

def ema_series(closes: np.ndarray, period: int) -> np.ndarray:
    """Full EMA series via pandas-style EWMA approximation."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(closes)
    result[0] = closes[0]
    for i in range(1, len(closes)):
        result[i] = alpha * closes[i] + (1 - alpha) * result[i - 1]
    return result


def rsi_series(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI series."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = np.mean(gains[:period])
    avg_l  = np.mean(losses[:period])
    for i in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs    = avg_g / avg_l if avg_l != 0 else 100.0
        rsi[i + 1] = 100 - 100 / (1 + rs)
    rsi[period] = 100 - 100 / (1 + (avg_g / avg_l if avg_l != 0 else 100.0))
    return rsi


def vwap_series(ohlcv: np.ndarray) -> np.ndarray:
    """Rolling VWAP over entire series."""
    highs   = ohlcv[:, 2]
    lows    = ohlcv[:, 3]
    closes  = ohlcv[:, 4]
    volumes = ohlcv[:, 5]
    typical = (highs + lows + closes) / 3.0
    cum_tpv = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cum_vol > 0, cum_tpv / cum_vol, closes)
    return vwap


def ema50_slope_series(closes: np.ndarray, window: int = 20) -> np.ndarray:
    """EMA50 rolling slope — used to classify regime bars."""
    ema50  = ema_series(closes, 50)
    slopes = np.zeros_like(ema50)
    for i in range(window, len(ema50)):
        slopes[i] = (ema50[i] - ema50[i - window]) / (ema50[i - window] + 1e-12)
    return slopes


# ── Regime Slice ──────────────────────────────────────────────────────────────

def regime_mask(closes: np.ndarray, regime: str) -> np.ndarray:
    """
    Return boolean mask of bars matching the requested regime.
    Regime is determined by 20-bar EMA50 slope:
      bull      slope >  +0.3%
      bear      slope <  -0.3%
      sideways  |slope| <= 0.3%
    """
    slopes = ema50_slope_series(closes)
    THRESH = 0.003  # 0.3%
    if regime == "bull":
        return slopes > THRESH
    elif regime == "bear":
        return slopes < -THRESH
    else:  # sideways
        return np.abs(slopes) <= THRESH


# ── Simulation Engine ─────────────────────────────────────────────────────────

def simulate_momentum(ohlcv: np.ndarray, mask: np.ndarray) -> dict:
    """EMA9 > EMA21 → LONG, else SHORT. TP=1.5%, SL=0.7%."""
    closes = ohlcv[:, 4]
    highs  = ohlcv[:, 2]
    lows   = ohlcv[:, 3]
    ema9   = ema_series(closes, 9)
    ema21  = ema_series(closes, 21)

    TP = 0.015
    SL = 0.007

    trades: list[float] = []
    i = 22  # warm-up
    while i < len(ohlcv) - 1:
        if not mask[i]:
            i += 1
            continue
        entry_price = closes[i]
        direction   = 1 if ema9[i] > ema21[i] else -1
        # Walk forward until TP/SL hit
        j = i + 1
        while j < len(ohlcv):
            hi = highs[j]
            lo = lows[j]
            if direction == 1:
                if hi >= entry_price * (1 + TP):
                    pnl = TP - 2 * TAKER_FEE
                    trades.append(pnl)
                    break
                if lo <= entry_price * (1 - SL):
                    pnl = -SL - 2 * TAKER_FEE
                    trades.append(pnl)
                    break
            else:
                if lo <= entry_price * (1 - TP):
                    pnl = TP - 2 * TAKER_FEE
                    trades.append(pnl)
                    break
                if hi >= entry_price * (1 + SL):
                    pnl = -SL - 2 * TAKER_FEE
                    trades.append(pnl)
                    break
            j += 1
        else:
            # end of data — close at last close
            last = closes[-1]
            pnl  = direction * (last - entry_price) / entry_price - 2 * TAKER_FEE
            trades.append(pnl)
        i = j + 1
    return trades


def simulate_mean_reversion(ohlcv: np.ndarray, mask: np.ndarray) -> list[float]:
    """RSI < 35 → LONG, RSI > 65 → SHORT. TP=1.0%, SL=0.5%."""
    closes = ohlcv[:, 4]
    highs  = ohlcv[:, 2]
    lows   = ohlcv[:, 3]
    rsi    = rsi_series(closes, 14)

    TP = 0.010
    SL = 0.005

    trades: list[float] = []
    i = 15  # warm-up
    while i < len(ohlcv) - 1:
        if not mask[i]:
            i += 1
            continue
        if rsi[i] >= 35 and rsi[i] <= 65:
            i += 1
            continue
        direction   = 1 if rsi[i] < 35 else -1
        entry_price = closes[i]
        j = i + 1
        while j < len(ohlcv):
            hi = highs[j]
            lo = lows[j]
            if direction == 1:
                if hi >= entry_price * (1 + TP):
                    trades.append(TP - 2 * TAKER_FEE)
                    break
                if lo <= entry_price * (1 - SL):
                    trades.append(-SL - 2 * TAKER_FEE)
                    break
            else:
                if lo <= entry_price * (1 - TP):
                    trades.append(TP - 2 * TAKER_FEE)
                    break
                if hi >= entry_price * (1 + SL):
                    trades.append(-SL - 2 * TAKER_FEE)
                    break
            j += 1
        else:
            last = closes[-1]
            pnl  = direction * (last - entry_price) / entry_price - 2 * TAKER_FEE
            trades.append(pnl)
        i = j + 1
    return trades


def simulate_vwap_reversal(ohlcv: np.ndarray, mask: np.ndarray) -> list[float]:
    """Price > VWAP*1.015 → SHORT, < VWAP*0.985 → LONG. TP=1.0%, SL=0.5%."""
    closes = ohlcv[:, 4]
    highs  = ohlcv[:, 2]
    lows   = ohlcv[:, 3]
    vwap   = vwap_series(ohlcv)

    TP = 0.010
    SL = 0.005

    trades: list[float] = []
    i = 20  # warm-up
    while i < len(ohlcv) - 1:
        if not mask[i]:
            i += 1
            continue
        price = closes[i]
        if price > vwap[i] * 1.015:
            direction = -1
        elif price < vwap[i] * 0.985:
            direction = 1
        else:
            i += 1
            continue
        entry_price = price
        j = i + 1
        while j < len(ohlcv):
            hi = highs[j]
            lo = lows[j]
            if direction == 1:
                if hi >= entry_price * (1 + TP):
                    trades.append(TP - 2 * TAKER_FEE)
                    break
                if lo <= entry_price * (1 - SL):
                    trades.append(-SL - 2 * TAKER_FEE)
                    break
            else:
                if lo <= entry_price * (1 - TP):
                    trades.append(TP - 2 * TAKER_FEE)
                    break
                if hi >= entry_price * (1 + SL):
                    trades.append(-SL - 2 * TAKER_FEE)
                    break
            j += 1
        else:
            last = closes[-1]
            pnl  = direction * (last - entry_price) / entry_price - 2 * TAKER_FEE
            trades.append(pnl)
        i = j + 1
    return trades


def simulate_funding_arb(
    ohlcv: np.ndarray,
    mask: np.ndarray,
    funding_rates: list[float],
) -> list[float]:
    """
    If |funding_rate| > 0.05% enter opposite direction.
    Hold 8 bars (2h on 15m). Collect simulated funding payment.
    """
    closes = ohlcv[:, 4]
    n      = len(closes)
    HOLD   = 8
    THRESH = 0.0005   # 0.05%

    # Tile funding rates to match bar count (funding paid every 32 bars on 15m)
    rates_cycle = list(funding_rates) or [0.0001]
    bar_rates   = np.array([rates_cycle[i % len(rates_cycle)] for i in range(n)])

    trades: list[float] = []
    i = 1
    while i < n - HOLD - 1:
        if not mask[i]:
            i += 1
            continue
        fr = bar_rates[i]
        if abs(fr) <= THRESH:
            i += 1
            continue
        # direction opposite to funding rate sign
        direction = -1 if fr > 0 else 1
        entry     = closes[i]
        exit_idx  = min(i + HOLD, n - 1)
        exit_px   = closes[exit_idx]
        price_pnl = direction * (exit_px - entry) / entry
        # funding payment received over hold period (8 bars ≈ 2h)
        funding_collected = abs(fr)
        total_pnl = price_pnl + funding_collected - 2 * TAKER_FEE
        trades.append(total_pnl)
        i = exit_idx + 1
    return trades


# ── Statistics ────────────────────────────────────────────────────────────────

def sharpe(returns: list[float]) -> float:
    """Annualised Sharpe assuming 15m bars (35040 bars/year)."""
    if len(returns) < 3:
        return 0.0
    arr = np.array(returns)
    mu  = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-12:
        return 0.0
    # 1000 bars ≈ 10.4 days; annualise factor = sqrt(35040 / len)
    annual_factor = math.sqrt(35040 / max(len(arr), 1))
    return float((mu / std) * annual_factor)


def aggregate_results(all_trades: list[float]) -> dict:
    if not all_trades:
        return {"trades": 0, "wins": 0, "total_pnl": 0.0, "sharpe": 0.0}
    wins      = sum(1 for t in all_trades if t > 0)
    total_pnl = sum(all_trades)
    sh        = sharpe(all_trades)
    return {
        "trades":    len(all_trades),
        "wins":      wins,
        "total_pnl": round(total_pnl, 5),
        "sharpe":    round(sh, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  MODE-vs-REGIME EDGE BACKTEST  (15m × 1000 bars)")
    print("=" * 72)

    # 1. Connect
    print("\n[1/3] Connecting to Binance USDT-M futures...")
    try:
        ex = build_exchange()
        print("      Connected.")
    except Exception as e:
        print(f"      FATAL: {e}")
        sys.exit(1)

    # 2. Fetch OHLCV + funding for each symbol
    print("\n[2/3] Fetching data...")
    symbol_data: dict[str, dict] = {}
    for sym in SYMBOLS:
        print(f"  {sym}...", end=" ", flush=True)
        ohlcv = fetch_ohlcv(ex, sym)
        if ohlcv is None:
            print("SKIPPED")
            continue
        funding = fetch_funding_history(ex, sym)
        symbol_data[sym] = {"ohlcv": ohlcv, "funding": funding}
        print(f"OK ({len(ohlcv)} bars, {len(funding)} funding points)")
        time.sleep(0.3)  # be polite to the API

    if not symbol_data:
        print("\nNo data fetched — aborting.")
        sys.exit(1)

    # 3. Run backtest for each (mode, regime, symbol) triple
    print("\n[3/3] Running simulations...\n")

    # Accumulate trades across all symbols per (mode, regime)
    # structure: results[(mode, regime)] = [trade_pnl, ...]
    results: dict[tuple[str, str], list[float]] = {
        (m, r): [] for m in MODES for r in REGIMES
    }

    for sym, data in symbol_data.items():
        ohlcv   = data["ohlcv"]
        funding = data["funding"]
        closes  = ohlcv[:, 4]

        for regime in REGIMES:
            mask   = regime_mask(closes, regime)
            n_bars = int(mask.sum())
            print(f"  {sym:<22} regime={regime:<10} ({n_bars:>4} bars match)")

            if n_bars < 10:
                print(f"    → too few bars for regime '{regime}', skipping")
                continue

            for mode in MODES:
                if mode == "momentum":
                    trades = simulate_momentum(ohlcv, mask)
                elif mode == "mean_reversion":
                    trades = simulate_mean_reversion(ohlcv, mask)
                elif mode == "vwap_reversal":
                    trades = simulate_vwap_reversal(ohlcv, mask)
                elif mode == "funding_arb":
                    trades = simulate_funding_arb(ohlcv, mask, funding)
                else:
                    trades = []

                results[(mode, regime)].extend(trades)

    # 4. Build report rows
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for mode in MODES:
        for regime in REGIMES:
            trades = results[(mode, regime)]
            stats  = aggregate_results(trades)
            win_rate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0.0
            verdict  = (
                "EDGE"    if stats["total_pnl"] > 0 and stats["sharpe"] > 0.5
                else "WEAK" if stats["total_pnl"] > 0
                else "NO EDGE"
            )
            rows.append({
                "mode":      mode,
                "regime":    regime,
                "trades":    stats["trades"],
                "win_rate":  round(win_rate, 1),
                "total_pnl": stats["total_pnl"],
                "sharpe":    stats["sharpe"],
                "verdict":   verdict,
            })

    # 5. Print table
    hdr = f"{'Mode':<18}| {'Regime':<10}| {'Trades':>7}| {'WinRate':>8}| {'Total PnL':>10}| {'Sharpe':>7}| Verdict"
    sep = "-" * len(hdr)
    print("\n")
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        sign      = "+" if r["total_pnl"] >= 0 else ""
        verdict_s = (
            "✅ EDGE"    if r["verdict"] == "EDGE"
            else "⚠️  WEAK"  if r["verdict"] == "WEAK"
            else "❌ NO EDGE"
        )
        print(
            f"{r['mode']:<18}| {r['regime']:<10}| {r['trades']:>7}| "
            f"{r['win_rate']:>7.1f}%| {sign}{r['total_pnl']:>9.2%}| "
            f"{r['sharpe']:>7.2f}| {verdict_s}"
        )
    print(sep)

    # 6. Recommendations
    edge_cells  = [(r["mode"], r["regime"]) for r in rows if r["verdict"] == "EDGE"]
    weak_cells  = [(r["mode"], r["regime"]) for r in rows if r["verdict"] == "WEAK"]
    no_edge_modes = [
        m for m in MODES
        if all(r["verdict"] == "NO EDGE" for r in rows if r["mode"] == m)
    ]

    print("\n" + "=" * 72)
    print("  FINAL RECOMMENDATIONS")
    print("=" * 72)

    if edge_cells:
        print("\n  ENABLE (confirmed edge):")
        for mode, regime in edge_cells:
            print(f"    ✅  {mode:<20} in regime={regime}")
    else:
        print("\n  No combinations with confirmed strong edge found.")

    if weak_cells:
        print("\n  ENABLE WITH CAUTION (marginal positive PnL, low Sharpe):")
        for mode, regime in weak_cells:
            print(f"    ⚠️   {mode:<20} in regime={regime}")

    if no_edge_modes:
        print("\n  DISABLE (negative edge across all regimes):")
        for m in no_edge_modes:
            print(f"    ❌  {m}")

    # Per-mode overall verdict
    print("\n  SUMMARY BY MODE:")
    for mode in MODES:
        mode_rows  = [r for r in rows if r["mode"] == mode]
        total_pnl  = sum(r["total_pnl"] for r in mode_rows)
        best_regime = max(mode_rows, key=lambda r: r["sharpe"])
        print(
            f"    {mode:<20} overall_pnl={'+' if total_pnl >= 0 else ''}{total_pnl:.2%}"
            f"  best_regime={best_regime['regime']} (sharpe={best_regime['sharpe']:.2f})"
        )

    # 7. Write JSON report
    report = {
        "generated_at":   "2026-06-01",
        "symbols_tested": list(symbol_data.keys()),
        "timeframe":      TIMEFRAME,
        "bars_per_symbol": LIMIT,
        "rows":           rows,
        "recommendations": {
            "enable":  [{"mode": m, "regime": r} for m, r in edge_cells],
            "caution": [{"mode": m, "regime": r} for m, r in weak_cells],
            "disable": no_edge_modes,
        },
    }
    report_path = STORAGE_DIR / "mode_edge_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {report_path}")
    print()


if __name__ == "__main__":
    main()
