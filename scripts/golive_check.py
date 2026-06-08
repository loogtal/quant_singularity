#!/usr/bin/env python3
"""
Go-Live Checklist for Quant Singularity
========================================
Runs before switching from paper to live trading.
Checks API keys, capital, risk limits, and connectivity.

Usage:
    python scripts/golive_check.py

Exit 0 = ready to go live. Exit 1 = fix issues first.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_results: list[tuple[str, bool | None, str]] = []


def check(name: str, fn) -> bool:
    try:
        msg = fn()
        _results.append((name, True, msg or ""))
        return True
    except Exception as exc:
        _results.append((name, False, str(exc)))
        return False


def warn(name: str, fn) -> bool:
    try:
        msg = fn()
        _results.append((f"[W] {name}", True, msg or ""))
        return True
    except Exception as exc:
        _results.append((f"[W] {name}", None, str(exc)))
        return False


# ── 1. API Keys ───────────────────────────────────────────────────────────────

def _check_api_keys():
    key    = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key:
        raise ValueError("BINANCE_API_KEY not set in .env")
    if not secret:
        raise ValueError("BINANCE_API_SECRET not set in .env")
    if len(key) < 10 or len(secret) < 10:
        raise ValueError("API keys look too short — check .env")
    return f"API key={key[:4]}…{key[-4:]} (length {len(key)})"


def _check_paper_trading_off():
    live = os.environ.get("QS_LIVE_MODE", "false").lower()
    if live not in ("true", "1", "yes"):
        raise ValueError(
            "QS_LIVE_MODE=false — set to 'true' in .env to go live"
        )
    return f"QS_LIVE_MODE={live}"


def _check_binance_connectivity():
    from data.binance_client import BinanceClient
    client = BinanceClient()
    ex = client.get_exchange()
    if ex is None:
        raise ConnectionError("Cannot connect to Binance — check API keys and internet")
    markets = ex.load_markets()
    usdt_futures = [s for s in markets if s.endswith(":USDT")]
    if len(usdt_futures) < 10:
        raise ValueError(f"Too few USDT futures markets ({len(usdt_futures)}) — check account type")
    return f"Connected to Binance — {len(usdt_futures)} USDT-M futures markets"


def _check_futures_enabled():
    import ccxt
    key    = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        raise ValueError("API keys missing — check .env")
    # Create exchange directly (bypasses LIVE_MODE guard in BinanceClient)
    ex = ccxt.binanceusdm({
        "apiKey":  key,
        "secret":  secret,
        "options": {"defaultType": "future"},
    })
    try:
        balance = ex.fetch_balance({"type": "future"})
        usdt = float(balance.get("USDT", {}).get("free", 0))
        return f"Futures account accessible — free USDT balance: {usdt:.2f}"
    except Exception as e:
        err = str(e)
        if "-2015" in err or "Invalid API" in err:
            raise ValueError(
                "API key rejected by Binance (-2015).\n"
                "  FIX: Binance → API Management → Edit key → enable 'USD-M Futures'\n"
                "  Also: IP restriction must include your IP (or set Unrestricted)"
            )
        raise ValueError(f"Cannot access futures account: {e}")

# ── 2. Capital & Risk Config ──────────────────────────────────────────────────

def _check_capital_config():
    from config.dual_settings import (
        TOTAL_CAPITAL, PASSIVE_CAPITAL, ACTIVE_CAPITAL,
        ACTIVE_LEVERAGE, PASSIVE_LEVERAGE,
        ACTIVE_MAX_DAILY_LOSS, ACTIVE_DAILY_TARGET_PCT,
        KILL_SWITCH_EQUITY,
    )
    msgs = []
    if TOTAL_CAPITAL <= 0:
        raise ValueError("QS_TOTAL_CAPITAL must be > 0")
    if ACTIVE_CAPITAL <= 0 or PASSIVE_CAPITAL <= 0:
        raise ValueError("Active and passive capital must both be > 0")
    if abs(ACTIVE_CAPITAL + PASSIVE_CAPITAL - TOTAL_CAPITAL) > 1:
        raise ValueError(
            f"Active ({ACTIVE_CAPITAL}) + Passive ({PASSIVE_CAPITAL}) "
            f"!= Total ({TOTAL_CAPITAL})"
        )
    if ACTIVE_LEVERAGE > 5:
        raise ValueError(f"ACTIVE_LEVERAGE={ACTIVE_LEVERAGE} is too high for safety (max 5)")
    if PASSIVE_LEVERAGE > 3:
        raise ValueError(f"PASSIVE_LEVERAGE={PASSIVE_LEVERAGE} is too high for passive trades")
    daily_loss_pct = ACTIVE_MAX_DAILY_LOSS / ACTIVE_CAPITAL
    if daily_loss_pct > 0.10:
        raise ValueError(f"Daily loss limit {daily_loss_pct:.0%} > 10% — too loose")
    if KILL_SWITCH_EQUITY < TOTAL_CAPITAL * 0.70:
        raise ValueError(
            f"Kill switch at {KILL_SWITCH_EQUITY:.0f} = "
            f"{KILL_SWITCH_EQUITY/TOTAL_CAPITAL:.0%} of capital — too low"
        )
    daily_target_usdt = ACTIVE_CAPITAL * ACTIVE_DAILY_TARGET_PCT
    return (
        f"Total={TOTAL_CAPITAL} | Active={ACTIVE_CAPITAL} | Passive={PASSIVE_CAPITAL} | "
        f"Leverage={ACTIVE_LEVERAGE}x | Daily target={daily_target_usdt:.2f} USDT | "
        f"Kill switch={KILL_SWITCH_EQUITY:.0f}"
    )


def _check_risk_reward_ratio():
    from config.dual_settings import ACTIVE_TAKE_PROFIT, ACTIVE_STOP_LOSS
    rr = ACTIVE_TAKE_PROFIT / ACTIVE_STOP_LOSS
    if rr < 1.5:
        raise ValueError(f"Active RR={rr:.2f} < 1.5 — need TP > 1.5× SL")
    from config.dual_settings import PASSIVE_TAKE_PROFIT, PASSIVE_STOP_LOSS
    rr_p = PASSIVE_TAKE_PROFIT / PASSIVE_STOP_LOSS
    if rr_p < 1.5:
        raise ValueError(f"Passive RR={rr_p:.2f} < 1.5")
    return f"Active TP/SL={rr:.2f}× | Passive TP/SL={rr_p:.2f}×"


# ── 3. System State ────────────────────────────────────────────────────────────

def _check_lgbm_accuracy():
    import json
    from config.settings import STORAGE_DIR
    # Prefer accuracy from running bot's state (most up-to-date)
    acc = 0.0
    state_file = STORAGE_DIR / "system_state.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            acc = float(state.get("dual", {}).get("lgbm_accuracy", {}).get("global", 0))
        except Exception:
            pass
    # Fall back to loading model directly
    if acc == 0.0:
        from models.lgbm_predictor import LGBMPredictor
        p = LGBMPredictor()
        if not p.trained:
            raise ValueError(
                "LGBM model not trained yet — run main.py first or: python scripts/train_lgbm.py"
            )
        acc = p.get_accuracy("global")
    if acc < 0.50:
        raise ValueError(
            f"LGBM accuracy={acc:.3f} < 0.50 — model may be stale. "
            f"Wait for auto-retrain (every 500 cycles) or run: python scripts/train_lgbm.py"
        )
    return f"LGBM global accuracy={acc:.3f} ✓"


def _check_no_corrupt_state():
    from config.settings import STORAGE_DIR
    import json
    corrupt = list(STORAGE_DIR.glob("*.corrupt.*"))
    if corrupt:
        raise ValueError(
            f"Corrupt state files detected: {[c.name for c in corrupt]}\n"
            f"Run: python scripts/clean_workspace.py"
        )
    state = STORAGE_DIR / "system_state.json"
    if state.exists():
        try:
            json.loads(state.read_text())
        except Exception:
            raise ValueError("system_state.json is corrupt — delete it and restart")
    return "No corrupt state files"


def _check_telegram_configured():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set\n"
            "Without Telegram you won't know when to withdraw profits!"
        )
    return f"Telegram token=…{token[-8:]} chat={chat}"


def _check_loop_delay():
    from config.settings import LOOP_DELAY_SECONDS
    if LOOP_DELAY_SECONDS < 5:
        raise ValueError(f"LOOP_DELAY_SECONDS={LOOP_DELAY_SECONDS} too fast — min 5s")
    if LOOP_DELAY_SECONDS > 60:
        raise ValueError(f"LOOP_DELAY_SECONDS={LOOP_DELAY_SECONDS} too slow — max 60s for active")
    return f"Loop delay={LOOP_DELAY_SECONDS}s ✓"


# ── 4. Pre-flight simulation ──────────────────────────────────────────────────

def _check_regime_router():
    from strategy.dual_regime_router import DualRegimeRouter
    router = DualRegimeRouter()
    for regime in ("bull", "bear", "sideways", "volatile"):
        r = router.route({"regime": regime, "volatility": 0.3, "risk_on": True})
        if regime == "sideways":
            if r["active_mode"] in ("momentum", "vwap_reversal"):
                raise ValueError(
                    f"Sideways still allows {r['active_mode']} — momentum/vwap banned in ranging market"
                )
        if r["size_mult"] <= 0 or r["size_mult"] > 2.0:
            raise ValueError(f"size_mult={r['size_mult']} out of range for regime={regime}")
    return "All regime routes sane; sideways correctly restricted"


def _check_all_validations_pass():
    import subprocess
    result = subprocess.run(
        ["python", "scripts/validate_dual.py"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    if "66 pass" not in result.stdout and "66 pass" not in result.stderr:
        # Extract result line
        for line in result.stdout.splitlines():
            if "pass" in line and "fail" in line:
                raise ValueError(f"Not all 66 checks pass: {line.strip()}")
        raise ValueError("validate_dual.py did not report 66 pass")
    return "validate_dual.py: 66/66 PASS"


# ── print ─────────────────────────────────────────────────────────────────────

def _print_results():
    print()
    width = max((len(n) for n, _, _ in _results), default=30) + 4
    passes = fails = warns = 0
    for name, ok, msg in _results:
        if ok is True:
            tag = f"{GREEN}PASS{RESET}"
            passes += 1
        elif ok is False:
            tag = f"{RED}FAIL{RESET}"
            fails += 1
        else:
            tag = f"{YELLOW}WARN{RESET}"
            warns += 1
        print(f"  [{tag}] {name:<{width}} ({msg})")

    print()
    print(f"  Results: {GREEN}{passes} pass{RESET} / {RED}{fails} fail{RESET} / {YELLOW}{warns} warn{RESET}")
    print()

    if fails == 0:
        print(f"  {GREEN}{BOLD}✓ READY TO GO LIVE{RESET}")
        print()
        print("  Next steps:")
        print("  1. Set QS_PAPER_TRADING=false in .env")
        print("  2. Run: ./scripts/start.sh")
        print("  3. Monitor Telegram for first trade alerts")
        print("  4. After daily target hit, withdraw via Binance app")
    else:
        print(f"  {RED}{BOLD}✗ {fails} issue(s) must be fixed before going live{RESET}")

    print()
    return fails


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}=== Quant Singularity Go-Live Checklist ==={RESET}\n")

    print(f"  {BOLD}Section 1: API & Connectivity{RESET}")
    check("API keys configured",        _check_api_keys)
    check("Paper trading disabled",     _check_paper_trading_off)
    check("Binance connectivity",       _check_binance_connectivity)
    check("Futures account accessible", _check_futures_enabled)

    print(f"\n  {BOLD}Section 2: Capital & Risk Config{RESET}")
    check("Capital configuration",      _check_capital_config)
    check("Risk/reward ratio >= 1.5×",  _check_risk_reward_ratio)

    print(f"\n  {BOLD}Section 3: System State{RESET}")
    check("LGBM accuracy >= 0.50",      _check_lgbm_accuracy)
    check("No corrupt state files",     _check_no_corrupt_state)
    warn ("Telegram configured",        _check_telegram_configured)
    check("Loop delay in range [5,60]s",_check_loop_delay)

    print(f"\n  {BOLD}Section 4: Pre-flight Simulation{RESET}")
    check("Regime router sane",         _check_regime_router)
    check("validate_dual.py 66/66",     _check_all_validations_pass)

    fails = _print_results()
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
