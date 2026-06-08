#!/usr/bin/env python3
"""
Switch from Paper Trading → Live Trading
=========================================
Runs full safety checks, then writes QS_LIVE_MODE=true to .env.

Usage:
    python scripts/switch_to_live.py [--force]

Without --force: shows checklist and asks for confirmation.
With    --force: skips confirmation (for automated pipelines).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Load .env with override so existing empty os.environ values get replaced
try:
    from dotenv import load_dotenv as _load
    _load(dotenv_path=str(ENV_FILE), override=True)
except Exception:
    pass

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _read_env() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    result = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def _write_env_key(key: str, value: str) -> None:
    """Update or append a key in .env."""
    if not ENV_FILE.exists():
        ENV_FILE.write_text(f"{key}={value}\n")
        return
    lines = ENV_FILE.read_text().splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n")


def run_checks() -> list[str]:
    """Return list of blocking issues. Empty = all clear."""
    issues = []
    env = _read_env()

    # API keys
    if not env.get("BINANCE_API_KEY"):
        issues.append("BINANCE_API_KEY is not set in .env")
    if not env.get("BINANCE_API_SECRET"):
        issues.append("BINANCE_API_SECRET is not set in .env")

    # Alerts (recommended but not blocking)
    if not env.get("TELEGRAM_BOT_TOKEN"):
        print(f"  {YELLOW}[W] Telegram not set — you won't get withdrawal notifications{RESET}")
        print(f"      (optional: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)")

    # LGBM — auto-train on live data if not trained or accuracy too low
    try:
        from models.lgbm_predictor import LGBMPredictor
        p   = LGBMPredictor()
        acc = p.get_accuracy("global") if p.trained else 0.0

        if not p.trained or acc < 0.45:
            print(f"  {YELLOW}[!] LGBM accuracy={acc:.3f} — retraining on live data...{RESET}")
            try:
                from data.market_data import MarketData
                mkt, data = MarketData(), []
                for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]:
                    try:
                        df = mkt.get_ohlcv_df(sym, "15m", 2000)
                        if len(df) >= 300:
                            data.append((df["close"].values, df["volume"].values))
                    except Exception:
                        pass
                if data:
                    acc = p.train_multi(data, "global")
                    print(f"  {GREEN}✓{RESET} LGBM retrained: accuracy={acc:.3f}")
                    if acc < 0.45:
                        issues.append(f"LGBM accuracy still low ({acc:.3f}) — market may be unpredictable")
                else:
                    issues.append("LGBM: could not fetch live data for training")
            except Exception as te:
                issues.append(f"LGBM retrain failed: {te}")
        else:
            print(f"  {GREEN}✓{RESET} LGBM accuracy={acc:.3f}")
    except Exception as e:
        issues.append(f"LGBM check failed: {e}")

    # Capital config
    try:
        total = float(env.get("QS_TOTAL_CAPITAL", "1000"))
        active = float(env.get("QS_ACTIVE_CAPITAL", "300"))
        passive = float(env.get("QS_PASSIVE_CAPITAL", "700"))
        if abs(active + passive - total) > 1:
            issues.append(
                f"Capital mismatch: active({active}) + passive({passive}) != total({total})"
            )
        if total < 100:
            issues.append(f"QS_TOTAL_CAPITAL={total} too low — min $100 recommended")
    except Exception:
        pass

    # Binance connectivity check
    key    = env.get("BINANCE_API_KEY")    or os.environ.get("BINANCE_API_KEY", "")
    secret = env.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        issues.append("BINANCE_API_KEY / BINANCE_API_SECRET not set in .env")
    else:
        try:
            import ccxt
            ex = ccxt.binanceusdm({
                "apiKey": key, "secret": secret,
                "enableRateLimit": True, "timeout": 15000,
            })
            # Step 1: check markets load (no auth needed)
            markets = ex.load_markets()
            print(f"  {GREEN}✓{RESET} Binance connected — {len([s for s in markets if s.endswith(':USDT')])} USDT-M markets")

            # Step 2: check futures balance (needs Futures permission — warn if missing)
            try:
                balance   = ex.fetch_balance({"type": "future"})
                usdt      = float(balance.get("USDT", {}).get("free", 0) or 0)
                total_cap = float(env.get("QS_TOTAL_CAPITAL", "1000"))
                if usdt < total_cap * 0.90:
                    issues.append(
                        f"Futures wallet: {usdt:.2f} USDT but QS_TOTAL_CAPITAL={total_cap}\n"
                        f"     Deposit USDT to Binance Futures wallet, or lower QS_TOTAL_CAPITAL in .env"
                    )
                else:
                    print(f"  {GREEN}✓{RESET} Futures wallet: {usdt:.2f} USDT")
            except Exception as bal_err:
                code = str(bal_err)
                if "-2015" in code:
                    print(f"  {YELLOW}[W] Futures permission not yet enabled on API key{RESET}")
                    print(f"      Go to: binance.com → API Management → Edit → Enable Futures")
                    print(f"      (If no option: first open futures.binance.com to activate account)")
                    # Don't block — let user try; bot handles auth errors at runtime
                else:
                    issues.append(f"Binance balance check failed: {bal_err}")
        except Exception as e:
            issues.append(f"Binance connection failed: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Switch Quant Singularity to live trading")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    print(f"\n{BOLD}=== Quant Singularity: Paper → Live Switch ==={RESET}\n")

    env = _read_env()
    current_mode = env.get("QS_LIVE_MODE", "false").lower()

    if current_mode in ("true", "1", "yes"):
        print(f"{YELLOW}Already in LIVE mode.{RESET} Nothing to do.\n")
        sys.exit(0)

    print("Running safety checks...\n")
    issues = run_checks()

    if issues:
        print(f"{RED}{BOLD}✗ Cannot switch to live — fix these issues first:{RESET}\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {RED}{issue}{RESET}")
        print()
        sys.exit(1)

    print(f"{GREEN}✓ All safety checks passed{RESET}\n")

    # Show what will change
    total  = env.get("QS_TOTAL_CAPITAL", "1000")
    active = env.get("QS_ACTIVE_CAPITAL", "300")
    target_pct = float(env.get("QS_ACTIVE_DAILY_TARGET_PCT", "0.0286"))
    daily_usdt = float(active) * target_pct

    print(f"  {BOLD}What will happen:{RESET}")
    print(f"  • QS_LIVE_MODE=true (was false)")
    print(f"  • QS_LIVE_CONFIRM=true")
    print(f"  • BINANCE_TESTNET=false")
    print(f"  • Capital: {total} USDT total ({active} USDT active)")
    print(f"  • Daily target: {daily_usdt:.2f} USDT/day ({target_pct:.2%} of active)")
    print()
    print(f"  {YELLOW}{BOLD}WARNING: Real money will be used. Losses are possible.{RESET}")
    print()

    if not args.force:
        answer = input("  Type 'GO LIVE' to confirm: ").strip()
        if answer != "GO LIVE":
            print("\n  Cancelled.\n")
            sys.exit(0)

    # Apply changes
    _write_env_key("QS_LIVE_MODE",    "true")
    _write_env_key("QS_LIVE_CONFIRM", "true")
    _write_env_key("BINANCE_TESTNET", "false")

    print(f"\n  {GREEN}{BOLD}✓ LIVE MODE ENABLED{RESET}")
    print()
    print("  Next steps:")
    print("  1. Start the bot:  ./scripts/start.sh")
    print("  2. Watch Telegram for first trade alerts")
    print("  3. To revert:  set QS_LIVE_MODE=false in .env")
    print()


if __name__ == "__main__":
    main()
