"""Startup checks — Binance, API, live safety, ML model."""

from config.settings import (
    AUTO_TRAIN_ML,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_TESTNET,
    LIVE_CONFIRM,
    LIVE_MODE,
    MODEL_DIR,
    USE_ML,
)
from data.binance_client import BinanceClient


class Preflight:
    def run(self) -> dict:
        checks = []
        ok = True

        client = BinanceClient()
        if client.connected:
            checks.append({"name": "binance", "ok": True, "msg": "Connected"})
        else:
            checks.append({"name": "binance", "ok": False, "msg": "No connection"})
            ok = False

        if LIVE_MODE:
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                checks.append({"name": "api_keys", "ok": False, "msg": "Missing API keys"})
                ok = False
            else:
                checks.append({"name": "api_keys", "ok": True, "msg": "API keys set"})
            if not LIVE_CONFIRM:
                checks.append({
                    "name": "live_confirm",
                    "ok": False,
                    "msg": "QS_LIVE_CONFIRM=false — orders blocked",
                })
            else:
                checks.append({"name": "live_confirm", "ok": True, "msg": "Live confirm ON"})
            net = "testnet" if BINANCE_TESTNET else "mainnet"
            checks.append({"name": "network", "ok": True, "msg": f"Futures {net}"})
        else:
            checks.append({"name": "mode", "ok": True, "msg": "PAPER mode"})

        if USE_ML:
            model_file = MODEL_DIR / "logistic_weights.json"
            if model_file.exists():
                checks.append({"name": "ml_model", "ok": True, "msg": "ML weights loaded"})
            elif AUTO_TRAIN_ML:
                checks.append({"name": "ml_model", "ok": True, "msg": "Will auto-train on start"})
            else:
                checks.append({
                    "name": "ml_model",
                    "ok": True,
                    "msg": "No ML file — using heuristic (run scripts/train_ml.py)",
                })

        return {"ok": ok, "checks": checks}

    def print_report(self, result: dict) -> None:
        print("\n--- PREFLIGHT ---")
        for c in result["checks"]:
            mark = "OK" if c["ok"] else "!!"
            print(f"  [{mark}] {c['name']}: {c['msg']}")
        if not result["ok"]:
            print("  Fix issues above before live trading.")
        print("-----------------\n")
