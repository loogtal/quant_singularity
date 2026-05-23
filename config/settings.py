"""
Central configuration — load from environment or defaults.
Paper mode is default; live trading requires explicit opt-in.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# --- Mode ---
LIVE_MODE = _env_bool("QS_LIVE_MODE", False)
PAPER_MODE = not LIVE_MODE

# --- Binance API (only needed for live) ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET = _env_bool("BINANCE_TESTNET", True)

# --- Capital ---
INITIAL_CAPITAL = _env_float("QS_INITIAL_CAPITAL", 10000.0)

# --- Loop ---
LOOP_DELAY_SECONDS = _env_int("QS_LOOP_DELAY", 15)
MAX_POSITIONS = _env_int("QS_MAX_POSITIONS", 3)
COOLDOWN_SECONDS = _env_int("QS_COOLDOWN", 120)

# --- Risk ---
MIN_CONFIDENCE = _env_float("QS_MIN_CONFIDENCE", 0.55)
MAX_DRAWDOWN = _env_float("QS_MAX_DRAWDOWN", 0.15)
VOLATILITY_CUTOFF = _env_float("QS_VOL_CUTOFF", 0.85)
MAX_DAILY_LOSS_PCT = _env_float("QS_MAX_DAILY_LOSS", 0.03)
RISK_PER_TRADE = _env_float("QS_RISK_PER_TRADE", 0.008)
MAX_POSITION_FRACTION = _env_float("QS_MAX_POS_FRACTION", 0.08)

# --- Scanner ---
SCAN_TOP_N = _env_int("QS_SCAN_TOP_N", 25)
MIN_24H_VOLUME_USDT = _env_float("QS_MIN_VOLUME", 50_000_000)

# --- Targets (informational — not guaranteed) ---
DAILY_TARGET_PCT = _env_float("QS_DAILY_TARGET", 0.005)

# --- Maintenance ---
RESET_STATS_ON_START = _env_bool("QS_RESET_STATS", False)
PERSIST_PORTFOLIO = _env_bool("QS_PERSIST_PORTFOLIO", True)

# --- Live safety ---
LIVE_CONFIRM = _env_bool("QS_LIVE_CONFIRM", False)
MAX_LIVE_ORDER_USDT = _env_float("QS_MAX_LIVE_ORDER_USDT", 500.0)

# --- Alerts ---
ALERTS_ENABLED = _env_bool("QS_ALERTS", False)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Auto-tune ---
AUTO_TUNE_ENABLED = _env_bool("QS_AUTO_TUNE", True)

# --- Paths ---
STORAGE_DIR = PROJECT_ROOT / "storage"
TRADES_DB = STORAGE_DIR / "trades.db"
STATE_FILE = STORAGE_DIR / "system_state.json"
MODEL_DIR = STORAGE_DIR / "models"

# --- ML ---
USE_ML = _env_bool("QS_USE_ML", True)
AUTO_TRAIN_ML = _env_bool("QS_AUTO_TRAIN_ML", True)
ML_RETRAIN_EVERY_CYCLES = _env_int("QS_ML_RETRAIN_CYCLES", 500)

# --- Daily ops ---
AUTO_DAILY_REPORT = _env_bool("QS_DAILY_REPORT", True)
PREFLIGHT_ON_START = _env_bool("QS_PREFLIGHT", True)

# --- WebSocket ---
USE_WEBSOCKET = _env_bool("QS_USE_WEBSOCKET", True)
WS_SYMBOLS_LIMIT = _env_int("QS_WS_SYMBOLS", 80)

# --- Dashboard ---
DASHBOARD_ENABLED = _env_bool("QS_DASHBOARD", True)
DASHBOARD_PORT = _env_int("QS_DASHBOARD_PORT", 8787)

# --- Funding ---
USE_FUNDING_ARB = _env_bool("QS_FUNDING_ARB", True)
