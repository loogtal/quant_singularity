"""
Quant Singularity — Autonomous AI Crypto Trader (Binance Futures)
Paper by default. Set QS_LIVE_MODE=true + API keys for live.
"""

from dotenv import load_dotenv

load_dotenv()

from core.engine import TradingEngine


if __name__ == "__main__":
    TradingEngine().run()
