"""Broker factory — live or paper, based on QS_LIVE_MODE."""

import logging

from config.settings import LIVE_MODE
from execution.paper_broker import PaperBroker


def create_broker(price_feed=None):
    if LIVE_MODE:
        try:
            from execution.live_broker import LiveBroker
            broker = LiveBroker(price_feed=price_feed)
            logging.getLogger("QS").info("[LiveBroker] Connected to Binance — LIVE MODE ACTIVE")
            return broker
        except Exception as exc:
            logging.getLogger("QS").error(
                f"[LiveBroker] FAILED to connect: {exc} — "
                f"falling back to PAPER MODE (set QS_LIVE_MODE=false to silence)"
            )
            # Fall through to paper broker so bot doesn't crash loop
    return PaperBroker(price_feed=price_feed)
