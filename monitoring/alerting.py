"""Unified trade alerts — Discord + Telegram."""

from config.settings import (
    ALERTS_ENABLED,
    DISCORD_WEBHOOK_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)
from monitoring.discord_notifier import DiscordNotifier
from monitoring.telegram_notifier import TelegramNotifier


class Alerting:
    def __init__(self):
        self.enabled = ALERTS_ENABLED
        self.discord = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    def _broadcast(self, title: str, body: str, color: int = 3447003) -> None:
        if not self.enabled:
            return
        text = f"{title}\n{body}"
        self.discord.send(title, body, color=color)
        self.telegram.send(text)

    def trade_open(self, symbol: str, side: str, price: float, size: float, equity: float):
        self._broadcast(
            "OPEN",
            f"{side} {symbol}\nPrice: {price}\nSize: {size}\nEquity: {equity:.2f}",
            color=3066993,
        )

    def trade_close(self, symbol: str, reason: str, pnl: float, equity: float):
        color = 3066993 if pnl >= 0 else 15158332
        self._broadcast(
            reason,
            f"{symbol}\nPnL: {pnl:+.2f} USDT\nEquity: {equity:.2f}",
            color=color,
        )

    def halt(self, reason: str):
        self._broadcast("HALT", reason, color=15158332)

    def daily_summary(self, equity: float, daily_pct: float, trades: int):
        self._broadcast(
            "Daily summary",
            f"Equity: {equity:.2f}\nDaily: {daily_pct:+.2%}\nTrades: {trades}",
        )
