"""Unified trade alerts — Discord + Telegram."""

from datetime import datetime, timezone

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
        self.enabled  = ALERTS_ENABLED
        self.discord  = DiscordNotifier(DISCORD_WEBHOOK_URL)
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self._last_locked_day: int = -1   # track which day we sent "LOCKED" alert

    def _broadcast(self, title: str, body: str, color: int = 3447003) -> None:
        if not self.enabled:
            return
        text = f"{title}\n{body}"
        self.discord.send(title, body, color=color)
        self.telegram.send(text)

    # ── trade events ──────────────────────────────────────────────────────────

    def trade_open(self, symbol: str, side: str, price: float, size: float, equity: float):
        self._broadcast(
            "📈 OPEN" if side == "LONG" else "📉 OPEN",
            f"{side} {symbol}\nPrice: {price:,.4f}\nSize: {size}\nEquity: {equity:,.2f} USDT",
            color=3066993,
        )

    def trade_close(self, symbol: str, reason: str, pnl: float, equity: float):
        emoji = "✅" if pnl >= 0 else "❌"
        color = 3066993 if pnl >= 0 else 15158332
        self._broadcast(
            f"{emoji} CLOSE [{reason}]",
            f"{symbol}\nPnL: {pnl:+.2f} USDT\nEquity: {equity:,.2f} USDT",
            color=color,
        )

    def halt(self, reason: str):
        self._broadcast("🛑 HALT", reason, color=15158332)

    # ── daily target reached → withdrawal notification ────────────────────────

    def daily_target_reached(
        self,
        pnl_usdt: float,
        withdrawable_usdt: float,
        thb_per_usd: float = 35.0,
    ) -> None:
        """
        Called once per day when DailyProfitEngine transitions to LOCKED.
        Tells the user exactly how much they can withdraw today.
        """
        today_ordinal = datetime.now(timezone.utc).toordinal()
        if today_ordinal == self._last_locked_day:
            return   # already sent today
        self._last_locked_day = today_ordinal

        withdrawable_thb = round(withdrawable_usdt * thb_per_usd, 0)
        pnl_thb          = round(pnl_usdt * thb_per_usd, 0)

        self._broadcast(
            "💰 วันนี้ถึงเป้าแล้ว! ถอนได้เลย",
            (
                f"กำไรวันนี้: +{pnl_usdt:.2f} USDT (+{pnl_thb:,.0f} บาท)\n"
                f"ถอนได้: {withdrawable_usdt:.2f} USDT ({withdrawable_thb:,.0f} บาท)\n"
                f"(เก็บไว้ compound 20% อัตโนมัติ)\n"
                f"เวลา: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            ),
            color=15844367,  # gold
        )

    # ── daily summary ─────────────────────────────────────────────────────────

    def daily_summary(
        self,
        equity: float,
        daily_pct: float,
        trades: int,
        *,
        passive_equity: float = 0.0,
        active_equity: float = 0.0,
        pnl_today_usdt: float = 0.0,
        withdrawable_usdt: float = 0.0,
        thb_per_usd: float = 35.0,
        lgbm_accuracy: float = 0.0,
        regime: str = "",
        btc_cycle_phase: str = "",
        active_winrate: float = 0.0,
        passive_winrate: float = 0.0,
    ) -> None:
        pnl_thb          = round(pnl_today_usdt * thb_per_usd, 0)
        withdrawable_thb = round(withdrawable_usdt * thb_per_usd, 0)
        daily_usdt       = round(equity * daily_pct, 2)

        lines = [
            f"วันที่: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            f"",
            f"💼 Portfolio รวม: {equity:,.2f} USDT",
            f"  Passive: {passive_equity:,.2f} USDT",
            f"  Active:  {active_equity:,.2f} USDT",
            f"",
            f"📊 วันนี้",
            f"  กำไร/ขาดทุน: {pnl_today_usdt:+.2f} USDT ({pnl_thb:+,.0f} บาท)",
            f"  Return: {daily_pct:+.2%} ({daily_usdt:+.2f} USDT)",
            f"  Trades: {trades}",
        ]

        if active_winrate > 0:
            lines.append(f"  Winrate: active={active_winrate:.0%} passive={passive_winrate:.0%}")

        if withdrawable_usdt > 0:
            lines += [
                f"",
                f"💰 ถอนได้วันนี้: {withdrawable_usdt:.2f} USDT ({withdrawable_thb:,.0f} บาท)",
            ]
        else:
            lines.append(f"  ยังถอนไม่ได้วันนี้ (ยังไม่ถึงเป้า)")

        if regime or btc_cycle_phase:
            lines += [f"", f"🌐 ตลาด"]
            if regime:
                lines.append(f"  Regime: {regime.upper()}")
            if btc_cycle_phase:
                lines.append(f"  BTC Cycle: {btc_cycle_phase}")
            if lgbm_accuracy > 0:
                flag = " ⚠ ต่ำ" if lgbm_accuracy < 0.50 else " ✓"
                lines.append(f"  ML Accuracy: {lgbm_accuracy:.1%}{flag}")

        self._broadcast(
            "📅 รายงานประจำวัน",
            "\n".join(lines),
            color=3447003,
        )

    # ── regime change ─────────────────────────────────────────────────────────

    def regime_change(self, old: str, new: str) -> None:
        emoji = {"bull": "🐂", "bear": "🐻", "sideways": "↔️", "volatile": "⚡"}.get(new, "🔄")
        self._broadcast(
            f"{emoji} Regime เปลี่ยน",
            f"{old.upper()} → {new.upper()}",
            color=10181046,
        )
