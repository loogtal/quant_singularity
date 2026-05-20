"""Telegram bot alerts."""

import json
import urllib.parse
import urllib.request


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, message: str) -> bool:
        if not self.enabled:
            return False
        url = (
            f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            f"?{urllib.parse.urlencode({'chat_id': self.chat_id, 'text': message})}"
        )
        try:
            urllib.request.urlopen(url, timeout=10)
            return True
        except Exception as e:
            print(f"[Telegram] {e}")
            return False
