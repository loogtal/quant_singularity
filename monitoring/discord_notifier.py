"""Discord webhook alerts."""

import json
import urllib.request


class DiscordNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, title: str, message: str, color: int = 3447003) -> bool:
        if not self.enabled:
            return False
        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                }
            ]
        }
        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            print(f"[Discord] {e}")
            return False
