"""Webhook activation (Module 2).

Fires outbound webhooks on data changes so the Claude app's visual layers and
case logs update in real time. Uses ``httpx`` when installed, else the stdlib
``urllib``. No configured URLs => silent no-op.
"""
from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from ..config import Settings, get_settings

log = logging.getLogger("claude_legal.routing.webhooks")


class WebhookNotifier:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.webhooks_enabled

    def notify(self, event: str, payload: Any) -> List[bool]:
        if not self.enabled:
            log.debug("webhooks disabled; event '%s' not dispatched", event)
            return []
        body = {"event": event, "payload": payload}
        return [self._post(url, body) for url in self.settings.webhook_urls]

    def _post(self, url: str, body: dict) -> bool:
        try:  # pragma: no cover - network
            import httpx

            resp = httpx.post(url, json=body, timeout=10.0)
            resp.raise_for_status()
            return True
        except Exception:
            return self._post_urllib(url, body)

    @staticmethod
    def _post_urllib(url: str, body: dict) -> bool:  # pragma: no cover - network
        import urllib.request

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:
            log.error("webhook POST to %s failed: %s", url, exc)
            return False
