"""Centralized payload routing (Module 2).

Passes structured JSON/text payloads between services by named route. New
handlers register against a route name, so services plug in WITHOUT editing the
orchestrator — preserving existing routing logic (Block A requirement #3).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("claude_legal.routing")

Handler = Callable[[Any], Any]


class PayloadRouter:
    def __init__(self, notifier: Optional["object"] = None) -> None:
        self._routes: Dict[str, List[Handler]] = {}
        self._notifier = notifier  # optional WebhookNotifier

    def register(self, route: str, handler: Handler) -> None:
        self._routes.setdefault(route, []).append(handler)
        log.info("registered handler for route '%s'", route)

    def routes(self) -> List[str]:
        return sorted(self._routes)

    def dispatch(self, route: str, payload: Any) -> List[Any]:
        """Run every handler registered for ``route`` and fire change webhooks."""
        handlers = self._routes.get(route, [])
        if not handlers:
            log.warning("no handler registered for route '%s'", route)
        results = [h(payload) for h in handlers]
        if self._notifier is not None:
            try:
                self._notifier.notify(event=route, payload=_jsonable(payload))
            except Exception as exc:  # pragma: no cover
                log.error("webhook notify failed for '%s': %s", route, exc)
        return results


def _jsonable(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    return payload
