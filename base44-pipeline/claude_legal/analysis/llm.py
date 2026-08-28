"""Factual legal analysis via Claude (Module 3 core constraint).

Real Anthropic SDK client, gated behind ``CLAUDE_LEGAL_ANTHROPIC_API_KEY``. The system
prompt enforces Module 3's constraints: strictly factual analysis, no
speculation, no emotion, no informal language. When no key is configured the
analyst returns a structured "disabled" note instead of raising, so the pipeline
still completes offline.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..config import Settings, get_settings
from ..schemas import SearchHit

log = logging.getLogger("claude_legal.analysis.llm")

_SYSTEM_PROMPT = (
    "You are Claude, acting as the legal analysis matrix. Provide STRICTLY FACTUAL legal "
    "analysis grounded only in the supplied source passages. Omit speculation, "
    "emotion, and informal language. Cite the source doc_id for every assertion. "
    "If the sources do not support a conclusion, say so explicitly."
)


class LegalAnalyst:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled

    def analyze(self, question: str, hits: List[SearchHit]) -> str:
        context = "\n\n".join(
            f"[{h.doc_id}] (score={h.score}) {h.snippet or ''}" for h in hits
        )
        if not self.enabled:
            return (
                "[LLM analysis disabled — set CLAUDE_LEGAL_ANTHROPIC_API_KEY to enable]\n"
                f"Question: {question}\nTop sources: "
                + ", ".join(h.doc_id for h in hits[:5])
            )
        try:  # pragma: no cover - requires network + credentials
            import anthropic

            client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            msg = client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Question: {question}\n\nSource passages:\n{context}",
                }],
            )
            return "".join(block.text for block in msg.content if block.type == "text")
        except Exception as exc:  # pragma: no cover
            log.error("LLM analysis failed: %s", exc)
            return f"[LLM analysis error: {exc}]"
