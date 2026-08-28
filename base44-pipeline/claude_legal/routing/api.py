"""REST routing surface (Module 2 — API Management).

Exposes the pipeline over HTTP so external services (including a Claude-driven
front end) can submit case payloads and receive structured JSON. FastAPI is optional; this
module is imported lazily by the CLI only when the ``api`` extra is installed.
"""
from __future__ import annotations

from typing import Optional

from ..config import get_settings
from ..pipeline import ClaudePipeline
from ..schemas import CourtFormatConfig


def create_app(pipeline: Optional[ClaudePipeline] = None):
    """Build and return a FastAPI app. Requires ``pip install '.[api]'``."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    pipe = pipeline or ClaudePipeline()
    app = FastAPI(title="Claude Legal Analysis Pipeline", version="0.1.0")

    class CaseRequest(BaseModel):
        case_id: str
        text: str
        query: Optional[str] = None
        court: Optional[CourtFormatConfig] = None

    @app.get("/health")
    def health():
        return {"status": "ready", "routes": pipe.router.routes()}

    @app.post("/case")
    def process_case(req: CaseRequest):
        result = pipe.process_case(
            case_id=req.case_id, text=req.text, query=req.query, court=req.court,
        )
        return result.model_dump(mode="json")

    return app
