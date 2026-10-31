"""FastAPI application.

Request lifecycle for POST /ask:
  1. FastAPI parses the JSON body and validates it with the ``AskRequest`` Pydantic model
     (empty / oversized questions are rejected with HTTP 422 before any of our code runs).
  2. ``BIService.ask`` interprets the question, validates the intent, runs approved SQL.
  3. The ``AskResponse`` model is serialised back to JSON; the HTTP status reflects the outcome.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings, configure_logging, get_settings
from app.schemas import AskRequest, AskResponse, DataQualityReport, SummaryResponse
from app.service import BIService

logger = logging.getLogger(__name__)

# Business outcome -> HTTP status. A valid question with no matching rows is still a 200.
STATUS_TO_HTTP = {"success": 200, "no_data": 200, "unsupported": 400, "invalid": 400}


def create_app(settings: Settings | None = None, service: BIService | None = None) -> FastAPI:
    """Application factory (lets tests inject a temporary database)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or BIService(settings)
        logger.info("Service ready (llm_enabled=%s)", settings.llm_enabled)
        yield

    app = FastAPI(
        title="AI-Powered Business Intelligence Assistant",
        description="Ask natural-language questions about sales data. The LLM interprets the "
                    "question; approved, parameterised SQL produces the answer.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})

    def svc(request: Request) -> BIService:
        return request.app.state.service

    @app.get("/health", tags=["system"])
    def health(request: Request) -> JSONResponse:
        service_ = svc(request)
        database_ok = service_.database_ok()
        body = {
            "status": "ok" if database_ok else "degraded",
            "database": "ok" if database_ok else "unavailable",
            "llm_enabled": settings.llm_enabled,
            "llm_model": settings.llm_model if settings.llm_enabled else None,
        }
        return JSONResponse(status_code=200 if database_ok else 503, content=body)

    @app.post("/ask", response_model=AskResponse, tags=["analytics"])
    def ask(payload: AskRequest, request: Request) -> JSONResponse:
        response = svc(request).ask(payload.question, payload.session_id)
        return JSONResponse(status_code=STATUS_TO_HTTP[response.status], content=response.model_dump(mode="json"))

    @app.get("/analytics/summary", response_model=SummaryResponse, tags=["analytics"])
    def analytics_summary(request: Request) -> SummaryResponse:
        return svc(request).summary()

    @app.get("/analytics/data-quality", response_model=DataQualityReport, tags=["analytics"])
    def analytics_data_quality(request: Request) -> DataQualityReport:
        return svc(request).data_quality()

    return app


app = create_app()
