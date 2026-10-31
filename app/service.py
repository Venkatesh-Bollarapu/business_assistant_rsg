"""The question-answering pipeline that ties everything together.

    question -> safety pre-check -> interpreter (LLM or rules) -> Intent
             -> follow-up context merge -> filter validation
             -> approved SQL template -> read-only query -> answer built from rows

The LLM is only ever involved in the interpreter step.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date

from sqlalchemy import Engine

from app.analytics import get_dimension_values, get_summary, run_intent
from app.config import Settings
from app.db import database_has_data, get_readonly_engine, get_write_engine
from app.ingestion import ingest_csv
from app.interpreter import (
    SUPPORTED_QUESTIONS_HELP,
    Interpreter,
    InterpreterContext,
    build_interpreter,
    check_request_safety,
)
from app.quality import get_data_quality_report
from app.schemas import AskResponse, DataQualityReport, Filters, Intent, SummaryResponse

logger = logging.getLogger(__name__)

FILTER_FIELDS = ("product", "region", "category", "sales_channel", "start_date", "end_date")
VALUE_FILTERS = ("product", "region", "category", "sales_channel")
_FOLLOW_UP_RE = re.compile(
    r"\b(its|it|that|this|those|these|their|them|same|there)\b|^\s*(and|also|what about|how about|then)\b",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(r"\b(its|it|that|this|their|them|same|there)\b", re.IGNORECASE)
MAX_SESSIONS = 200
# Which dimension an intent groups by (so a follow-up never filters on the thing being ranked).
INTENT_DIMENSION = {
    "highest_revenue_product": "product", "revenue_by_product": "product",
    "highest_revenue_region": "region", "revenue_by_region": "region",
    "revenue_by_category": "category", "revenue_by_channel": "sales_channel",
}


@dataclass
class SessionState:
    intent: Intent
    top_entity: tuple[str, str] | None


def ensure_database(settings: Settings) -> None:
    """Create and load the database from the CSV on first run."""
    if database_has_data(settings.database_path):
        return
    logger.info("Database empty or missing: ingesting %s", settings.csv_path)
    engine = get_write_engine(settings.database_path)
    try:
        ingest_csv(settings.csv_path, engine)
    finally:
        engine.dispose()


class FilterValidationError(ValueError):
    pass


def canonicalise_filters(filters: Filters, known: dict[str, list[str]]) -> Filters:
    """Match filter values to real database values (case-insensitive) or raise."""
    updated = filters.model_copy()
    for name in VALUE_FILTERS:
        value = getattr(filters, name)
        if value is None:
            continue
        lookup = {v.lower(): v for v in known.get(name, [])}
        match = lookup.get(value.lower()) or lookup.get(value.lower().rstrip("s"))
        if match is None:
            options = ", ".join(known.get(name, []))
            raise FilterValidationError(
                f"Unknown {name.replace('_', ' ')} '{value}'. Available: {options}."
            )
        setattr(updated, name, match)
    return updated


def apply_context(intent: Intent, previous: SessionState | None, question: str) -> tuple[Intent, list[str]]:
    """Fill gaps in a follow-up question from the previous turn.

    Rules that keep this safe:
      * only runs when the question looks like a follow-up ("its", "that", "what about"...);
      * only fills filters the user did NOT state explicitly - explicit filters always win;
      * dates are carried only if the new question gives no dates and no comparison months.
    """
    if previous is None or intent.intent == "unsupported" or not _FOLLOW_UP_RE.search(question):
        return intent, []

    merged = intent.filters.model_copy()
    applied: list[str] = []
    has_new_dates = bool(merged.start_date or merged.end_date) or intent.intent == "revenue_comparison"

    for name in FILTER_FIELDS:
        is_date = name in ("start_date", "end_date")
        if getattr(merged, name) is None and getattr(previous.intent.filters, name) is not None:
            if is_date and has_new_dates:
                continue
            setattr(merged, name, getattr(previous.intent.filters, name))
            applied.append(f"{name}={getattr(merged, name)}")

    # "What about ITS monthly trend?" -> the entity the previous answer named (e.g. region South).
    if previous.top_entity and _PRONOUN_RE.search(question):
        dimension, value = previous.top_entity
        grouped_by_same_dimension = INTENT_DIMENSION.get(intent.intent) == dimension
        if getattr(merged, dimension) is None and not grouped_by_same_dimension:
            setattr(merged, dimension, value)
            applied.append(f"{dimension}={value}")

    if not applied:
        return intent, []
    return intent.model_copy(update={"filters": merged}), applied


class BIService:
    def __init__(self, settings: Settings, interpreter: Interpreter | None = None) -> None:
        ensure_database(settings)
        self.settings = settings
        self.engine: Engine = get_readonly_engine(settings.database_path)
        self.interpreter: Interpreter = interpreter or build_interpreter(settings)
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- session store
    def _get_session(self, session_id: str | None) -> SessionState | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def _save_session(self, session_id: str | None, state: SessionState) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions[session_id] = state
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

    # ----------------------------------------------------------------- main pipeline
    def ask(self, question: str, session_id: str | None = None) -> AskResponse:
        question = question.strip()
        base = {"question": question, "session_id": session_id}

        refusal = check_request_safety(question)
        if refusal:
            logger.warning("Refused request: %r", question)
            return AskResponse(**base, status="unsupported", intent="unsupported", answer=refusal)

        known = get_dimension_values(self.engine)
        context = InterpreterContext(known_values=known, today=date.today())
        interpretation = self.interpreter.interpret(question, context)
        intent, interpreter_name = interpretation.intent, interpretation.interpreter

        previous = self._get_session(session_id)
        if (intent.intent == "unsupported" and previous is not None
                and intent.filters.active() and _FOLLOW_UP_RE.search(question)):
            # "And for retail?" - keep the previous analysis, swap in the newly stated filters.
            intent = previous.intent.model_copy(update={"filters": intent.filters})

        if intent.intent == "unsupported":
            reason = intent.reason or "That question is outside what this assistant can answer."
            if SUPPORTED_QUESTIONS_HELP not in reason:
                reason = f"{reason} {SUPPORTED_QUESTIONS_HELP}"
            return AskResponse(**base, status="unsupported", intent="unsupported",
                               answer=reason, interpreter=interpreter_name)

        intent, applied = apply_context(intent, previous, question)

        try:
            intent = intent.model_copy(update={"filters": canonicalise_filters(intent.filters, known)})
        except FilterValidationError as exc:
            return AskResponse(**base, status="invalid", intent=intent.intent, answer=str(exc),
                               interpreter=interpreter_name, context_applied=applied)

        outcome = run_intent(self.engine, intent)
        logger.info("ask intent=%s interpreter=%s status=%s filters=%s context=%s",
                    intent.intent, interpreter_name, outcome.status, intent.filters.active(), applied)

        if outcome.status == "success":
            self._save_session(session_id, SessionState(intent, outcome.top_entity))

        return AskResponse(
            **base,
            status=outcome.status,  # type: ignore[arg-type]
            intent=intent.intent,
            filters=intent.filters.active(),
            answer=outcome.answer,
            result={"headline": outcome.headline, "rows": outcome.rows,
                    **({"months": intent.months} if intent.months else {})},
            sql_executed=outcome.sql,
            sql_parameters=outcome.params,
            interpreter=interpreter_name,  # type: ignore[arg-type]
            context_applied=applied,
        )

    # ----------------------------------------------------------------- read-only reports
    def summary(self) -> SummaryResponse:
        return get_summary(self.engine)

    def data_quality(self) -> DataQualityReport:
        return get_data_quality_report(self.engine)

    def database_ok(self) -> bool:
        try:
            get_dimension_values(self.engine)
            return True
        except Exception:
            logger.exception("Database health check failed")
            return False


__all__ = ["BIService", "ensure_database", "apply_context", "canonicalise_filters"]
