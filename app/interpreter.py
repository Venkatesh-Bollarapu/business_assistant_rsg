"""Question interpretation: natural language -> validated ``Intent``.

Two interchangeable interpreters share one job (return an ``Intent``):

* ``LLMInterpreter``      - Groq model via LangChain structured output (Pydantic schema).
* ``RuleBasedInterpreter`` - deterministic keyword/regex parser. It is used when no API
  key is configured, when the LLM call fails, or when the LLM output fails validation,
  so the application keeps working and the tests need no network access.

Neither interpreter answers the question or writes SQL.
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from app.config import Settings
from app.schemas import MONTH_PATTERN, Filters, Intent

logger = logging.getLogger(__name__)

SUPPORTED_QUESTIONS_HELP = (
    "Try: 'What is the total revenue?', 'Which product generated the highest revenue?', "
    "'Show the monthly revenue trend', 'Which region has the highest revenue?', "
    "'Compare revenue between February and March'."
)


@dataclass
class InterpreterContext:
    """Real values from the database, so the interpreter never has to guess them."""

    known_values: dict[str, list[str]]
    today: date

    @property
    def default_year(self) -> int:
        months = self.known_values.get("month") or []
        return int(months[-1][:4]) if months else self.today.year


@dataclass
class Interpretation:
    intent: Intent
    interpreter: str  # "llm" | "rule_based"


class Interpreter(Protocol):
    def interpret(self, question: str, context: InterpreterContext) -> Interpretation: ...


# ------------------------------------------------------------------ safety pre-check
_MODIFY_RE = re.compile(
    r"\b(delete|drop|truncate|insert|update|alter|erase|wipe|overwrite|modify|remove\s+all)\b", re.IGNORECASE
)
_SQL_SYNTAX_RE = re.compile(r"(--|;|/\*|\bunion\s+select\b|\bselect\b.+\bfrom\b)", re.IGNORECASE | re.DOTALL)


def check_request_safety(question: str) -> str | None:
    """Return a refusal message for data-modifying or raw-SQL requests, else None."""
    if _MODIFY_RE.search(question):
        return "This assistant is read-only. It cannot delete, update, insert or otherwise modify data."
    if _SQL_SYNTAX_RE.search(question):
        return "Please ask in plain language. Raw SQL is not accepted; queries are generated from approved templates."
    return None


# ------------------------------------------------------------------ LLM interpreter
SYSTEM_PROMPT = """You convert a business user's question about a sales database into a structured intent.
You NEVER answer the question and NEVER write SQL. You only choose an intent and its filters.

Supported intents:
- total_revenue: total revenue (optionally filtered)
- highest_revenue_product: which product generated the highest revenue
- highest_revenue_region: which region generated the highest revenue / sales
- monthly_revenue_trend: revenue for each month
- revenue_comparison: compare revenue of exactly two months; set months to [first, second] as YYYY-MM in the order mentioned
- revenue_by_product, revenue_by_region, revenue_by_category, revenue_by_channel: revenue breakdowns
- unsupported: anything else (profit, forecasts, customers, general chat, requests to modify data). Put a short reason in reason.

Filters (set only what the user explicitly mentions, otherwise leave null):
- product, region, category, sales_channel: use the exact spelling from the lists below
- start_date / end_date: YYYY-MM-DD. For "in March" or "Q1" use the first and last day of that period.
  Do not use dates for revenue_comparison; use months instead.

Known products: {products}
Known regions: {regions}
Known categories: {categories}
Known sales channels: {channels}
Months with data: {months}
Today is {today}. If a month is given without a year, use {default_year}."""


class LLMInterpreter:
    """Groq chat model with Pydantic structured output (tool/function calling)."""

    def __init__(self, chain: Any) -> None:
        # ``chain`` is any LangChain Runnable mapping the prompt variables to an ``Intent``.
        # Injected so tests can supply a fake chain without network access.
        self._chain = chain

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMInterpreter":
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
        return cls(prompt | llm.with_structured_output(Intent, method="function_calling"))

    def interpret(self, question: str, context: InterpreterContext) -> Interpretation:
        known = context.known_values
        result = self._chain.invoke(
            {
                "question": question,
                "products": ", ".join(known.get("product", [])),
                "regions": ", ".join(known.get("region", [])),
                "categories": ", ".join(known.get("category", [])),
                "channels": ", ".join(known.get("sales_channel", [])),
                "months": ", ".join(known.get("month", [])),
                "today": context.today.isoformat(),
                "default_year": context.default_year,
            }
        )
        # with_structured_output already returns an Intent; re-validate defensively so a
        # malformed or hand-built object can never skip the Pydantic checks.
        intent = result if isinstance(result, Intent) else Intent.model_validate(result)
        return Interpretation(Intent.model_validate(intent.model_dump()), "llm")


# ------------------------------------------------------------------ rule-based interpreter
_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")\b(?!\s+(?:i|you|we|be)\b)(?:\s+(\d{4}))?",
    re.IGNORECASE,
)
_ISO_MONTH_RE = re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])\b")
_QUARTER_RE = re.compile(r"\bq([1-4])\b(?:\s*(\d{4}))?", re.IGNORECASE)

_COMPARE_RE = re.compile(
    r"\b(compare|compared|comparison|versus|vs\.?|change|changed|difference|differ|differs|grow|grew|growth|rise|rose|drop|fell|fall)\b",
    re.IGNORECASE,
)
_TREND_RE = re.compile(r"\b(trend|trends|monthly|month[- ]by[- ]month|month[- ]wise|by month|per month|each month|over time)\b", re.IGNORECASE)
_SUPERLATIVE_RE = re.compile(r"\b(highest|top|best|most|largest|biggest|maximum|max|leading|greatest)\b", re.IGNORECASE)
_BREAKDOWN_RE = re.compile(r"\b(by|per|each|every|breakdown|break\s*down|split|wise)\b", re.IGNORECASE)
_REVENUE_RE = re.compile(r"\b(revenue|revenues|sales|sale|income|earn|earned|earnings|turnover|sold|total)\b", re.IGNORECASE)
_RANGE_RE = re.compile(r"\b(between|from)\b.*\b(and|to|through|till|until)\b", re.IGNORECASE)
_UNSUPPORTED_TOPIC_RE = re.compile(
    r"\b(profit|profits|margin|margins|forecast|predict|prediction|projection|churn|customer|customers|"
    r"inventory|stock|cost|costs|expense|expenses|average|mean|median|how many|count|units)\b",
    re.IGNORECASE,
)
_DIMENSION_WORDS: dict[str, re.Pattern[str]] = {
    "product": re.compile(r"\b(product|products|item|items)\b", re.IGNORECASE),
    "region": re.compile(r"\b(region|regions|area|areas|zone|zones)\b", re.IGNORECASE),
    "category": re.compile(r"\b(category|categories)\b", re.IGNORECASE),
    "sales_channel": re.compile(r"\b(channel|channels)\b", re.IGNORECASE),
}
_DIMENSION_INTENT = {
    ("highest", "product"): "highest_revenue_product",
    ("highest", "region"): "highest_revenue_region",
    ("by", "product"): "revenue_by_product",
    ("by", "region"): "revenue_by_region",
    ("by", "category"): "revenue_by_category",
    ("by", "sales_channel"): "revenue_by_channel",
    ("highest", "category"): "revenue_by_category",  # ranking is shown; top row is the answer
    ("highest", "sales_channel"): "revenue_by_channel",
}


def _month_end(year: int, month: int) -> str:
    return date(year, month, calendar.monthrange(year, month)[1]).isoformat()


def _extract_months(question: str, default_year: int) -> list[tuple[int, int]]:
    """(year, month) pairs in the order they appear in the question."""
    found: list[tuple[int, tuple[int, int]]] = []
    for m in _ISO_MONTH_RE.finditer(question):
        found.append((m.start(), (int(m.group(1)), int(m.group(2)))))
    for m in _MONTH_RE.finditer(question):
        year = int(m.group(2)) if m.group(2) else default_year
        found.append((m.start(), (year, _MONTH_NAMES[m.group(1).lower()])))
    return [pair for _, pair in sorted(found)]


def _match_known_values(question: str, known: dict[str, list[str]]) -> dict[str, str]:
    """Find product/region/category/channel names mentioned in the question."""
    matches: dict[str, str] = {}
    for field in ("product", "region", "category", "sales_channel"):
        for value in sorted(known.get(field, []), key=len, reverse=True):
            pattern = r"\b" + re.escape(value) + r"s?\b"
            if re.search(pattern, question, re.IGNORECASE):
                matches[field] = value
                break
    return matches


_SENTENCE_STARTERS = {"i", "what", "which", "how", "show", "compare", "total", "tell", "give", "who", "when",
                      "and", "or", "not", "vs", "versus", "in", "for", "by"}


def _find_unknown_entity(question: str, known: dict[str, list[str]]) -> str | None:
    """Heuristic for the fallback parser: a Capitalised word in mid-sentence that is not a known
    product/region/category/channel/month is probably a misspelt or non-existent entity.
    (With the LLM, the same case is caught by filter validation against the database.)"""
    allowed: set[str] = set(_MONTH_NAMES) | _SENTENCE_STARTERS
    for field in ("product", "region", "category", "sales_channel"):
        for value in known.get(field, []):
            allowed.update(re.split(r"[\s\-]+", value.lower()))
            allowed.add(value.lower() + "s")
    for match in re.finditer(r"\b[A-Z][a-zA-Z]+\b", question):
        if match.start() == 0:
            continue
        word = match.group(0).lower()
        if word not in allowed and not re.fullmatch(r"q[1-4]", word):
            return match.group(0)
    return None


class RuleBasedInterpreter:
    """Deterministic fallback. Simple by design: easy to read, explain and test."""

    def interpret(self, question: str, context: InterpreterContext) -> Interpretation:
        return Interpretation(self._parse(question, context), "rule_based")

    @staticmethod
    def _unsupported(reason: str) -> Intent:
        return Intent(intent="unsupported", reason=reason)

    def _parse(self, question: str, context: InterpreterContext) -> Intent:
        text = question.strip()
        year = context.default_year

        if _UNSUPPORTED_TOPIC_RE.search(text):
            return self._unsupported(
                "That metric is not available. This assistant answers revenue questions "
                "(totals, products, regions, categories, channels and monthly trends)."
            )

        months = _extract_months(text, year)
        quarter = _QUARTER_RE.search(text)
        filters: dict[str, Any] = dict(_match_known_values(text, context.known_values))
        dimension = next((name for name, pattern in _DIMENSION_WORDS.items() if pattern.search(text)), None)

        unknown = _find_unknown_entity(text, context.known_values)
        if unknown:
            return self._unsupported(
                f"I don't recognise '{unknown}' as a product, region, category or sales channel. "
                f"Available products: {', '.join(context.known_values.get('product', []))}; "
                f"regions: {', '.join(context.known_values.get('region', []))}."
            )

        # 1) month-to-month comparison (needs two different months)
        if _COMPARE_RE.search(text) and "trend" not in text.lower():
            chosen = [f"{y:04d}-{m:02d}" for y, m in months[:2]]
            if len(chosen) == 2 and chosen[0] != chosen[1]:
                return Intent(intent="revenue_comparison", months=chosen, filters=Filters(**filters))
            if dimension is None and not filters:
                return self._unsupported(
                    "To compare revenue, name two different months, e.g. 'Compare February and March'."
                )

        # date filters for every other intent
        if quarter:
            q = int(quarter.group(1))
            q_year = int(quarter.group(2)) if quarter.group(2) else year
            filters["start_date"] = date(q_year, 3 * q - 2, 1).isoformat()
            filters["end_date"] = _month_end(q_year, 3 * q)
        elif len(months) >= 2 and _RANGE_RE.search(text):
            (y1, m1), (y2, m2) = months[0], months[-1]
            if (y1, m1) > (y2, m2):
                (y1, m1), (y2, m2) = (y2, m2), (y1, m1)
            filters["start_date"] = date(y1, m1, 1).isoformat()
            filters["end_date"] = _month_end(y2, m2)
        elif len(months) == 1:
            y, m = months[0]
            filters["start_date"], filters["end_date"] = date(y, m, 1).isoformat(), _month_end(y, m)

        # 2) monthly trend
        if _TREND_RE.search(text):
            return Intent(intent="monthly_revenue_trend", filters=Filters(**filters))

        # 3) ranking / breakdown by a business dimension ("highest ... region", "revenue by category")
        if dimension:
            kind = "highest" if _SUPERLATIVE_RE.search(text) else "by" if _BREAKDOWN_RE.search(text) else None
            intent_name = _DIMENSION_INTENT.get((kind, dimension)) if kind else None
            if intent_name:
                if kind == "by":
                    filters.pop(dimension, None)  # "by channel" means all channels, not one
                return Intent(intent=intent_name, filters=Filters(**filters))

        # 4) plain totals
        if _REVENUE_RE.search(text):
            return Intent(intent="total_revenue", filters=Filters(**filters))

        if filters:  # e.g. "And for retail?" - a follow-up fragment; the service reuses the previous intent
            return Intent(intent="unsupported", filters=Filters(**filters),
                          reason="No analysis was named in that question.")
        return self._unsupported(f"I could not map that question to a supported analysis. {SUPPORTED_QUESTIONS_HELP}")


# ------------------------------------------------------------------ orchestration
class HybridInterpreter:
    """Use the LLM when available; fall back to rules on any failure."""

    def __init__(self, llm: LLMInterpreter | None, rules: RuleBasedInterpreter | None = None) -> None:
        self._llm = llm
        self._rules = rules or RuleBasedInterpreter()

    def interpret(self, question: str, context: InterpreterContext) -> Interpretation:
        if self._llm is not None:
            try:
                return self._llm.interpret(question, context)
            except Exception as exc:  # network, rate limit, bad/invalid structured output...
                logger.warning("LLM interpretation failed (%s: %s); using rule-based fallback.",
                               type(exc).__name__, exc)
        return self._rules.interpret(question, context)


def build_interpreter(settings: Settings) -> HybridInterpreter:
    if not settings.llm_enabled:
        logger.info("GROQ_API_KEY not set: using the rule-based interpreter only.")
        return HybridInterpreter(None)
    try:
        llm = LLMInterpreter.from_settings(settings)
        logger.info("LLM interpreter enabled (model=%s).", settings.llm_model)
        return HybridInterpreter(llm)
    except Exception as exc:
        logger.error("Could not initialise the LLM (%s); using rule-based fallback.", exc)
        return HybridInterpreter(None)


__all__ = [
    "Interpretation", "InterpreterContext", "Interpreter", "LLMInterpreter", "RuleBasedInterpreter",
    "HybridInterpreter", "build_interpreter", "check_request_safety", "MONTH_PATTERN",
]
