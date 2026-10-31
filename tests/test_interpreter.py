"""Question interpretation: rule-based parser, LLM path (with a fake chain) and fallback."""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.interpreter import HybridInterpreter, InterpreterContext, LLMInterpreter, RuleBasedInterpreter
from app.schemas import Filters, Intent
from app.service import apply_context, SessionState
from app.analytics import get_dimension_values


@pytest.fixture
def context(engine) -> InterpreterContext:
    return InterpreterContext(known_values=get_dimension_values(engine), today=date(2026, 10, 31))


@pytest.mark.parametrize("question, expected", [
    ("What is the total revenue?", "total_revenue"),
    ("Which product generated the highest revenue?", "highest_revenue_product"),
    ("What is the monthly revenue trend?", "monthly_revenue_trend"),
    ("Which region generated the highest sales?", "highest_revenue_region"),
    ("How did revenue change between February and March?", "revenue_comparison"),
    ("revenue by category", "revenue_by_category"),
    ("what is the profit margin?", "unsupported"),
    ("tell me a joke", "unsupported"),
])
def test_rule_based_maps_questions_to_intents(context, question, expected):
    assert RuleBasedInterpreter().interpret(question, context).intent.intent == expected


def test_rule_based_extracts_filters_and_months(context):
    parsed = RuleBasedInterpreter().interpret("Total revenue for Monitor in the South region in May", context).intent
    assert parsed.filters.active() == {"product": "Monitor", "region": "South",
                                      "start_date": "2026-05-01", "end_date": "2026-05-31"}
    comparison = RuleBasedInterpreter().interpret("compare revenue in March vs May", context).intent
    assert comparison.months == ["2026-03", "2026-05"]


def test_comparison_without_two_months_is_unsupported(context):
    assert RuleBasedInterpreter().interpret("How did revenue change?", context).intent.intent == "unsupported"


def test_llm_output_is_used_when_valid(context):
    fake = RunnableLambda(lambda _: Intent(intent="highest_revenue_region", filters={"sales_channel": "Online"}))
    result = HybridInterpreter(LLMInterpreter(fake)).interpret("anything", context)
    assert result.interpreter == "llm" and result.intent.intent == "highest_revenue_region"


def test_llm_failure_falls_back_to_rules(context):
    def boom(_):
        raise TimeoutError("groq unreachable")

    result = HybridInterpreter(LLMInterpreter(RunnableLambda(boom))).interpret("What is the total revenue?", context)
    assert result.interpreter == "rule_based" and result.intent.intent == "total_revenue"


def test_invalid_llm_output_is_rejected_and_falls_back(context):
    bad = RunnableLambda(lambda _: {"intent": "drop_all_tables", "filters": {}})
    result = HybridInterpreter(LLMInterpreter(bad)).interpret("What is the total revenue?", context)
    assert result.interpreter == "rule_based"


def test_llm_saying_unsupported_is_respected(context):
    fake = RunnableLambda(lambda _: Intent(intent="unsupported", reason="Not about sales"))
    result = HybridInterpreter(LLMInterpreter(fake)).interpret("what is the weather", context)
    assert result.interpreter == "llm" and result.intent.intent == "unsupported"


def test_service_rejects_llm_filter_value_that_does_not_exist(settings):
    from app.service import BIService
    fake = RunnableLambda(lambda _: Intent(intent="total_revenue", filters={"product": "Banana"}))
    service = BIService(settings, interpreter=HybridInterpreter(LLMInterpreter(fake)))
    response = service.ask("total revenue for bananas")
    assert response.status == "invalid" and "Banana" in response.answer and response.sql_executed is None


# ---------------------------------------------------------------- Pydantic schema
def test_schema_normalises_llm_blank_values():
    intent = Intent.model_validate({"intent": "total_revenue", "filters": {"region": "all", "product": "", "category": "null"}})
    assert intent.filters.active() == {}


@pytest.mark.parametrize("payload", [
    {"intent": "total_revenue", "filters": {"start_date": "not-a-date"}},
    {"intent": "total_revenue", "filters": {"start_date": "2026-05-01", "end_date": "2026-01-01"}},
    {"intent": "revenue_comparison", "months": ["2026-01"]},
    {"intent": "revenue_comparison", "months": ["2026-01", "2026-13"]},
    {"intent": "revenue_comparison", "months": ["2026-01", "2026-01"]},
    {"intent": "delete_everything"},
])
def test_schema_rejects_invalid_intents(payload):
    with pytest.raises(ValidationError):
        Intent.model_validate(payload)


# ---------------------------------------------------------------- follow-up context
def test_followup_uses_previous_top_entity_but_explicit_filters_win():
    previous = SessionState(Intent(intent="highest_revenue_region"), top_entity=("region", "South"))
    merged, applied = apply_context(Intent(intent="monthly_revenue_trend"), previous, "What about its monthly trend?")
    assert merged.filters.region == "South" and applied == ["region=South"]

    explicit, applied = apply_context(
        Intent(intent="monthly_revenue_trend", filters=Filters(region="North")), previous, "What about its trend for North?")
    assert explicit.filters.region == "North" and applied == []


def test_standalone_question_does_not_inherit_context():
    previous = SessionState(Intent(intent="highest_revenue_region", filters=Filters(region="South")), ("region", "South"))
    merged, applied = apply_context(Intent(intent="total_revenue"), previous, "What is the total revenue?")
    assert merged.filters.active() == {} and applied == []
