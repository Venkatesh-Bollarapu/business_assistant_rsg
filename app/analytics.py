"""Deterministic analytics: intent -> approved SQL template -> rows -> answer text.

Nothing in this module calls an LLM. Every number in an answer comes from a
SQL result, and the SQL comes from the closed set of templates below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine, text

from app.formatting import inr, month_label
from app.schemas import Filters, Intent, SummaryResponse
from app.sql_guard import validate_read_only_sql

logger = logging.getLogger(__name__)

# --- Approved building blocks ------------------------------------------------
# Column names cannot be bound as parameters, so they come ONLY from these
# fixed dictionaries (never from user or LLM text). Values are always bound.
GROUP_COLUMNS = {
    "product": "product",
    "region": "region",
    "category": "category",
    "sales_channel": "sales_channel",
    "month": "month",
}
FILTER_CLAUSES = {
    "product": "LOWER(product) = LOWER(:product)",
    "region": "LOWER(region) = LOWER(:region)",
    "category": "LOWER(category) = LOWER(:category)",
    "sales_channel": "LOWER(sales_channel) = LOWER(:sales_channel)",
    "start_date": "order_date >= :start_date",
    "end_date": "order_date <= :end_date",
}

TOTAL_TEMPLATE = (
    "SELECT ROUND(COALESCE(SUM(net_revenue), 0), 2) AS revenue, COUNT(*) AS transactions "
    "FROM transactions WHERE {where}"
)
GROUPED_TEMPLATE = (
    "SELECT {column} AS label, ROUND(SUM(net_revenue), 2) AS revenue, COUNT(*) AS transactions "
    "FROM transactions WHERE {where} GROUP BY {column} ORDER BY {order}"
)
ORDER_BY_REVENUE = "revenue DESC, label ASC"
ORDER_BY_LABEL = "label ASC"

# intent -> (template kind, grouping dimension)
INTENT_PLAN: dict[str, tuple[str, str | None]] = {
    "total_revenue": ("total", None),
    "highest_revenue_product": ("grouped", "product"),
    "highest_revenue_region": ("grouped", "region"),
    "monthly_revenue_trend": ("grouped", "month"),
    "revenue_comparison": ("grouped", "month"),
    "revenue_by_product": ("grouped", "product"),
    "revenue_by_region": ("grouped", "region"),
    "revenue_by_category": ("grouped", "category"),
    "revenue_by_channel": ("grouped", "sales_channel"),
}
SUPPORTED_INTENTS = tuple(INTENT_PLAN)
PLURALS = {"product": "products", "region": "regions", "category": "categories", "sales channel": "sales channels"}


@dataclass
class AnalyticsOutcome:
    status: str  # "success" | "no_data"
    answer: str
    headline: dict[str, Any]
    rows: list[dict[str, Any]]
    sql: str
    params: dict[str, Any]
    top_entity: tuple[str, str] | None = None  # (filter name, value) for follow-up questions


@dataclass
class _Query:
    sql: str
    params: dict[str, Any] = field(default_factory=dict)


def _where_clause(filters: Filters, months: list[str] | None) -> tuple[str, dict[str, Any]]:
    clauses = ["is_revenue_eligible = 1"]
    params: dict[str, Any] = {}
    for name, clause in FILTER_CLAUSES.items():
        value = getattr(filters, name)
        if value is not None:
            clauses.append(clause)
            params[name] = value
    if months:
        clauses.append("month IN (:month_a, :month_b)")
        params.update(month_a=months[0], month_b=months[1])
    return " AND ".join(clauses), params


def build_query(intent: Intent) -> _Query:
    """Map a validated intent to an approved, parameterised SQL statement."""
    if intent.intent not in INTENT_PLAN:
        raise ValueError(f"Intent '{intent.intent}' has no approved SQL template.")
    kind, dimension = INTENT_PLAN[intent.intent]
    where, params = _where_clause(intent.filters, intent.months)
    if kind == "total":
        sql = TOTAL_TEMPLATE.format(where=where)
    else:
        column = GROUP_COLUMNS[dimension]  # type: ignore[index]
        order = ORDER_BY_LABEL if dimension == "month" else ORDER_BY_REVENUE
        sql = GROUPED_TEMPLATE.format(column=column, where=where, order=order)
    return _Query(sql=sql, params=params)


def execute_query(engine: Engine, query: _Query) -> list[dict[str, Any]]:
    """Run an approved query on the read-only engine, after the SQL guard."""
    validate_read_only_sql(query.sql)
    with engine.connect() as conn:
        result = conn.execute(text(query.sql), query.params)
        return [dict(row) for row in result.mappings().all()]


# --- Answer building ----------------------------------------------------------
def describe_filters(filters: Filters, months: list[str] | None = None) -> str:
    parts = [f"{name.replace('_', ' ')}: {value}" for name, value in filters.active().items()
             if name not in ("start_date", "end_date")]
    if filters.start_date or filters.end_date:
        parts.append(f"period: {filters.start_date or 'start'} to {filters.end_date or 'end'}")
    return f" (filters: {', '.join(parts)})" if parts else ""


def _no_data(intent: Intent, query: _Query) -> AnalyticsOutcome:
    return AnalyticsOutcome(
        status="no_data",
        answer="No completed sales with a valid price match this request"
               f"{describe_filters(intent.filters)}.",
        headline={}, rows=[], sql=query.sql, params=query.params,
    )


def run_intent(engine: Engine, intent: Intent) -> AnalyticsOutcome:
    """Execute an intent and build the answer from the returned rows only."""
    query = build_query(intent)
    rows = execute_query(engine, query)
    kind, dimension = INTENT_PLAN[intent.intent]
    suffix = describe_filters(intent.filters)

    if kind == "total":
        row = rows[0]
        if row["transactions"] == 0:
            return _no_data(intent, query)
        return AnalyticsOutcome(
            status="success",
            answer=f"Total net revenue is {inr(row['revenue'])} across "
                   f"{row['transactions']} completed transaction"
                   f"{'' if row['transactions'] == 1 else 's'}{suffix}.",
            headline={"total_revenue": row["revenue"], "transaction_count": row["transactions"]},
            rows=rows, sql=query.sql, params=query.params,
        )

    if not rows:
        return _no_data(intent, query)

    total = round(sum(r["revenue"] for r in rows), 2)

    if intent.intent == "revenue_comparison":
        by_month = {r["label"]: r["revenue"] for r in rows}
        missing = [m for m in intent.months or [] if m not in by_month]
        if missing:
            labels = ", ".join(month_label(m) for m in missing)
            return AnalyticsOutcome(
                status="no_data",
                answer=f"No completed sales found for {labels}{suffix}, so the months cannot be compared.",
                headline={}, rows=rows, sql=query.sql, params=query.params,
            )
        month_a, month_b = intent.months  # type: ignore[misc]
        revenue_a, revenue_b = by_month[month_a], by_month[month_b]
        change = round(revenue_b - revenue_a, 2)
        pct = round(change / revenue_a * 100, 1) if revenue_a else None
        direction = "increased" if change > 0 else "decreased" if change < 0 else "did not change"
        pct_text = f" ({abs(pct)}%)" if pct is not None and change else ""
        amount_text = f" by {inr(abs(change))}{pct_text}" if change else ""
        return AnalyticsOutcome(
            status="success",
            answer=f"Revenue {direction}{amount_text} from {inr(revenue_a)} in {month_label(month_a)} "
                   f"to {inr(revenue_b)} in {month_label(month_b)}{suffix}.",
            headline={"month_a": month_a, "revenue_a": revenue_a, "month_b": month_b,
                      "revenue_b": revenue_b, "change": change, "change_pct": pct},
            rows=rows, sql=query.sql, params=query.params,
        )

    if intent.intent == "monthly_revenue_trend":
        peak = max(rows, key=lambda r: r["revenue"])
        low = min(rows, key=lambda r: r["revenue"])
        return AnalyticsOutcome(
            status="success",
            answer=f"Revenue over {len(rows)} month{'' if len(rows) == 1 else 's'} totals {inr(total)}. Peak: "
                   f"{month_label(peak['label'])} ({inr(peak['revenue'])}); lowest: "
                   f"{month_label(low['label'])} ({inr(low['revenue'])}){suffix}.",
            headline={"total_revenue": total, "peak_month": peak["label"], "peak_revenue": peak["revenue"],
                      "lowest_month": low["label"], "lowest_revenue": low["revenue"]},
            rows=rows, sql=query.sql, params=query.params,
        )

    # Remaining intents are grouped by a business dimension; rows are sorted by revenue.
    top = rows[0]
    share = round(top["revenue"] / total * 100, 1) if total else 0.0
    noun = {"product": "product", "region": "region", "category": "category",
            "sales_channel": "sales channel"}[dimension]  # type: ignore[index]
    if intent.intent.startswith("highest_"):
        answer = (f"{top['label']} generated the highest revenue among {noun}s: "
                  f"{inr(top['revenue'])} ({share}% of {inr(total)}){suffix}.")
    else:
        answer = (f"Revenue by {noun}{suffix}: {top['label']} leads with {inr(top['revenue'])} "
                  f"({share}% of {inr(total)}) across {len(rows)} {PLURALS[noun] if len(rows) != 1 else noun}.")
    return AnalyticsOutcome(
        status="success",
        answer=answer,
        headline={noun.replace(" ", "_"): top["label"], "revenue": top["revenue"],
                  "share_of_total_pct": share, "total_revenue": total},
        rows=rows, sql=query.sql, params=query.params,
        top_entity=(dimension, top["label"]),  # type: ignore[arg-type]
    )


# --- Reference data & summary --------------------------------------------------
def get_dimension_values(engine: Engine) -> dict[str, list[str]]:
    """Known values used to validate filters and to give the LLM real context."""
    sql = {
        "product": "SELECT DISTINCT product FROM transactions ORDER BY 1",
        "region": "SELECT DISTINCT region FROM transactions ORDER BY 1",
        "category": "SELECT DISTINCT category FROM transactions ORDER BY 1",
        "sales_channel": "SELECT DISTINCT sales_channel FROM transactions ORDER BY 1",
        "month": "SELECT DISTINCT month FROM transactions ORDER BY 1",
    }
    with engine.connect() as conn:
        return {key: [r[0] for r in conn.execute(text(stmt))] for key, stmt in sql.items()}


def get_summary(engine: Engine) -> SummaryResponse:
    def run(name: str) -> AnalyticsOutcome:
        return run_intent(engine, Intent(intent=name))  # type: ignore[arg-type]

    total = run("total_revenue")
    with engine.connect() as conn:
        counts = conn.execute(text(
            "SELECT COUNT(*) AS total, MIN(order_date) AS first_date, MAX(order_date) AS last_date "
            "FROM transactions")).mappings().one()
    revenue = total.headline.get("total_revenue", 0.0)
    completed = total.headline.get("transaction_count", 0)
    by_product, by_region = run("revenue_by_product"), run("revenue_by_region")

    def top(outcome: AnalyticsOutcome) -> dict[str, Any] | None:
        return {"label": outcome.rows[0]["label"], "revenue": outcome.rows[0]["revenue"]} if outcome.rows else None

    return SummaryResponse(
        total_net_revenue=revenue,
        completed_transactions=completed,
        total_transactions=counts["total"],
        average_order_value=round(revenue / completed, 2) if completed else 0.0,
        date_range={"first": counts["first_date"], "last": counts["last_date"]},
        top_product=top(by_product),
        top_region=top(by_region),
        monthly_revenue=run("monthly_revenue_trend").rows,
        revenue_by_product=by_product.rows,
        revenue_by_region=by_region.rows,
        revenue_by_category=run("revenue_by_category").rows,
        revenue_by_channel=run("revenue_by_channel").rows,
    )
