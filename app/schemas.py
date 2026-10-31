"""Pydantic models: the contract between the LLM, the backend and the API clients.

The LLM may only produce an ``Intent``. Anything that does not validate against
these models is rejected before a single SQL statement is run.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IntentName = Literal[
    "total_revenue",
    "highest_revenue_product",
    "highest_revenue_region",
    "monthly_revenue_trend",
    "revenue_comparison",
    "revenue_by_product",
    "revenue_by_region",
    "revenue_by_category",
    "revenue_by_channel",
    "unsupported",
]

MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_BLANK_VALUES = {"", "null", "none", "all", "any", "n/a", "na"}


def _blank_to_none(value: Any) -> Any:
    """LLMs often send "" / "null" / "all" for 'no filter'. Normalise to None."""
    if isinstance(value, str) and value.strip().lower() in _BLANK_VALUES:
        return None
    if isinstance(value, str):
        return value.strip()
    return value


class Filters(BaseModel):
    """Optional filters. Only these fields can ever reach a SQL query."""

    model_config = ConfigDict(extra="ignore")

    product: str | None = Field(default=None, description="Exact product name, e.g. 'Monitor'")
    region: str | None = Field(default=None, description="North, South, East or West")
    category: str | None = Field(default=None, description="Product category, e.g. 'Accessories'")
    sales_channel: str | None = Field(default=None, description="'Online' or 'Retail'")
    start_date: str | None = Field(default=None, description="Inclusive start, YYYY-MM-DD")
    end_date: str | None = Field(default=None, description="Inclusive end, YYYY-MM-DD")

    @field_validator("product", "region", "category", "sales_channel", "start_date", "end_date", mode="before")
    @classmethod
    def _normalise_blank(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("start_date", "end_date")
    @classmethod
    def _valid_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError(f"'{value}' is not a valid YYYY-MM-DD date") from exc

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "Filters":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        return self

    def active(self) -> dict[str, str]:
        """Only the filters that are actually set."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Intent(BaseModel):
    """Structured interpretation of a user's question."""

    model_config = ConfigDict(extra="ignore")

    intent: IntentName = Field(description="One of the supported analytics operations, or 'unsupported'")
    filters: Filters = Field(default_factory=Filters)
    months: list[str] | None = Field(
        default=None,
        description="Exactly two YYYY-MM months, only for 'revenue_comparison'",
    )
    reason: str | None = Field(default=None, description="Why the question is unsupported")

    @field_validator("filters", mode="before")
    @classmethod
    def _none_filters(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("months", mode="before")
    @classmethod
    def _blank_months(cls, value: Any) -> Any:
        if value in (None, "", []):
            return None
        return value

    @model_validator(mode="after")
    def _check_months(self) -> "Intent":
        if self.intent == "revenue_comparison":
            if not self.months or len(self.months) != 2 or self.months[0] == self.months[1]:
                raise ValueError("revenue_comparison needs exactly two different months (YYYY-MM)")
            for month in self.months:
                if not MONTH_PATTERN.match(month):
                    raise ValueError(f"'{month}' is not a valid YYYY-MM month")
        else:
            self.months = None
        return self


# ---------------------------------------------------------------- API models
ResponseStatus = Literal["success", "no_data", "unsupported", "invalid"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class AskResponse(BaseModel):
    question: str
    status: ResponseStatus
    intent: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    answer: str
    result: dict[str, Any] | None = Field(
        default=None, description="{'headline': {...key figures...}, 'rows': [...supporting rows...]}"
    )
    sql_executed: str | None = None
    sql_parameters: dict[str, Any] | None = None
    interpreter: Literal["llm", "rule_based"] | None = None
    context_applied: list[str] = Field(default_factory=list)
    session_id: str | None = None


class DataQualityIssue(BaseModel):
    transaction_id: str
    issue: str
    detail: str


class RejectedRecord(BaseModel):
    source_row: int | None
    transaction_id: str | None
    reasons: str
    raw_record: dict[str, Any]


class DataQualityReport(BaseModel):
    source_file: str | None
    last_ingested_at: str | None
    rows_read: int
    rows_loaded: int
    rows_rejected: int
    duplicate_ids_rejected: int
    revenue_eligible_rows: int
    revenue_excluded_rows: int
    missing_unit_price: list[str]
    zero_quantity: list[str]
    returned: list[str]
    other_anomalies: list[DataQualityIssue]
    rejected_records: list[RejectedRecord]
    business_rules: list[str]


class SummaryResponse(BaseModel):
    total_net_revenue: float
    completed_transactions: int
    total_transactions: int
    average_order_value: float
    date_range: dict[str, str | None]
    top_product: dict[str, Any] | None
    top_region: dict[str, Any] | None
    monthly_revenue: list[dict[str, Any]]
    revenue_by_product: list[dict[str, Any]]
    revenue_by_region: list[dict[str, Any]]
    revenue_by_category: list[dict[str, Any]]
    revenue_by_channel: list[dict[str, Any]]
