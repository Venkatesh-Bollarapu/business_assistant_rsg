"""Deterministic analytics. Expected values are checked against both the assessment's
published figures and an independent pandas calculation on the raw CSV."""

from __future__ import annotations

import pandas as pd
import pytest

from app.analytics import get_summary, run_intent
from app.schemas import Intent
from tests.conftest import SAMPLE_CSV


@pytest.fixture(scope="module")
def oracle() -> pd.DataFrame:
    """Independent computation straight from the CSV (not using any application code)."""
    df = pd.read_csv(SAMPLE_CSV)
    df["Order_date"] = pd.to_datetime(df["Order_date"], format="%d-%m-%Y")
    df["net"] = df["Quantity"] * df["Unit_price"] - df["Discount_amount"]
    return df[(df["Status"] == "Completed") & df["Unit_price"].notna()]


def test_total_revenue(engine, oracle):
    outcome = run_intent(engine, Intent(intent="total_revenue"))
    assert outcome.headline["total_revenue"] == 1_439_200
    assert outcome.headline["total_revenue"] == oracle["net"].sum()
    assert "₹14,39,200" in outcome.answer


def test_highest_revenue_product(engine, oracle):
    outcome = run_intent(engine, Intent(intent="highest_revenue_product"))
    assert outcome.headline["product"] == "Monitor"
    assert outcome.headline["revenue"] == 473_000
    assert oracle.groupby("Product")["net"].sum().idxmax() == "Monitor"
    assert outcome.top_entity == ("product", "Monitor")


def test_highest_revenue_region(engine):
    outcome = run_intent(engine, Intent(intent="highest_revenue_region"))
    assert outcome.headline["region"] == "South"
    assert outcome.headline["revenue"] == 411_200


def test_monthly_revenue_aggregation(engine, oracle):
    outcome = run_intent(engine, Intent(intent="monthly_revenue_trend"))
    got = {row["label"]: row["revenue"] for row in outcome.rows}
    assert got == {
        "2026-01": 210_250, "2026-02": 234_600, "2026-03": 223_900,
        "2026-04": 240_700, "2026-05": 318_500, "2026-06": 211_250,
    }
    expected = oracle.groupby(oracle["Order_date"].dt.strftime("%Y-%m"))["net"].sum().to_dict()
    assert got == expected


def test_month_to_month_comparison(engine):
    outcome = run_intent(engine, Intent(intent="revenue_comparison", months=["2026-02", "2026-03"]))
    assert outcome.headline["revenue_a"] == 234_600
    assert outcome.headline["revenue_b"] == 223_900
    assert outcome.headline["change"] == -10_700
    assert outcome.headline["change_pct"] == -4.6
    assert "decreased" in outcome.answer


def test_comparison_with_month_that_has_no_data_is_no_data(engine):
    outcome = run_intent(engine, Intent(intent="revenue_comparison", months=["2026-02", "2026-12"]))
    assert outcome.status == "no_data"
    assert "December 2026" in outcome.answer


def test_filters_combine(engine, oracle):
    intent = Intent(intent="total_revenue", filters={"region": "south", "category": "Accessories"})
    outcome = run_intent(engine, intent)
    expected = oracle[(oracle["Region"] == "South") & (oracle["Category"] == "Accessories")]["net"].sum()
    assert outcome.headline["total_revenue"] == expected
    assert outcome.params == {"region": "south", "category": "Accessories"}  # bound, not interpolated


def test_date_range_filter(engine, oracle):
    intent = Intent(intent="total_revenue", filters={"start_date": "2026-03-01", "end_date": "2026-04-30"})
    outcome = run_intent(engine, intent)
    window = oracle[(oracle["Order_date"] >= "2026-03-01") & (oracle["Order_date"] <= "2026-04-30")]
    assert outcome.headline["total_revenue"] == window["net"].sum()


def test_empty_result_is_reported_not_fabricated(engine):
    intent = Intent(intent="total_revenue", filters={"product": "Desk", "start_date": "2026-01-01", "end_date": "2026-01-02"})
    outcome = run_intent(engine, intent)
    assert outcome.status == "no_data"
    assert outcome.headline == {} and outcome.rows == []


def test_returned_and_missing_price_transactions_do_not_change_revenue(engine):
    """Monitor total excludes the returned TXN0033 (2 x 9500) - the figure matches the assessment."""
    rows = run_intent(engine, Intent(intent="revenue_by_product")).rows
    monitor = next(r for r in rows if r["label"] == "Monitor")
    assert monitor["revenue"] == 473_000 and monitor["transactions"] == 9  # 10 Monitor rows, 1 returned
    laptop_stand = next(r for r in rows if r["label"] == "Laptop Stand")
    assert laptop_stand["transactions"] == 6  # 7 rows, TXN0051 has no price


def test_summary_is_consistent_with_individual_queries(engine):
    summary = get_summary(engine)
    assert summary.total_net_revenue == 1_439_200
    assert summary.completed_transactions == 49 and summary.total_transactions == 51
    assert summary.top_product["label"] == "Monitor" and summary.top_region["label"] == "South"
    assert sum(r["revenue"] for r in summary.monthly_revenue) == summary.total_net_revenue
