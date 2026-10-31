"""SQL safety: guard, read-only connection, parameterisation and injection resistance."""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy.exc import OperationalError

from app.analytics import SUPPORTED_INTENTS, build_query
from app.schemas import Intent
from app.sql_guard import UnsafeSQLError, validate_read_only_sql

DANGEROUS_SQL = [
    "DELETE FROM transactions",
    "DROP TABLE transactions",
    "UPDATE transactions SET net_revenue = 0",
    "INSERT INTO transactions VALUES (1)",
    "SELECT * FROM transactions; DROP TABLE transactions",
    "SELECT * FROM transactions -- comment",
    "SELECT * FROM transactions /* x */",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM rejected_records",
    "PRAGMA table_info(transactions)",
    "ATTACH DATABASE 'x.db' AS x",
    "",
    "   ",
    "WITH x AS (DELETE FROM transactions) SELECT 1",
    "EXPLAIN SELECT * FROM transactions",           # only the SELECT-only rule catches this
    "WITH t AS (SELECT 1) SELECT * FROM transactions",  # CTEs are not part of the approved templates
]


@pytest.mark.parametrize("sql", DANGEROUS_SQL)
def test_guard_rejects_dangerous_or_non_select_sql(sql):
    with pytest.raises(UnsafeSQLError):
        validate_read_only_sql(sql)


def test_every_approved_template_passes_the_guard():
    for name in SUPPORTED_INTENTS:
        months = ["2026-01", "2026-02"] if name == "revenue_comparison" else None
        intent = Intent(
            intent=name, months=months,
            filters={"product": "Monitor", "region": "South", "category": "Electronics",
                     "sales_channel": "Online", "start_date": "2026-01-01", "end_date": "2026-06-30"},
        )
        validate_read_only_sql(build_query(intent).sql)


def test_connection_is_physically_read_only(engine):
    with pytest.raises(OperationalError, match="readonly"):
        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM transactions")


def test_filter_values_are_bound_parameters_never_part_of_the_sql_text():
    payload = "Monitor' OR 1=1; DROP TABLE transactions; --"
    query = build_query(Intent(intent="total_revenue", filters={"product": payload}))
    assert payload not in query.sql and "DROP" not in query.sql.upper()
    assert query.params == {"product": payload}


@pytest.mark.parametrize("question", [
    "'; DELETE FROM transactions; --",
    "DROP TABLE transactions",
    "update transactions set net_revenue = 0",
    "SELECT * FROM transactions",
    "total revenue UNION SELECT 1 FROM transactions",
])
def test_destructive_or_raw_sql_questions_are_refused_and_data_is_untouched(service, settings, question):
    response = service.ask(question)
    assert response.status == "unsupported" and response.sql_executed is None
    with sqlite3.connect(settings.database_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 51


def test_injection_text_in_a_question_only_ever_becomes_a_bound_parameter(service, settings):
    response = service.ask("total revenue for product Monitor' OR 1=1")
    assert response.status == "success"
    assert response.sql_parameters == {"product": "Monitor"}       # the payload never reaches SQL
    assert "OR 1=1" not in response.sql_executed
    assert response.result["headline"]["total_revenue"] == 473_000  # Monitor only, not the whole table
