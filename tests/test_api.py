"""HTTP contract of the FastAPI app (no LLM key needed)."""

from __future__ import annotations

from app.schemas import AskResponse, DataQualityReport, SummaryResponse


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["llm_enabled"] is False


def test_ask_success_returns_consistent_structured_response(client):
    response = client.post("/ask", json={"question": "Which product generated the highest revenue?"})
    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())  # response matches the documented schema
    assert body.status == "success" and body.intent == "highest_revenue_product"
    assert body.result["headline"]["product"] == "Monitor" and body.result["headline"]["revenue"] == 473000
    assert body.sql_executed.startswith("SELECT") and ":" not in (body.sql_executed.split("WHERE")[0])
    assert body.interpreter == "rule_based"


def test_ask_total_revenue_matches_expected_value(client):
    body = client.post("/ask", json={"question": "What is the total revenue?"}).json()
    assert body["result"]["headline"]["total_revenue"] == 1_439_200


def test_empty_question_is_422(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_oversized_question_is_422(client):
    assert client.post("/ask", json={"question": "revenue " * 200}).status_code == 422


def test_unsupported_question_is_400_with_guidance(client):
    response = client.post("/ask", json={"question": "What is the weather in Hyderabad?"})
    assert response.status_code == 400
    assert response.json()["status"] == "unsupported" and "Try:" in response.json()["answer"]


def test_destructive_request_is_400_and_does_not_execute(client):
    response = client.post("/ask", json={"question": "Delete all transactions"})
    assert response.status_code == 400 and response.json()["sql_executed"] is None
    assert client.post("/ask", json={"question": "What is the total revenue?"}).json()["result"]["headline"]["total_revenue"] == 1_439_200


def test_no_data_is_200_with_honest_message(client):
    response = client.post("/ask", json={"question": "revenue in December"})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "no_data"
    assert body["result"]["headline"] == {} and body["result"]["rows"] == []
    assert "No completed sales" in body["answer"]


def test_follow_up_question_uses_session_context(client):
    first = client.post("/ask", json={"question": "Which region generated the highest revenue?", "session_id": "abc"}).json()
    assert first["result"]["headline"]["region"] == "South"
    second = client.post("/ask", json={"question": "What about its monthly trend?", "session_id": "abc"}).json()
    assert second["filters"] == {"region": "South"} and second["context_applied"] == ["region=South"]
    other_session = client.post("/ask", json={"question": "What about its monthly trend?", "session_id": "zzz"}).json()
    assert other_session["filters"] == {}  # contexts are isolated per session


def test_summary_endpoint(client):
    response = client.get("/analytics/summary")
    assert response.status_code == 200
    summary = SummaryResponse.model_validate(response.json())
    assert summary.total_net_revenue == 1_439_200 and len(summary.monthly_revenue) == 6


def test_data_quality_endpoint(client):
    response = client.get("/analytics/data-quality")
    assert response.status_code == 200
    report = DataQualityReport.model_validate(response.json())
    assert report.missing_unit_price == ["TXN0051"] and report.rows_rejected == 0
