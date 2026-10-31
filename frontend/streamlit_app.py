"""Streamlit frontend for the AI-Powered Business Intelligence Assistant.

The UI is deliberately thin: it only calls the FastAPI backend and renders what comes back.
Run:  streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 40
ACCENT = "#2F6FDE"  # one hue for single-series charts: identity is the chart title, no legend needed

EXAMPLE_QUESTIONS = [
    "What is the total revenue?",
    "Which product generated the highest revenue?",
    "Show the monthly revenue trend",
    "Which region generated the highest revenue?",
    "Compare revenue between February and March",
    "Total revenue for Monitor in the South region",
    "Revenue by category",
    "What about its monthly trend?",
]

st.set_page_config(page_title="AI BI Assistant", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 14px 16px; border-radius: 10px;}
      .answer-card {border-left: 4px solid #2F6FDE; padding: 14px 18px; border-radius: 6px;
                    background: rgba(47,111,222,0.08); font-size: 1.05rem;}
      .error-card {border-left: 4px solid #D9534F; padding: 14px 18px; border-radius: 6px;
                   background: rgba(217,83,79,0.08);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------- API helpers
class ApiError(Exception):
    pass


def call_api(method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
    """Return (status_code, json). Raises ApiError with a friendly message if unreachable."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Cannot reach the API at {API_URL}. Start it with: uvicorn app.main:app --port 8000"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("The request timed out. Please try again.") from exc
    try:
        return response.status_code, response.json()
    except ValueError as exc:
        raise ApiError(f"The API returned an unexpected response (HTTP {response.status_code}).") from exc


@st.cache_data(ttl=30, show_spinner=False)
def fetch(path: str) -> Any:
    status, body = call_api("GET", path)
    if status != 200:
        raise ApiError(f"{path} failed with HTTP {status}")
    return body


def money(value: float) -> str:
    """₹ with Indian digit grouping."""
    whole, _, fraction = f"{abs(value):.2f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    text = ",".join(groups + [tail])
    if fraction != "00":
        text += f".{fraction}"
    return f"{'-' if value < 0 else ''}₹{text}"


# ----------------------------------------------------------------------- charts
def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={"label": "Label", "revenue": "Revenue", "transactions": "Transactions"})
    return frame


def bar_chart(rows: list[dict[str, Any]], title: str, height: int = 260) -> None:
    frame = _frame(rows)
    chart = (
        alt.Chart(frame, title=title)
        .mark_bar(color=ACCENT, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=18)
        .encode(
            y=alt.Y("Label:N", sort="-x", title=None),
            x=alt.X("Revenue:Q", title=None, axis=alt.Axis(format="~s", grid=True, gridOpacity=0.15)),
            tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Revenue:Q", format=",.0f", title="Revenue (₹)"),
                     alt.Tooltip("Transactions:Q")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def line_chart(rows: list[dict[str, Any]], title: str, height: int = 260) -> None:
    frame = _frame(rows)
    base = alt.Chart(frame, title=title).encode(
        x=alt.X("Label:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Revenue:Q", title=None, axis=alt.Axis(format="~s", gridOpacity=0.15), scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("Label:N", title="Month"), alt.Tooltip("Revenue:Q", format=",.0f", title="Revenue (₹)"),
                 alt.Tooltip("Transactions:Q")],
    )
    chart = (base.mark_line(color=ACCENT, strokeWidth=2) + base.mark_point(color=ACCENT, size=60, filled=True)).properties(height=height)
    st.altair_chart(chart, width="stretch")


def show_rows_chart(intent: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    if intent == "monthly_revenue_trend" and len(rows) > 1:
        line_chart(rows, "Revenue by month")
    elif intent == "revenue_comparison":
        bar_chart(rows, "Revenue by month", 120)
    elif len(rows) > 1:
        bar_chart(rows, "Revenue", height=max(120, 34 * len(rows)))


# ----------------------------------------------------------------------- state
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "history" not in st.session_state:
    st.session_state.history = []  # newest first


def ask_question(question: str) -> None:
    question = question.strip()
    if not question:
        st.warning("Please type a question first.")
        return
    try:
        with st.spinner("Interpreting the question and querying the database..."):
            status, body = call_api("POST", "/ask", json={"question": question, "session_id": st.session_state.session_id})
    except ApiError as exc:
        st.session_state.history.insert(0, {"question": question, "error": str(exc)})
        return
    if status == 422:
        detail = body.get("detail")
        message = "Please enter a valid question (1-500 characters)."
        st.session_state.history.insert(0, {"question": question, "error": message})
    elif status >= 500:
        st.session_state.history.insert(0, {"question": question, "error": "The server hit an error. Please try again."})
    else:
        st.session_state.history.insert(0, {"question": question, "response": body})


def render_response(item: dict[str, Any], index: int) -> None:
    st.markdown(f"**🧑 {item['question']}**")
    if "error" in item:
        st.markdown(f"<div class='error-card'>⚠️ {item['error']}</div>", unsafe_allow_html=True)
        return
    r = item["response"]
    if r["status"] in ("unsupported", "invalid"):
        st.markdown(f"<div class='error-card'>🚫 {r['answer']}</div>", unsafe_allow_html=True)
        return

    css = "answer-card"
    st.markdown(f"<div class='{css}'>{r['answer']}</div>", unsafe_allow_html=True)
    caption = [f"Intent: `{r['intent']}`", f"Interpreter: `{r['interpreter']}`"]
    if r.get("filters"):
        caption.append("Filters: " + ", ".join(f"{k}={v}" for k, v in r["filters"].items()))
    if r.get("context_applied"):
        caption.append("From previous question: " + ", ".join(r["context_applied"]))
    st.caption(" · ".join(caption))

    if r["status"] == "no_data":
        return
    result = r.get("result") or {}
    headline, rows = result.get("headline", {}), result.get("rows", [])

    if r["intent"] == "revenue_comparison":  # the change is the point of this question
        headline = {k: headline[k] for k in ("revenue_a", "revenue_b", "change", "change_pct") if k in headline}
    cols = st.columns(min(4, max(1, len(headline))))
    shown = 0
    for key, value in headline.items():
        if shown >= len(cols):
            break
        if isinstance(value, (int, float)):
            is_money = "revenue" in key or key in ("change",)
            text = money(value) if is_money else (f"{value}%" if key.endswith("pct") else f"{value:,}")
        else:
            text = str(value)
        cols[shown].metric(key.replace("_", " ").title(), text)
        shown += 1

    left, right = st.columns([3, 2])
    with left:
        show_rows_chart(r["intent"], rows)
    with right:
        if rows:
            table = _frame(rows)
            st.dataframe(
                table, hide_index=True, width="stretch",
                column_config={"Revenue": st.column_config.NumberColumn("Revenue (₹)", format="₹%.0f")},
            )
    with st.expander("🔍 SQL executed (approved template, parameterised)"):
        st.code(r["sql_executed"], language="sql")
        st.json(r["sql_parameters"] or {})
    st.divider()


# ----------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📊 AI BI Assistant")
    try:
        _, health = call_api("GET", "/health")
        st.success("API online" if health.get("status") == "ok" else "API degraded")
        st.caption("LLM: " + (f"Groq · {health['llm_model']}" if health.get("llm_enabled")
                              else "rule-based interpreter (no GROQ_API_KEY)"))
    except ApiError as exc:
        st.error(str(exc))
    st.subheader("Example questions")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, key=f"ex-{example}", width="stretch"):
            ask_question(example)
    if st.button("🗑️ New conversation", width="stretch"):
        st.session_state.history = []
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()

# ----------------------------------------------------------------------- main page
st.title("AI-Powered Business Intelligence Assistant")
st.write(
    "Ask questions about sales in plain English. An LLM interprets your question; the numbers come "
    "from approved, read-only SQL on the database, never from the model."
)

tab_ask, tab_dashboard, tab_quality = st.tabs(["💬 Ask", "📈 Dashboard", "🧹 Data quality"])

with tab_ask:
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="e.g. Which region generated the highest revenue?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        ask_question(question)
    if not st.session_state.history:
        st.info("Pick an example from the sidebar or type a question above. Follow-ups like "
                "“What about its monthly trend?” use the previous answer as context.")
    for i, item in enumerate(st.session_state.history):
        render_response(item, i)

with tab_dashboard:
    try:
        summary = fetch("/analytics/summary")
    except ApiError as exc:
        st.error(str(exc))
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total net revenue", money(summary["total_net_revenue"]))
        k2.metric("Completed transactions", summary["completed_transactions"])
        k3.metric("Average order value", money(summary["average_order_value"]))
        k4.metric("Top product", summary["top_product"]["label"] if summary["top_product"] else "-")
        k5.metric("Top region", summary["top_region"]["label"] if summary["top_region"] else "-")
        line_chart(summary["monthly_revenue"], "Monthly net revenue", 280)
        c1, c2 = st.columns(2)
        with c1:
            bar_chart(summary["revenue_by_product"], "Revenue by product", 300)
        with c2:
            bar_chart(summary["revenue_by_region"], "Revenue by region", 300)
        with st.expander("Table view of the charts above"):
            st.dataframe(_frame(summary["monthly_revenue"]), hide_index=True, width="stretch")

with tab_quality:
    try:
        quality = fetch("/analytics/data-quality")
    except ApiError as exc:
        st.error(str(exc))
    else:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows read", quality["rows_read"])
        q2.metric("Rows loaded", quality["rows_loaded"])
        q3.metric("Rows rejected", quality["rows_rejected"])
        q4.metric("Excluded from revenue", quality["revenue_excluded_rows"])
        st.subheader("Flagged transactions")
        st.write(f"**Missing unit price** (excluded from revenue): {', '.join(quality['missing_unit_price']) or 'none'}")
        st.write(f"**Zero quantity** (anomaly): {', '.join(quality['zero_quantity']) or 'none'}")
        st.write(f"**Returned** (excluded from revenue): {', '.join(quality['returned']) or 'none'}")
        if quality["other_anomalies"]:
            st.dataframe(pd.DataFrame(quality["other_anomalies"]), hide_index=True)
        if quality["rejected_records"]:
            st.subheader("Rejected records (kept for inspection)")
            st.dataframe(pd.DataFrame(quality["rejected_records"]), hide_index=True)
        st.subheader("Business rules")
        for rule in quality["business_rules"]:
            st.markdown(f"- {rule}")
