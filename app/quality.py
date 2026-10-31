"""Data-quality report built from what is actually stored in the database."""

from __future__ import annotations

import json

from sqlalchemy import Engine, text

from app.schemas import DataQualityIssue, DataQualityReport, RejectedRecord

BUSINESS_RULES = [
    "Net revenue = quantity x unit_price - discount_amount (discount_amount is a line total).",
    "Only transactions with status 'Completed' and a valid unit price count as revenue.",
    "Returned transactions are stored but excluded from revenue.",
    "Transactions with a missing unit price are stored but excluded from revenue and reported.",
    "Zero-quantity transactions are flagged as anomalies (they contribute 0 to revenue).",
    "Rows that cannot be parsed, and duplicate transaction IDs, are rejected into "
    "rejected_records with their raw values, never silently discarded.",
]


def get_data_quality_report(engine: Engine) -> DataQualityReport:
    with engine.connect() as conn:
        run = conn.execute(
            text("SELECT run_at, source_file, rows_read, rows_loaded, rows_rejected "
                 "FROM ingestion_runs ORDER BY id DESC LIMIT 1")
        ).mappings().first()
        counts = conn.execute(
            text("SELECT COUNT(*) AS total, COALESCE(SUM(is_revenue_eligible), 0) AS eligible FROM transactions")
        ).mappings().one()
        flagged = conn.execute(
            text("SELECT transaction_id, quality_flag FROM transactions "
                 "WHERE quality_flag IS NOT NULL ORDER BY transaction_id")
        ).mappings().all()
        rejected = conn.execute(
            text("SELECT source_row, transaction_id, reasons, raw_record FROM rejected_records ORDER BY id")
        ).mappings().all()

    by_flag: dict[str, list[str]] = {}
    for row in flagged:
        for flag in row["quality_flag"].split(","):
            by_flag.setdefault(flag, []).append(row["transaction_id"])

    handled = {"missing_unit_price", "zero_quantity", "returned"}
    other = [
        DataQualityIssue(transaction_id=tid, issue=flag, detail=flag.replace("_", " "))
        for flag, ids in sorted(by_flag.items()) if flag not in handled for tid in ids
    ]
    duplicates = sum(1 for r in rejected if "duplicate_transaction_id" in r["reasons"])

    return DataQualityReport(
        source_file=run["source_file"] if run else None,
        last_ingested_at=run["run_at"] if run else None,
        rows_read=run["rows_read"] if run else counts["total"],
        rows_loaded=counts["total"],
        rows_rejected=len(rejected),
        duplicate_ids_rejected=duplicates,
        revenue_eligible_rows=int(counts["eligible"]),
        revenue_excluded_rows=int(counts["total"] - counts["eligible"]),
        missing_unit_price=by_flag.get("missing_unit_price", []),
        zero_quantity=by_flag.get("zero_quantity", []),
        returned=by_flag.get("returned", []),
        other_anomalies=other,
        rejected_records=[
            RejectedRecord(
                source_row=r["source_row"],
                transaction_id=r["transaction_id"],
                reasons=r["reasons"],
                raw_record=json.loads(r["raw_record"]),
            )
            for r in rejected
        ],
        business_rules=BUSINESS_RULES,
    )
