"""CSV ingestion: validate -> clean -> load into SQLite. Repeatable and non-destructive.

Run from the command line:
    python -m app.ingestion                 # uses DATA_CSV_PATH / DATABASE_PATH
    python -m app.ingestion --csv other.csv

Business rules implemented here (also documented in the README):
  * net_revenue = quantity * unit_price - discount_amount  (discount is a line total)
  * a row is revenue-eligible only if status == 'Completed' AND unit_price is present
    AND net_revenue is not negative
  * nothing is silently dropped: unusable rows go to ``rejected_records`` with the raw
    values, and usable-but-odd rows are stored with a ``quality_flag``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from app.config import configure_logging, get_settings
from app.db import get_write_engine

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Transaction_Id",
    "Order_date",
    "Product",
    "Category",
    "Region",
    "Sales_channel",
    "Quantity",
    "Unit_price",
    "Discount_amount",
    "Status",
]
DATE_FORMAT = "%d-%m-%Y"  # the supplied CSV uses DD-MM-YYYY
TEXT_FIELDS = {
    "Product": "product",
    "Category": "category",
    "Region": "region",
    "Sales_channel": "sales_channel",
    "Status": "status",
}


class IngestionError(Exception):
    """The file as a whole cannot be ingested (e.g. missing columns)."""


@dataclass
class IngestionSummary:
    rows_read: int = 0
    rows_loaded: int = 0
    rows_rejected: int = 0
    duplicates_rejected: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)


def _parse_number(raw: str, label: str, reasons: list[str], *, integer: bool = False) -> float | None:
    """Parse a numeric cell. Empty -> None. Invalid or negative -> a rejection reason."""
    if raw == "":
        return None
    try:
        number = float(raw.replace(",", ""))
    except ValueError:
        reasons.append(f"invalid_{label}")
        return None
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        reasons.append(f"invalid_{label}")
        return None
    if number < 0:
        reasons.append(f"negative_{label}")
        return None
    if integer and number != int(number):
        reasons.append(f"invalid_{label}")
        return None
    return number


def validate_row(raw: dict[str, str]) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate and clean one CSV row.

    Returns (clean_record, rejection_reasons). Exactly one of them is meaningful:
    if reasons is non-empty the row is rejected and clean_record is None.
    """
    reasons: list[str] = []

    transaction_id = raw["Transaction_Id"]
    if not transaction_id:
        reasons.append("missing_transaction_id")

    order_date = None
    try:
        order_date = datetime.strptime(raw["Order_date"], DATE_FORMAT).date()
    except ValueError:
        reasons.append("invalid_date")

    texts: dict[str, str] = {}
    for csv_name, column in TEXT_FIELDS.items():
        if not raw[csv_name]:
            reasons.append(f"missing_{column}")
        texts[column] = raw[csv_name]

    quantity_raw = raw["Quantity"]
    if quantity_raw == "":
        reasons.append("missing_quantity")
        quantity = None
    else:
        quantity = _parse_number(quantity_raw, "quantity", reasons, integer=True)

    unit_price = _parse_number(raw["Unit_price"], "unit_price", reasons)  # empty is allowed -> None
    discount = _parse_number(raw["Discount_amount"], "discount_amount", reasons)

    if reasons:
        return None, reasons

    discount = discount or 0.0  # an empty discount cell means "no discount"
    quantity = int(quantity)  # type: ignore[arg-type]
    status = texts["status"].title()  # "completed" / "COMPLETED" -> "Completed"
    flags: list[str] = []

    gross = quantity * unit_price if unit_price is not None else None
    net = gross - discount if gross is not None else None

    if unit_price is None:
        flags.append("missing_unit_price")
    if quantity == 0:
        flags.append("zero_quantity")
    if status == "Returned":
        flags.append("returned")
    elif status != "Completed":
        flags.append("non_completed_status")
    if net is not None and net < 0:
        flags.append("discount_exceeds_gross")

    eligible = status == "Completed" and net is not None and net >= 0

    return {
        "transaction_id": transaction_id,
        "order_date": order_date.isoformat(),
        "month": order_date.strftime("%Y-%m"),
        "product": texts["product"],
        "category": texts["category"],
        "region": texts["region"],
        "sales_channel": texts["sales_channel"],
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_amount": discount,
        "status": status,
        "gross_revenue": gross,
        "net_revenue": net,
        "is_revenue_eligible": 1 if eligible else 0,
        "quality_flag": ",".join(flags) if flags else None,
    }, []


def read_csv(csv_path: Path) -> pd.DataFrame:
    """Read every cell as text so that nothing is coerced or lost before validation."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise IngestionError(f"CSV file not found: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise IngestionError(f"Could not parse CSV: {exc}") from exc
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise IngestionError(f"CSV is missing required columns: {', '.join(missing)}")
    return frame[REQUIRED_COLUMNS].apply(lambda col: col.str.strip())


_UPSERT_SQL = """
INSERT INTO transactions (
    transaction_id, order_date, month, product, category, region, sales_channel,
    quantity, unit_price, discount_amount, status, gross_revenue, net_revenue,
    is_revenue_eligible, quality_flag
) VALUES (
    :transaction_id, :order_date, :month, :product, :category, :region, :sales_channel,
    :quantity, :unit_price, :discount_amount, :status, :gross_revenue, :net_revenue,
    :is_revenue_eligible, :quality_flag
)
ON CONFLICT(transaction_id) DO UPDATE SET
    order_date = excluded.order_date, month = excluded.month, product = excluded.product,
    category = excluded.category, region = excluded.region,
    sales_channel = excluded.sales_channel, quantity = excluded.quantity,
    unit_price = excluded.unit_price, discount_amount = excluded.discount_amount,
    status = excluded.status, gross_revenue = excluded.gross_revenue,
    net_revenue = excluded.net_revenue, is_revenue_eligible = excluded.is_revenue_eligible,
    quality_flag = excluded.quality_flag
"""


def ingest_csv(csv_path: Path, engine: Engine) -> IngestionSummary:
    """Validate the CSV and load it. Safe to run repeatedly (upsert on transaction_id)."""
    frame = read_csv(csv_path)
    summary = IngestionSummary(rows_read=len(frame))

    clean_rows: list[dict[str, Any]] = []
    first_seen: dict[str, dict[str, str]] = {}

    for row_number, raw in enumerate(frame.to_dict(orient="records"), start=1):
        record, reasons = validate_row(raw)
        transaction_id = raw["Transaction_Id"]

        if record is not None and transaction_id in first_seen:
            identical = first_seen[transaction_id] == raw
            reasons = ["duplicate_transaction_id" + ("_identical" if identical else "_conflicting")]
            record = None
            summary.duplicates_rejected += 1

        if record is None:
            summary.rejected.append(
                {
                    "source_row": row_number,
                    "transaction_id": transaction_id or None,
                    "reasons": ",".join(reasons),
                    "raw_record": json.dumps(raw),
                }
            )
            continue

        first_seen[transaction_id] = raw
        clean_rows.append(record)

    summary.rows_loaded = len(clean_rows)
    summary.rows_rejected = len(summary.rejected)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # One transaction: either the whole load succeeds or nothing changes.
    with engine.begin() as conn:
        if clean_rows:
            conn.exec_driver_sql(_UPSERT_SQL, clean_rows)
        # Rejected rows reflect the latest file, so re-running never piles up copies.
        conn.exec_driver_sql("DELETE FROM rejected_records")
        for item in summary.rejected:
            conn.exec_driver_sql(
                "INSERT INTO rejected_records (source_row, transaction_id, reasons, raw_record, ingested_at) "
                "VALUES (:source_row, :transaction_id, :reasons, :raw_record, :ingested_at)",
                {**item, "ingested_at": now},
            )
        conn.exec_driver_sql(
            "INSERT INTO ingestion_runs (run_at, source_file, rows_read, rows_loaded, rows_rejected) "
            "VALUES (:run_at, :source_file, :rows_read, :rows_loaded, :rows_rejected)",
            {
                "run_at": now,
                "source_file": str(csv_path),
                "rows_read": summary.rows_read,
                "rows_loaded": summary.rows_loaded,
                "rows_rejected": summary.rows_rejected,
            },
        )

    logger.info(
        "Ingested %s: read=%d loaded=%d rejected=%d",
        csv_path, summary.rows_read, summary.rows_loaded, summary.rows_rejected,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the sales CSV into SQLite.")
    parser.add_argument("--csv", type=Path, help="CSV path (default: DATA_CSV_PATH)")
    parser.add_argument("--db", type=Path, help="SQLite path (default: DATABASE_PATH)")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    csv_path = args.csv or settings.csv_path
    db_path = args.db or settings.database_path

    engine = get_write_engine(db_path)
    try:
        summary = ingest_csv(csv_path, engine)
    except IngestionError as exc:
        raise SystemExit(f"Ingestion failed: {exc}") from exc

    from app.db import get_readonly_engine
    from app.quality import get_data_quality_report

    report = get_data_quality_report(get_readonly_engine(db_path))
    print(f"\nDatabase: {db_path}")
    print(f"Rows read: {summary.rows_read} | loaded: {summary.rows_loaded} | rejected: {summary.rows_rejected}")
    print(f"Revenue-eligible: {report.revenue_eligible_rows} | excluded from revenue: {report.revenue_excluded_rows}")
    print(f"Missing unit price : {report.missing_unit_price or 'none'}")
    print(f"Zero quantity      : {report.zero_quantity or 'none'}")
    print(f"Returned           : {report.returned or 'none'}")
    for rejected in report.rejected_records:
        print(f"REJECTED row {rejected.source_row} ({rejected.transaction_id}): {rejected.reasons}")


if __name__ == "__main__":
    main()
