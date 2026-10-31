"""Data ingestion, validation and data-quality behaviour."""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.db import get_readonly_engine, get_write_engine
from app.ingestion import IngestionError, ingest_csv
from app.quality import get_data_quality_report


def test_sample_csv_loads_with_expected_anomalies(engine):
    report = get_data_quality_report(engine)
    assert (report.rows_read, report.rows_loaded, report.rows_rejected) == (51, 51, 0)
    assert report.missing_unit_price == ["TXN0051"]
    assert report.zero_quantity == ["TXN0034"]
    assert report.returned == ["TXN0033"]
    assert report.revenue_eligible_rows == 49  # 51 minus returned and missing-price rows


def test_returned_and_missing_price_rows_are_stored_but_excluded_from_revenue(settings, engine):
    with sqlite3.connect(settings.database_path) as conn:
        rows = {r[0]: r[1:] for r in conn.execute(
            "SELECT transaction_id, is_revenue_eligible, net_revenue FROM transactions "
            "WHERE transaction_id IN ('TXN0033', 'TXN0034', 'TXN0051')")}
    assert rows["TXN0033"][0] == 0            # returned -> excluded
    assert rows["TXN0051"] == (0, None)        # missing price -> excluded, no revenue computed
    assert rows["TXN0034"] == (1, 0.0)         # zero quantity -> kept, contributes 0 (and is flagged)


def test_discount_is_subtracted_as_a_line_total(settings, engine):
    with sqlite3.connect(settings.database_path) as conn:
        net = conn.execute("SELECT net_revenue FROM transactions WHERE transaction_id='TXN0004'").fetchone()[0]
    assert net == 6 * 9500 - 500


def test_ingestion_is_repeatable_without_duplicates(settings):
    write_engine = get_write_engine(settings.database_path)
    for _ in range(3):
        ingest_csv(settings.csv_path, write_engine)
    with sqlite3.connect(settings.database_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 51


def test_invalid_rows_are_rejected_with_raw_values_preserved(settings, write_csv):
    csv_path = write_csv(
        "T1,03-01-2026,Monitor,Electronics,South,Online,2,9500,0,Completed",   # valid
        "T1,04-01-2026,Monitor,Electronics,South,Online,3,9500,0,Completed",   # duplicate id, conflicting
        "T2,31-02-2026,Desk,Furniture,North,Retail,1,12500,0,Completed",       # impossible date
        "T3,05-01-2026,Desk,Furniture,North,Retail,-4,12500,0,Completed",      # negative quantity
        "T4,05-01-2026,Desk,Furniture,North,Retail,1,abc,0,Completed",         # non-numeric price
        "T5,05-01-2026,,Furniture,North,Retail,1,100,0,Completed",             # missing product
    )
    write_engine = get_write_engine(settings.database_path)
    summary = ingest_csv(csv_path, write_engine)
    assert (summary.rows_read, summary.rows_loaded, summary.rows_rejected) == (6, 1, 5)
    assert summary.duplicates_rejected == 1

    report = get_data_quality_report(get_readonly_engine(settings.database_path))
    reasons = {r.transaction_id: r.reasons for r in report.rejected_records}
    assert reasons["T2"] == "invalid_date"
    assert reasons["T3"] == "negative_quantity"
    assert reasons["T4"] == "invalid_unit_price"
    assert reasons["T5"] == "missing_product"
    assert "duplicate_transaction_id_conflicting" in reasons["T1"]
    assert report.duplicate_ids_rejected == 1
    assert report.rejected_records[0].raw_record["Product"] == "Monitor"  # nothing silently discarded


def test_identical_duplicate_is_labelled_and_first_row_wins(settings, write_csv):
    row = "T1,03-01-2026,Monitor,Electronics,South,Online,2,9500,0,Completed"
    ingest_csv(write_csv(row, row), get_write_engine(settings.database_path))
    report = get_data_quality_report(get_readonly_engine(settings.database_path))
    assert report.rows_loaded == 1
    assert "duplicate_transaction_id_identical" in report.rejected_records[0].reasons


def test_missing_required_column_fails_fast(settings, write_csv):
    csv_path = write_csv("T1,03-01-2026,Monitor", header="Transaction_Id,Order_date,Product")
    with pytest.raises(IngestionError, match="missing required columns"):
        ingest_csv(csv_path, get_write_engine(settings.database_path))


def test_missing_file_raises_clear_error(settings, tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        ingest_csv(tmp_path / "nope.csv", get_write_engine(settings.database_path))


def test_discount_larger_than_sale_is_flagged_and_excluded(settings, write_csv):
    csv_path = write_csv("T1,03-01-2026,Monitor,Electronics,South,Online,1,100,500,Completed")
    ingest_csv(csv_path, get_write_engine(settings.database_path))
    report = get_data_quality_report(get_readonly_engine(settings.database_path))
    assert [i.issue for i in report.other_anomalies] == ["discount_exceeds_gross"]
    assert report.revenue_eligible_rows == 0
    assert json.dumps(report.model_dump())  # report is JSON-serialisable
