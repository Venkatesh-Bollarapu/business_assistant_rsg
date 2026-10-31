"""SQLite engines and schema.

Two engines are used on purpose:
  * a *write* engine, used only by the ingestion pipeline;
  * a *read-only* engine, used for every analytics query. SQLite itself
    refuses writes on it (``mode=ro`` + ``PRAGMA query_only``), so even a
    bug in the SQL layer could not modify data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import Engine, create_engine, event

SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id      TEXT PRIMARY KEY,
        order_date          TEXT NOT NULL,            -- ISO 8601: YYYY-MM-DD
        month               TEXT NOT NULL,            -- YYYY-MM (derived, for grouping)
        product             TEXT NOT NULL,
        category            TEXT NOT NULL,
        region              TEXT NOT NULL,
        sales_channel       TEXT NOT NULL,
        quantity            INTEGER NOT NULL CHECK (quantity >= 0),
        unit_price          REAL CHECK (unit_price IS NULL OR unit_price >= 0),
        discount_amount     REAL NOT NULL DEFAULT 0 CHECK (discount_amount >= 0),
        status              TEXT NOT NULL,
        gross_revenue       REAL,                     -- quantity * unit_price (NULL if no price)
        net_revenue         REAL,                     -- gross_revenue - discount_amount
        is_revenue_eligible INTEGER NOT NULL CHECK (is_revenue_eligible IN (0, 1)),
        quality_flag        TEXT                      -- comma separated anomaly flags, NULL if clean
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tx_month  ON transactions(month)",
    "CREATE INDEX IF NOT EXISTS idx_tx_region ON transactions(region)",
    "CREATE INDEX IF NOT EXISTS idx_tx_product ON transactions(product)",
    """
    CREATE TABLE IF NOT EXISTS rejected_records (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        source_row     INTEGER,                       -- 1-based data row number in the CSV
        transaction_id TEXT,
        reasons        TEXT NOT NULL,
        raw_record     TEXT NOT NULL,                 -- original values as JSON, never discarded
        ingested_at    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_runs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at        TEXT NOT NULL,
        source_file   TEXT NOT NULL,
        rows_read     INTEGER NOT NULL,
        rows_loaded   INTEGER NOT NULL,
        rows_rejected INTEGER NOT NULL
    )
    """,
]


def get_write_engine(db_path: Path) -> Engine:
    """Engine used by ingestion only. Creates the file and schema if needed."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.exec_driver_sql(statement)
    return engine


def get_readonly_engine(db_path: Path) -> Engine:
    """Engine used for all analytics. The database file must already exist."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    def _connect() -> sqlite3.Connection:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)

    engine = create_engine("sqlite://", creator=_connect)

    @event.listens_for(engine, "connect")
    def _query_only(dbapi_connection, _record):  # pragma: no cover - trivial
        dbapi_connection.execute("PRAGMA query_only = ON")

    return engine


def database_has_data(db_path: Path) -> bool:
    """True if the DB file exists and contains at least one transaction."""
    db_path = Path(db_path)
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] > 0
    except sqlite3.Error:
        return False
