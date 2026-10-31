"""Shared fixtures. Every test gets its own temporary SQLite database and no LLM key."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import get_readonly_engine, get_write_engine
from app.ingestion import ingest_csv
from app.interpreter import HybridInterpreter
from app.main import create_app
from app.service import BIService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "sales_data.csv"
CSV_HEADER = "Transaction_Id,Order_date,Product,Category,Region,Sales_channel,Quantity,Unit_price,Discount_amount,Status"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(database_path=tmp_path / "test.db", csv_path=SAMPLE_CSV, groq_api_key=None)


@pytest.fixture
def engine(settings: Settings):
    """Read-only engine over a freshly ingested copy of the supplied CSV."""
    write_engine = get_write_engine(settings.database_path)
    ingest_csv(SAMPLE_CSV, write_engine)
    write_engine.dispose()
    return get_readonly_engine(settings.database_path)


@pytest.fixture
def service(settings: Settings) -> BIService:
    """Service using only the rule-based interpreter (no network, no API key)."""
    return BIService(settings, interpreter=HybridInterpreter(None))


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def write_csv(tmp_path: Path):
    """Factory: write a small CSV from row strings and return its path."""

    def _write(*rows: str, header: str = CSV_HEADER) -> Path:
        path = tmp_path / "custom.csv"
        path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        return path

    return _write
