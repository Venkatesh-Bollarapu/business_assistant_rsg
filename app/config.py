

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    database_path: Path
    csv_path: Path
    groq_api_key: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 20.0
    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.groq_api_key)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_settings() -> Settings:
    """Read settings from the environment (and a local .env file if present)."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        database_path=_resolve(os.getenv("DATABASE_PATH", "data/bi_assistant.db")),
        csv_path=_resolve(os.getenv("DATA_CSV_PATH", "data/sales_data.csv")),
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
