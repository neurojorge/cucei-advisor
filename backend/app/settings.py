from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / "backend" / ".env")


@dataclass(frozen=True)
class Settings:
    def _int_env(name: str, default: int) -> int:
        value = os.environ.get(name)
        try:
            return int(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    offer_data_parquet: str | None = os.environ.get("OFFER_DATA_PARQUET")
    offer_data_csv: str | None = os.environ.get("OFFER_DATA_CSV")
    plan_csv_path: str = os.environ.get(
        "PLAN_CSV_PATH", (ROOT_DIR / "semestres_materias_INFO_ICOM.csv").as_posix()
    )
    reviews_csv_path: str = os.environ.get(
        "REVIEWS_CSV_PATH", (ROOT_DIR / "backend" / "app" / "data" / "evaluaciones_con_departamentos.csv").as_posix()
    )
    aliases_path: str = os.environ.get(
        "PROFESSOR_ALIASES_PATH", (ROOT_DIR / "backend" / "app" / "data" / "professor_aliases.json").as_posix()
    )
    data_dir: str = os.environ.get(
        "OFFER_DATA_DIR", (ROOT_DIR / "backend" / "app" / "data").as_posix()
    )
    cache_dir: str = os.environ.get(
        "SCHEDULE_CACHE_DIR", (ROOT_DIR / "backend" / "app" / "cache").as_posix()
    )
    api_origin: str = os.environ.get("API_ORIGIN", "http://localhost:5173")
    evidence_quote_max_len: int = _int_env("EVIDENCE_QUOTE_MAX_LEN", 320)
