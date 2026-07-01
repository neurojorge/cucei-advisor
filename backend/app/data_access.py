from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .adapters import adapt_schema_offer
from .plan_loader import load_plan_df
from .reviews_features import load_reviews
from .settings import Settings

logger = logging.getLogger(__name__)

_CACHE: Dict[str, object] = {}
_LOGGED_COLUMNS = False


def _cycle_key(path: Path) -> Tuple[int, str]:
    name = path.stem
    digits = "".join([ch for ch in name if ch.isdigit()])
    rest = "".join([ch for ch in name if not ch.isdigit()])
    return (int(digits) if digits else -1, rest)


def _log_offer_columns(df: pd.DataFrame) -> None:
    global _LOGGED_COLUMNS
    if _LOGGED_COLUMNS:
        return
    logger.info("Columnas oferta detectadas: %s", list(df.columns))
    print(f"Columnas oferta detectadas: {list(df.columns)}")
    _LOGGED_COLUMNS = True


def _load_offer_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    _log_offer_columns(df)
    return adapt_schema_offer(df)


def sanitize_availability(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float | bool]]:
    if df.empty:
        return df, {
            "offer_avail_known_pct": 0.0,
            "offer_avail_nonzero_pct": 0.0,
            "offer_avail_all_zero": True,
        }

    cupo_num = pd.to_numeric(df.get("cupo"), errors="coerce")
    disp_num = pd.to_numeric(df.get("disponibles"), errors="coerce")
    total = len(df)

    known_pct = round(100 * float(disp_num.notna().sum()) / total, 2) if total else 0.0
    nonzero_pct = round(100 * float((disp_num > 0).sum()) / total, 2) if total else 0.0

    max_disp = disp_num.dropna().max()
    max_cupo = cupo_num.dropna().max()
    if pd.isna(max_disp):
        max_disp = 0
    if pd.isna(max_cupo):
        max_cupo = 0
    all_zero = bool(max_disp == 0 and max_cupo == 0)

    if all_zero:
        df = df.copy()
        mask = disp_num.fillna(0).eq(0) & cupo_num.fillna(0).eq(0)
        df.loc[mask, "disponibles"] = None
        df.loc[mask, "cupo"] = None

    return df, {
        "offer_avail_known_pct": known_pct,
        "offer_avail_nonzero_pct": nonzero_pct,
        "offer_avail_all_zero": all_zero,
    }


def load_offer_latest(settings: Settings) -> Tuple[pd.DataFrame, Dict[str, str]]:
    override_parquet = settings.offer_data_parquet
    override_csv = settings.offer_data_csv

    if override_parquet:
        p = Path(override_parquet)
        if not p.exists():
            raise FileNotFoundError(f"No se encontro archivo en {p}")
        return _load_offer_from_path(p), {"source": p.as_posix()}
    if override_csv:
        p = Path(override_csv)
        if not p.exists():
            raise FileNotFoundError(f"No se encontro archivo en {p}")
        return _load_offer_from_path(p), {"source": p.as_posix()}

    data_dirs = [
        Path(settings.data_dir),
        Path(settings.data_dir).parents[2] / "webscraping" / "data",
        Path(settings.data_dir).parents[2] / "data",
    ]

    files: list[Path] = []
    for base in data_dirs:
        if not base.exists():
            continue
        files.extend(base.glob("oferta_cucei_*.*"))

    files = [f for f in files if f.suffix in {".parquet", ".csv"}]
    if not files:
        raise FileNotFoundError("No se encontro dataset de oferta en backend/app/data o webscraping/data")

    files.sort(key=_cycle_key, reverse=True)
    chosen = files[0]
    df = _load_offer_from_path(chosen)
    meta = {"source": chosen.as_posix()}
    return df, meta


def get_offer(settings: Settings) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if "offer" not in _CACHE:
        df, meta = load_offer_latest(settings)
        df, avail_meta = sanitize_availability(df)
        cycle = ""
        if "ciclo" in df.columns and not df["ciclo"].empty:
            cycles = sorted(df["ciclo"].astype(str).dropna().unique().tolist())
            cycle = cycles[-1] if cycles else ""
        meta = {
            **meta,
            "cycle": cycle,
            "offer_rows": int(len(df)),
            "last_updated": None,
        }
        meta.update(avail_meta)
        meta["availability_quality"] = "unknown" if meta.get("offer_avail_all_zero") else "available"
        try:
            meta_path = Path(meta.get("source", ""))
            if meta_path.exists():
                meta["last_updated"] = meta_path.stat().st_mtime
        except Exception:
            pass
        _CACHE["offer"] = df
        _CACHE["offer_meta"] = meta
    return _CACHE["offer"], _CACHE["offer_meta"]  # type: ignore


def get_reviews(settings: Settings) -> pd.DataFrame:
    if "reviews" not in _CACHE:
        reviews_path = Path(settings.reviews_csv_path)
        if not reviews_path.exists():
            # fallback to root if present
            root_path = Path(settings.reviews_csv_path).parents[3] / "evaluaciones_con_departamentos.csv"
            reviews_path = root_path if root_path.exists() else reviews_path
        _CACHE["reviews"] = load_reviews(reviews_path.as_posix()) if reviews_path.exists() else pd.DataFrame()
    return _CACHE["reviews"]  # type: ignore


def get_plan(settings: Settings) -> pd.DataFrame:
    if "plan" not in _CACHE:
        _CACHE["plan"] = load_plan_df(settings.plan_csv_path)
    return _CACHE["plan"]  # type: ignore
