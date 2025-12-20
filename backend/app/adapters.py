from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

CANONICAL_COLS = [
    "carrera_clave",
    "clave_materia",
    "materia",
    "nrc",
    "seccion",
    "profesor",
    "horario_raw",
    "hora_inicio",
    "hora_fin",
    "dias",
    "cupo",
    "disponibles",
    "ciclo",
    "centro",
]


def _normalize_text(value: str) -> str:
    value = str(value or "").strip().upper()
    value = value.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
    value = value.replace("Ñ", "N")
    return value


def _find_col(
    df: pd.DataFrame,
    candidates: Iterable[str],
    contains: Iterable[str] | None = None,
    value_tokens: Iterable[str] | None = None,
) -> str:
    contains = list(contains or [])
    value_tokens = [t.upper() for t in (value_tokens or [])]
    lower_cols = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]

    for c in df.columns:
        cl = c.lower()
        if any(token in cl for token in contains):
            return c

    if value_tokens:
        for c in df.columns:
            series = df[c].astype(str).head(200).map(_normalize_text)
            if series.str.contains("|".join(map(re.escape, value_tokens))).any():
                return c

    return ""


def _normalize_carrera(value: str) -> str:
    clean = _normalize_text(value)
    if "INFORMATICA" in clean or clean.startswith("INFO"):
        return "INFO"
    if "COMPUTACION" in clean or clean.startswith("ICOM") or "COMPUT" in clean:
        return "ICOM"
    return clean


def _optional_int_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    return parsed.astype("Int64").astype(object).where(parsed.notna(), None)


def adapt_schema_offer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristically map an arbitrary oferta DataFrame into the canonical schema expected by the app.
    Raises a clear error if critical columns cannot be inferred.
    """
    df = df.copy()
    col_carrera = _find_col(
        df,
        ["carrera_clave", "carrera"],
        contains=["carrera", "programa"],
        value_tokens=["INFO", "ICOM", "INFORMATICA", "COMPUTACION"],
    )
    col_clave = _find_col(df, ["clave_materia", "clave", "cve"], contains=["clave", "cve"])
    col_mat = _find_col(df, ["materia", "asignatura"], contains=["materia"])
    col_nrc = _find_col(df, ["nrc"], contains=["nrc"])
    col_sec = _find_col(df, ["seccion", "sec"], contains=["secc"])
    col_prof = _find_col(df, ["profesor", "docente"], contains=["prof", "docent"])
    col_hor = _find_col(df, ["horario_raw", "horario"], contains=["horario", "hora"])
    col_cupo = _find_col(df, ["cupo"], contains=["cupo", "capacidad"])
    col_disp = _find_col(df, ["disponibles", "disponible", "dis"], contains=["dispon"])
    col_ciclo = _find_col(df, ["ciclo"], contains=["ciclo", "periodo"])
    col_centro = _find_col(df, ["centro"], contains=["centro", "campus"])
    col_inicio = _find_col(df, ["hora_inicio", "inicio"], contains=["hora_inicio", "inicio"])
    col_fin = _find_col(df, ["hora_fin", "fin"], contains=["hora_fin", "fin"])
    col_dias = _find_col(df, ["dias", "dia"], contains=["dia", "dias"])

    missing = []
    for name, col in [
        ("clave_materia", col_clave),
        ("horario_raw", col_hor),
        ("profesor", col_prof),
    ]:
        if not col:
            missing.append(name)
    if missing:
        raise ValueError(
            "Faltan columnas críticas en la oferta. "
            f"No se pudieron inferir: {', '.join(missing)}. "
            "Revisa el dataset o define env vars OFFER_DATA_PARQUET/OFFER_DATA_CSV con un esquema compatible."
        )

    out = pd.DataFrame()
    if col_carrera:
        out["carrera_clave"] = df[col_carrera].map(_normalize_carrera)
    else:
        out["carrera_clave"] = ""
    out["clave_materia"] = df[col_clave].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    out["materia"] = df[col_mat].astype(str) if col_mat else ""
    out["nrc"] = df[col_nrc].astype(str) if col_nrc else ""
    out["seccion"] = df[col_sec].astype(str) if col_sec else ""
    out["profesor"] = df[col_prof].astype(str)
    out["horario_raw"] = df[col_hor].astype(str)
    out["hora_inicio"] = df[col_inicio].astype(str) if col_inicio else ""
    out["hora_fin"] = df[col_fin].astype(str) if col_fin else ""
    out["dias"] = df[col_dias].astype(str) if col_dias else ""
    out["cupo"] = _optional_int_series(df[col_cupo]) if col_cupo else None
    out["disponibles"] = _optional_int_series(df[col_disp]) if col_disp else None
    out["ciclo"] = df[col_ciclo].astype(str) if col_ciclo else ""
    out["centro"] = df[col_centro].astype(str) if col_centro else ""
    if "semestre_estimado" in df.columns:
        out["semestre_estimado"] = pd.to_numeric(df["semestre_estimado"], errors="coerce").fillna(0).astype(int)
    else:
        out["semestre_estimado"] = 0

    return out
