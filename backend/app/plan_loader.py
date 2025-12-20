from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _clean_plan_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["carrera_clave"] = df["carrera_clave"].astype(str).str.strip().str.upper()
    df["clave_materia"] = df["clave_materia"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    df["materia"] = df.get("materia", "").astype(str)
    df["grupo"] = df.get("grupo", "CORE").astype(str).str.strip().str.upper()
    df["semestre"] = df["semestre"].astype(str).str.strip()
    df = df[df["semestre"].str.upper() != "MODULO"]
    df["semestre"] = pd.to_numeric(df["semestre"], errors="coerce").fillna(0).astype(int)
    df = df[df["semestre"] > 0]
    return df


def load_plan_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return _clean_plan_df(path)


def build_plan_map(df: pd.DataFrame) -> Dict[str, Dict[int, List[dict]]]:
    plan: Dict[str, Dict[int, List[dict]]] = {}
    for _, row in df.iterrows():
        carrera = row["carrera_clave"]
        semestre = int(row["semestre"])
        plan.setdefault(carrera, {}).setdefault(semestre, []).append(
            {
                "clave_materia": row["clave_materia"],
                "materia": row.get("materia", ""),
                "grupo": row.get("grupo", "CORE"),
            }
        )
    return plan


def get_semester_keys(df: pd.DataFrame, carrera: str, semestre: int, group: str = "CORE") -> List[str]:
    if df.empty:
        return []
    group = group.upper()
    subset = df[
        (df["carrera_clave"] == carrera.upper())
        & (df["semestre"] == int(semestre))
        & (df["grupo"].str.upper() == group)
    ]
    return sorted(subset["clave_materia"].unique().tolist())


def get_semester_materias(df: pd.DataFrame, carrera: str, semestre: int, group: str = "CORE") -> List[dict]:
    if df.empty:
        return []
    group = group.upper()
    subset = df[
        (df["carrera_clave"] == carrera.upper())
        & (df["semestre"] == int(semestre))
        & (df["grupo"].str.upper() == group)
    ]
    materias = (
        subset[["clave_materia", "materia"]]
        .drop_duplicates()
        .sort_values(["clave_materia", "materia"])
        .to_dict("records")
    )
    return materias


def semestres_disponibles(df: pd.DataFrame, carrera: str) -> List[int]:
    if df.empty:
        return []
    return sorted(df[df["carrera_clave"] == carrera.upper()]["semestre"].unique().tolist())
