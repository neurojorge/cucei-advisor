from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd

from .professor_match import normalize_professor_name

KEYWORDS = {
    "barco": ["BARCO", "FLEXIBLE", "FACIL", "FÁCIL"],
    "exigente": ["EXIGENTE", "ESTRICTO", "PESADO"],
    "claridad": ["CLARA", "CLARIDAD", "EXPLICA", "ENTIENDE"],
    "tareas": ["TAREA", "TAREAS", "TRABAJO"],
    "examenes": ["EXAMEN", "EXAMENES", "PRUEBA"],
    "proyectos": ["PROYECTO", "PROYECTOS"],
    "justicia": ["JUSTO", "JUSTA", "JUSTICIA"],
    "aprendizaje": ["APRENDER", "APRENDI", "APRENDE"],
    "pasar": ["PASAR", "PASA", "CALIFICACION"],
}

TAG_KEYS = ["barco", "exigente", "claridad", "tareas", "examenes"]
SPACE_RE = re.compile(r"\s+")
CONTEXT_RE = re.compile(
    r"(?:[\"'“”])?\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\\s\\-]+?)\\s*\\(([A-Z]{1,3}\\d{3,4})\\)\\s*(?:[\"'“”])?$",
    re.IGNORECASE,
)


def load_reviews(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["PROFESOR"] = df["PROFESOR"].fillna("").astype(str)
    df["COMENTARIOS"] = df["COMENTARIOS"].fillna("").astype(str)
    df["norm_name"] = df["PROFESOR"].apply(normalize_professor_name)
    df["COMENTARIO_CLEAN"] = df["COMENTARIOS"].fillna("").astype(str)
    return df


def _keyword_score(texts: List[str], kw_list: List[str]) -> int:
    score = 0
    for text in texts:
        upper = text.upper()
        for kw in kw_list:
            if kw in upper:
                score += 1
    return score


def compute_features(reviews: pd.DataFrame) -> Dict:
    comments = reviews["COMENTARIO_CLEAN"].tolist() if not reviews.empty else []
    total = len(comments)
    feats = {"total": total, "comments": comments}
    for key, kws in KEYWORDS.items():
        feats[key] = _keyword_score(comments, kws)
    return feats


def normalize_review_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "").strip())


def extract_context_materia(text: str) -> Tuple[str, str | None]:
    raw = str(text or "").strip()
    match = CONTEXT_RE.search(raw)
    if not match:
        return normalize_review_text(raw), None
    materia = match.group(1).strip()
    clave = match.group(2).strip()
    contexto = f"{materia} ({clave})"
    cleaned = raw[: match.start()].rstrip()
    cleaned = normalize_review_text(cleaned)
    return cleaned, contexto


def dedupe_and_clean_reviews(comments: List[str]) -> Tuple[List[Dict], int, int]:
    total_count = len(comments)
    seen = set()
    reviews: List[Dict] = []
    for raw in comments:
        raw_text = str(raw or "")
        normalized = normalize_review_text(raw_text)
        if normalized in seen:
            continue
        seen.add(normalized)
        text, contexto = extract_context_materia(raw_text)
        item = {"text": text, "raw": raw_text}
        if contexto:
            item["contexto_materia"] = contexto
        reviews.append(item)
    unique_count = len(seen)
    return reviews, total_count, unique_count


def extract_tags(features: Dict) -> Dict[str, int]:
    return {key: int(features.get(key, 0)) for key in TAG_KEYS}


def score_professor(features: Dict, barco_exigente: float, aprender_pasar: float) -> float:
    if not features or features.get("total", 0) == 0:
        return 0.0
    total = features["total"]
    barco_ratio = features.get("barco", 0) / total
    exig_ratio = features.get("exigente", 0) / total
    aprender_ratio = features.get("aprendizaje", 0) / total
    pasar_ratio = features.get("pasar", 0) / total

    # slider 0..1: 0 = barco/aprender, 1 = exigente/pasar
    discipline_weight = 1 - 2 * barco_exigente
    goal_weight = 1 - 2 * aprender_pasar

    score = (barco_ratio - exig_ratio) * discipline_weight
    score += (aprender_ratio - pasar_ratio) * goal_weight
    return float(score)
