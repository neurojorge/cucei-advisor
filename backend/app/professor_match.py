from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from rapidfuzz import fuzz, process

ACCENT_MAP = str.maketrans(
    {
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
        "Ñ": "N",
        "á": "A",
        "é": "E",
        "í": "I",
        "ó": "O",
        "ú": "U",
        "ñ": "N",
    }
)


def normalize_professor_name(name: str) -> str:
    if not name:
        return ""
    clean = str(name).translate(ACCENT_MAP).upper()
    for ch in [",", ".", ";", ":", "-", "_", "/"]:
        clean = clean.replace(ch, " ")
    return " ".join(clean.split())


def load_aliases(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {normalize_professor_name(k): normalize_professor_name(v) for k, v in data.items()}
    except Exception:
        return {}


def match_professor(
    siiau_name: str,
    reviews_names: list[str],
    aliases: Dict[str, str],
    threshold: int = 90,
) -> Tuple[str, int]:
    norm = normalize_professor_name(siiau_name)
    if norm in aliases:
        return aliases[norm], 100
    match = process.extractOne(norm, reviews_names, scorer=fuzz.ratio)
    if match:
        score = int(match[1])
        if score >= threshold:
            return match[0], score
        return norm, score
    return norm, 0
