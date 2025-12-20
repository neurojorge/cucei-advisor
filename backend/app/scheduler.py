from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .professor_match import normalize_professor_name, match_professor
from .reviews_features import compute_features, extract_tags, score_professor

DAY_MAP = {"L": 0, "M": 1, "I": 2, "J": 3, "V": 4, "S": 5, "D": 6}
DAY_NAMES = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]


@dataclass
class Section:
    materia: str
    clave: str
    nrc: str
    seccion: str
    profesor: str
    cupo: int | None
    disponibles: int | None
    carrera: str
    semestre: int
    slots: List[Tuple[List[int], int, int]]
    raw: str
    profile_score: float = 0.0
    profile_confianza: int = 0
    tags: Dict[str, int] = field(default_factory=dict)
    match_confidence: int = 0

    @property
    def dias(self) -> List[int]:
        return sorted({d for days, _, _ in self.slots for d in days})

    @property
    def inicio(self) -> int:
        return min([s[1] for s in self.slots]) if self.slots else 0

    @property
    def fin(self) -> int:
        return max([s[2] for s in self.slots]) if self.slots else 0


def parse_time_component(value: str) -> int:
    if value is None:
        return 0
    digits = "".join([ch for ch in str(value) if ch.isdigit()])
    if len(digits) == 3:
        h = int(digits[0])
        m = int(digits[1:])
    elif len(digits) >= 4:
        h = int(digits[:2])
        m = int(digits[2:4])
    else:
        return 0
    if h >= 24 or m >= 60:
        return 0
    return h * 60 + m


def _to_optional_int(value) -> int | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def parse_days(value: str) -> List[int]:
    if not value:
        return []
    clean = str(value).upper()
    days = [DAY_MAP[ch] for ch in clean if ch in DAY_MAP]
    return sorted(set(days))


def _split_parts(value: str) -> List[str]:
    text = str(value or "")
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text.strip()] if text.strip() else []


def parse_slots_from_horario_raw(raw: str) -> List[Tuple[List[int], int, int]]:
    import re

    text = str(raw or "")
    matches = list(re.finditer(r"(\d{3,4})\s*-\s*(\d{3,4})", text))
    slots: List[Tuple[List[int], int, int]] = []
    if not matches:
        return slots
    for idx, match in enumerate(matches):
        start = parse_time_component(match.group(1))
        end = parse_time_component(match.group(2))
        seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[match.end() : seg_end]
        days = parse_days(segment)
        if not days:
            days = parse_days(text)
        if days and start and end:
            slots.append((days, start, end))
    return slots


def parse_timeslots(row: Dict) -> List[Tuple[List[int], int, int]]:
    dias_raw = row.get("dias", "")
    inicio_raw = row.get("hora_inicio", "")
    fin_raw = row.get("hora_fin", "")

    dias_parts = _split_parts(dias_raw)
    ini_parts = _split_parts(inicio_raw)
    fin_parts = _split_parts(fin_raw)

    slots: List[Tuple[List[int], int, int]] = []
    max_len = max(len(dias_parts), len(ini_parts), len(fin_parts), 0)

    for idx in range(max_len):
        dias_str = dias_parts[idx] if idx < len(dias_parts) else (dias_parts[-1] if dias_parts else "")
        ini_str = ini_parts[idx] if idx < len(ini_parts) else (ini_parts[-1] if ini_parts else "")
        fin_str = fin_parts[idx] if idx < len(fin_parts) else (fin_parts[-1] if fin_parts else "")
        dias = parse_days(dias_str)
        ini = parse_time_component(ini_str)
        fin = parse_time_component(fin_str)
        if dias and ini and fin:
            slots.append((dias, ini, fin))

    if not slots:
        slots = parse_slots_from_horario_raw(row.get("horario_raw", ""))
    return slots


def sections_conflict(a: Section, b: Section) -> bool:
    for days_a, start_a, end_a in a.slots:
        for days_b, start_b, end_b in b.slots:
            if not set(days_a) & set(days_b):
                continue
            if not (end_a <= start_b or end_b <= start_a):
                return True
    return False


def _schedule_conflicts(schedule: List[Section], candidate: Section) -> bool:
    return any(sections_conflict(existing, candidate) for existing in schedule)


def schedule_has_conflicts(sections: List[Section]) -> bool:
    for idx, section in enumerate(sections):
        for other in sections[idx + 1 :]:
            if sections_conflict(section, other):
                return True
    return False


def build_meetings(slots: List[Tuple[List[int], int, int]]) -> List[Dict[str, int | str]]:
    meetings: List[Dict[str, int | str]] = []
    for days, start, end in slots:
        for day in sorted(set(days)):
            if 0 <= day < len(DAY_NAMES):
                meetings.append({"day": DAY_NAMES[day], "start_min": start, "end_min": end})
    meetings.sort(key=lambda item: (DAY_NAMES.index(item["day"]), item["start_min"]))
    return meetings


def _gap_penalty_hours(sections: List[Section]) -> float:
    penalty = 0.0
    by_day: Dict[int, List[Tuple[int, int]]] = {}
    for s in sections:
        for days, start, end in s.slots:
            for d in days:
                by_day.setdefault(d, []).append((start, end))
    for intervals in by_day.values():
        intervals.sort()
        for i in range(1, len(intervals)):
            gap = intervals[i][0] - intervals[i - 1][1]
            if gap > 0:
                penalty += gap / 60.0
    return penalty


def _time_pref_score(sections: List[Section], pref: str) -> float:
    if not sections:
        return 0.0
    starts = [s.inicio for s in sections if s.inicio]
    if not starts:
        return 0.0
    avg_start = sum(starts) / len(starts)
    if pref == "manana":
        return -max(0, (avg_start - 12 * 60) / 60)
    if pref == "tarde":
        return -max(0, (9 * 60 - avg_start) / 60)
    return 0.0


def _pref_match_pct(sections: List[Section], pref: str) -> float:
    if not sections:
        return 0.0
    if pref == "mixto":
        return 100.0
    count = 0
    for s in sections:
        start = s.inicio
        if pref == "manana" and start and start < 12 * 60:
            count += 1
        if pref == "tarde" and start and start >= 12 * 60:
            count += 1
    return round(100 * count / len(sections), 1)


def _prof_score(sections: List[Section]) -> float:
    scores = [s.profile_score for s in sections]
    return float(np.mean(scores)) if scores else 0.0


def _score_schedule(sections: List[Section], pref_time: str) -> Tuple[float, Dict]:
    prof_score = _prof_score(sections)
    time_score = _time_pref_score(sections, pref_time)
    gaps = _gap_penalty_hours(sections)
    days = len({d for s in sections for d in s.dias})
    score = prof_score + time_score - gaps - 0.1 * days
    breakdown = {
        "profesores": round(prof_score, 3),
        "tiempo": round(time_score, 3),
        "huecos_horas": round(gaps, 3),
        "dias": days,
    }
    return score, breakdown


def _build_candidates(
    df: pd.DataFrame,
    career: str,
    semester: int,
    reviews_df: pd.DataFrame,
    barco_exigente: float,
    aprender_pasar: float,
    aliases: Dict[str, str],
    allowed_claves: Optional[set] = None,
    only_available: bool = False,
    max_sections_per_subject: int = 12,
) -> Dict[str, List[Section]]:
    career_upper = career.upper()
    filtered = df[df["carrera_clave"].str.upper().fillna("") == career_upper]
    if only_available and "disponibles" in filtered.columns:
        disp_num = pd.to_numeric(filtered["disponibles"], errors="coerce")
        filtered = filtered[disp_num.notna() & (disp_num > 0)]
    if allowed_claves:
        filtered = filtered[filtered["clave_materia"].isin(allowed_claves)]
    if filtered.empty:
        return {}

    reviews_names = reviews_df["norm_name"].unique().tolist() if not reviews_df.empty else []
    candidates: Dict[str, List[Section]] = {}

    for _, row in filtered.iterrows():
        slots = parse_timeslots(row.to_dict())
        if not slots:
            continue
        prof_norm = normalize_professor_name(row.get("profesor", ""))
        match_name, match_conf = match_professor(prof_norm, reviews_names, aliases)
        prof_reviews = reviews_df[reviews_df["norm_name"] == match_name] if not reviews_df.empty else pd.DataFrame()
        features = compute_features(prof_reviews)
        prof_score = score_professor(features, barco_exigente, aprender_pasar)
        tags = extract_tags(features)

        section = Section(
            materia=row.get("materia", ""),
            clave=row.get("clave_materia", ""),
            nrc=str(row.get("nrc", "")),
            seccion=str(row.get("seccion", "")),
            profesor=row.get("profesor", ""),
            cupo=_to_optional_int(row.get("cupo")),
            disponibles=_to_optional_int(row.get("disponibles")),
            carrera=row.get("carrera_clave", ""),
            semestre=int(row.get("semestre_estimado") or semester or 0),
            slots=slots,
            raw=row.get("horario_raw", ""),
            profile_score=prof_score,
            profile_confianza=features.get("total", 0),
            tags=tags,
            match_confidence=match_conf,
        )
        key = section.clave or section.materia
        candidates.setdefault(key, []).append(section)

    for key, sections in candidates.items():
        sections.sort(
            key=lambda s: (
                s.disponibles if s.disponibles is not None else -1,
                s.profile_score,
                -s.inicio,
            ),
            reverse=True,
        )
        candidates[key] = sections[:max_sections_per_subject]

    return candidates


def build_top_schedules(
    df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    aliases: Dict[str, str],
    career: str,
    semester: int,
    time_pref: str,
    barco_exigente: float,
    aprender_pasar: float,
    max_schedules: int = 3,
    allowed_claves: Optional[set] = None,
    only_available: bool = False,
) -> List[Dict]:
    candidates = _build_candidates(
        df,
        career,
        semester,
        reviews_df,
        barco_exigente,
        aprender_pasar,
        aliases,
        allowed_claves=allowed_claves,
        only_available=only_available,
    )
    if not candidates:
        return []

    subject_keys = list(candidates.keys())
    subject_keys.sort(key=lambda key: len(candidates[key]))

    schedules: List[Tuple[List[Section], float, Dict]] = []
    max_combinations = 20000
    explored = 0

    def backtrack(idx: int, current: List[Section]):
        nonlocal explored
        if explored >= max_combinations:
            return
        if idx >= len(subject_keys):
            if schedule_has_conflicts(current):
                explored += 1
                return
            score, breakdown = _score_schedule(current, time_pref)
            schedules.append((current.copy(), score, breakdown))
            explored += 1
            return
        subject = subject_keys[idx]
        for section in candidates.get(subject, []):
            if _schedule_conflicts(current, section):
                continue
            current.append(section)
            backtrack(idx + 1, current)
            current.pop()

    backtrack(0, [])
    if not schedules:
        return []

    schedules.sort(key=lambda tup: tup[1], reverse=True)
    top = schedules[:max_schedules]

    result = []
    for sections, score, breakdown in top:
        if schedule_has_conflicts(sections):
            continue
        gaps_minutes = int(round(_gap_penalty_hours(sections) * 60))
        days_count = len({d for s in sections for d in s.dias})
        pref_match_pct = _pref_match_pct(sections, time_pref)
        reviews_coverage = (
            round(100 * len([s for s in sections if s.profile_confianza > 0]) / len(sections), 1)
            if sections
            else 0.0
        )

        classes = []
        for s in sections:
            classes.append(
                {
                    "materia": s.materia,
                    "clave_materia": s.clave,
                    "nrc": s.nrc,
                    "seccion": s.seccion,
                    "profesor": s.profesor,
                    "dias": [DAY_NAMES[d] for d in s.dias],
                    "inicio": s.inicio,
                    "fin": s.fin,
                    "meetings": build_meetings(s.slots),
                    "cupo": s.cupo,
                    "disponibles": s.disponibles,
                    "horario_raw": s.raw,
                    "profile_confianza": s.profile_confianza,
                    "tags": s.tags,
                }
            )

        breakdown_bullets = [
            (
                f"Preferencia {time_pref} cubierta en {pref_match_pct}% de clases"
                if time_pref != "mixto"
                else "Distribucion mixta equilibrada"
            ),
            f"{days_count} dias con clases y {gaps_minutes} min de huecos",
            f"{reviews_coverage}% de materias con reseñas",
        ]

        result.append(
            {
                "classes": classes,
                "score": round(score, 3),
                "stats": {
                    "days_count": days_count,
                    "gaps_minutes": gaps_minutes,
                    "pref_match_pct": pref_match_pct,
                    "reviews_coverage_pct": reviews_coverage,
                },
                "breakdown_bullets": breakdown_bullets,
                "breakdown": breakdown,
            }
        )
    return result
