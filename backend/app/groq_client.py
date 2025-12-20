from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

SUMMARY_CACHE: Dict[str, Tuple[float, Dict[str, object]]] = {}
SUMMARY_TTL_SECONDS = 24 * 60 * 60


def _groq_prompt(reviews: List[str]) -> str:
    return (
        "Eres un asistente que resume reseñas reales de estudiantes. "
        "Devuelve JSON estricto con bullets y evidencia literal. "
        "No inventes datos. Si no hay evidencia, usa claim='No hay suficiente evidencia'.\n"
        '{"bullets":[{"claim":"...","evidence_quote":"..."}]}\n'
        "Usa solo las reseñas provistas. Evidence_quote debe ser una cita corta literal.\n"
        "Reseñas:\n- "
        + "\n- ".join(reviews[:20])
    )


def _groq_summary_prompt(reviews: List[str]) -> str:
    return (
        "Eres un asistente que resume reseñas reales de estudiantes. "
        "Devuelve JSON estricto con summary, pros, cons y confidence (0-100). "
        "summary debe tener maximo 80 palabras.\n"
        '{"summary":"...","pros":["..."],"cons":["..."],"confidence":80}\n'
        "Usa solo las reseñas provistas. No inventes datos.\n"
        "Reseñas:\n- "
        + "\n- ".join(reviews[:30])
    )


def _safe_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def _trim_words(text: str, limit: int) -> str:
    words = str(text or "").strip().split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip() + "..."


def _normalize_list(values: object, max_items: int = 5) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    for item in values:
        text = str(item or "").strip()
        if text:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _normalize_cache_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return key or "unknown"


def _read_summary_cache(cache_dir: str | None, cache_key: str, ttl_seconds: int) -> Dict[str, object] | None:
    now = time.time()
    cached = SUMMARY_CACHE.get(cache_key)
    if cached and now - cached[0] < ttl_seconds:
        return cached[1]
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"reviews_summary_{cache_key}.json"
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        ts = float(payload.get("ts", 0))
        if now - ts > ttl_seconds:
            return None
        data = payload.get("data", {})
        if isinstance(data, dict):
            SUMMARY_CACHE[cache_key] = (ts, data)
            return data
    except Exception:
        return None
    return None


def _write_summary_cache(cache_dir: str | None, cache_key: str, data: Dict[str, object]) -> None:
    ts = time.time()
    SUMMARY_CACHE[cache_key] = (ts, data)
    if not cache_dir:
        return
    path = Path(cache_dir) / f"reviews_summary_{cache_key}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ts": ts, "data": data}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def summarize_reviews_groq(reviews: List[str]) -> List[Dict[str, str]]:
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
    if not api_key or not reviews:
        return []
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Genera un resumen conciso basado solo en el texto dado."},
                {"role": "user", "content": _groq_prompt(reviews)},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        bullets = data.get("bullets", [])
        valid = []
        for bullet in bullets:
            if isinstance(bullet, dict) and "claim" in bullet and "evidence_quote" in bullet:
                valid.append(
                    {
                        "claim": str(bullet["claim"]).strip() or "No hay suficiente evidencia",
                        "evidence_quote": str(bullet["evidence_quote"]).strip(),
                    }
                )
        return valid
    except Exception:
        return []


def summarize_reviews_with_groq(reviews: List[str]) -> Dict[str, object]:
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
    if not api_key or not reviews:
        return {}
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Genera un resumen conciso basado solo en el texto dado."},
                {"role": "user", "content": _groq_summary_prompt(reviews)},
            ],
            temperature=0.2,
            max_tokens=350,
        )
        content = response.choices[0].message.content
        data = _safe_json(content or "")
        if not data:
            return {}
        summary = _trim_words(data.get("summary", ""), 80)
        pros = _normalize_list(data.get("pros", []))
        cons = _normalize_list(data.get("cons", []))
        try:
            confidence = int(float(data.get("confidence", 0)))
        except Exception:
            confidence = 0
        confidence = max(0, min(100, confidence))
        if not summary:
            return {}
        return {
            "summary": summary,
            "pros": pros,
            "cons": cons,
            "confidence": confidence,
        }
    except Exception:
        return {}


def fallback_summary_generated(features: dict) -> Dict[str, object]:
    total = int(features.get("total", 0) or 0)
    if total <= 0:
        return {"summary": "Sin reseñas disponibles.", "pros": [], "cons": [], "confidence": 0}
    pros: List[str] = []
    cons: List[str] = []
    if features.get("claridad", 0):
        pros.append("Explicacion clara")
    if features.get("barco", 0) > features.get("exigente", 0):
        pros.append("Evaluacion flexible")
    if features.get("aprendizaje", 0):
        pros.append("Se aprende en clase")
    if features.get("exigente", 0):
        cons.append("Puede ser exigente")
    if features.get("tareas", 0):
        cons.append("Carga de tareas")
    if features.get("examenes", 0):
        cons.append("Examenes frecuentes")
    pros = pros[:3]
    cons = cons[:3]
    parts = [f"{total} reseñas analizadas."]
    if pros:
        parts.append("Pros: " + ", ".join(pros[:2]) + ".")
    if cons:
        parts.append("Contras: " + ", ".join(cons[:2]) + ".")
    summary = _trim_words(" ".join(parts), 80)
    confidence = min(100, max(20, total * 6))
    return {"summary": summary, "pros": pros, "cons": cons, "confidence": confidence}


def build_summary_generated(
    reviews: List[str],
    features: dict,
    tags: dict,
    cache_key: str,
    cache_dir: str | None = None,
    ttl_seconds: int = SUMMARY_TTL_SECONDS,
) -> Dict[str, object]:
    key = _normalize_cache_key(cache_key)
    cached = _read_summary_cache(cache_dir, key, ttl_seconds)
    if cached:
        cached["tags"] = tags
        return cached
    summary = summarize_reviews_with_groq(reviews)
    if not summary:
        summary = fallback_summary_generated(features)
    summary["tags"] = tags
    _write_summary_cache(cache_dir, key, summary)
    return summary


def fallback_summary(features: dict) -> List[Dict[str, str]]:
    if not features or features.get("total", 0) == 0:
        return [{"claim": "Sin reseñas disponibles.", "evidence_quote": ""}]
    comments = features.get("comments", [])[:6]
    bullets: List[Dict[str, str]] = []
    bullets.append(
        {
            "claim": f"Reseñas totales: {features.get('total', 0)}; barco={features.get('barco', 0)}; exigente={features.get('exigente', 0)}",
            "evidence_quote": comments[0] if comments else "",
        }
    )
    if features.get("aprendizaje", 0):
        bullets.append(
            {
                "claim": f"Enfasis en aprendizaje: {features['aprendizaje']} menciones",
                "evidence_quote": comments[1] if len(comments) > 1 else "",
            }
        )
    if features.get("claridad", 0):
        bullets.append(
            {
                "claim": f"Claridad/explicacion mencionada {features['claridad']} veces",
                "evidence_quote": comments[2] if len(comments) > 2 else "",
            }
        )
    while len(bullets) < 3 and comments:
        idx = min(len(bullets) + 2, len(comments) - 1)
        bullets.append({"claim": "Comentario destacado", "evidence_quote": comments[idx]})
    return bullets[:3]


def summarize_reviews(reviews: List[str], features: dict, use_groq: bool = True) -> List[Dict[str, str]]:
    if use_groq:
        bullets = summarize_reviews_groq(reviews)
        if bullets:
            return bullets
    return fallback_summary(features)
