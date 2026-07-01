from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from .data_access import get_offer, get_plan, get_reviews
from .groq_client import build_summary_generated, build_summary_long, summarize_reviews
from .models import GenerateRequest, GenerateResponse, HealthResponse, MetaResponse, PlanResponse, ReviewsResponse
from .plan_loader import get_semester_keys, get_semester_materias, semestres_disponibles
from .professor_match import load_aliases, match_professor, normalize_professor_name
from .reviews_features import compute_features, dedupe_and_clean_reviews, extract_tags
from .scheduler import build_top_schedules
from .settings import Settings

router = APIRouter(prefix="/api")
settings = Settings()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    offer_df, offer_meta = get_offer(settings)
    reviews_df = get_reviews(settings)
    plan_df = get_plan(settings)

    cycle = offer_meta.get("cycle") or ""
    if not cycle:
        source = offer_meta.get("source", "")
        digits = "".join([ch for ch in source if ch.isdigit()])
        cycle = digits or ""

    last_updated = None
    try:
        source = Path(offer_meta.get("source", ""))
        if source.exists():
            last_updated = datetime.fromtimestamp(source.stat().st_mtime).isoformat()
    except Exception:
        last_updated = None

    return MetaResponse(
        cycle=cycle or None,
        source=offer_meta.get("source", ""),
        offer_rows=int(offer_meta.get("offer_rows", len(offer_df))),
        professors=int(reviews_df["norm_name"].nunique()) if not reviews_df.empty else 0,
        plan_loaded=not plan_df.empty,
        last_updated=last_updated,
        availability_quality=str(offer_meta.get("availability_quality", "available")),
        offer_avail_known_pct=float(offer_meta.get("offer_avail_known_pct", 0.0)),
        offer_avail_nonzero_pct=float(offer_meta.get("offer_avail_nonzero_pct", 0.0)),
    )


@router.get("/plan", response_model=PlanResponse)
def plan(carrera: str = Query("INFO")) -> PlanResponse:
    plan_df = get_plan(settings)
    if plan_df.empty:
        raise HTTPException(status_code=404, detail="Plan no cargado")

    semestres = []
    for semestre in semestres_disponibles(plan_df, carrera):
        materias = get_semester_materias(plan_df, carrera, semestre, group="CORE")
        semestres.append({"semestre": semestre, "materias": materias})

    return PlanResponse(carrera=carrera.upper(), semestres=semestres)


@router.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    offer_df, offer_meta = get_offer(settings)
    reviews_df = get_reviews(settings)
    plan_df = get_plan(settings)

    allowed = set(get_semester_keys(plan_df, payload.carrera_clave, payload.semestre, group="CORE"))
    if not allowed:
        raise HTTPException(status_code=404, detail="No se encontraron materias CORE para ese semestre")

    aliases = load_aliases(settings.aliases_path)
    schedules = build_top_schedules(
        offer_df,
        reviews_df,
        aliases,
        payload.carrera_clave,
        payload.semestre,
        payload.pref_horario,
        payload.barco_exigente,
        payload.aprender_pasar,
        max_schedules=3,
        allowed_claves=allowed,
        only_available=payload.solo_con_cupo,
    )

    availability_quality = str(offer_meta.get("availability_quality", "available"))
    offer_avail_nonzero_pct = float(offer_meta.get("offer_avail_nonzero_pct", 0.0))
    diagnostic_bullets = []
    if payload.solo_con_cupo and availability_quality == "unknown":
        diagnostic_bullets.append(
            "No es posible filtrar por cupo: SIIAU no proporciono disponibles/cupo en este ciclo."
        )
    if not schedules:
        diagnostic_bullets.append(
            "No existe combinacion sin traslapes para este semestre con la oferta actual. "
            "Prueba ajustar filtros o preferencia."
        )

    result = []
    for schedule in schedules:
        schedule_id = uuid4().hex[:8]
        schedule["schedule_id"] = schedule_id
        result.append(schedule)

    return GenerateResponse(
        schedules=result,
        availability_quality=availability_quality,
        offer_avail_nonzero_pct=offer_avail_nonzero_pct,
        diagnostic_bullets=diagnostic_bullets,
    )


@router.get("/reviews", response_model=ReviewsResponse)
def reviews(
    profesor: str,
    limit: int = Query(20, ge=1, le=50),
    include_raw: bool = Query(False),
) -> ReviewsResponse:
    reviews_df = get_reviews(settings)
    if reviews_df.empty:
        summary_generated = build_summary_generated(
            [],
            {"total": 0, "comments": []},
            {},
            cache_key=normalize_professor_name(profesor),
            cache_dir=settings.cache_dir,
            require_groq=settings.require_groq_summary,
        )
        summary_long = build_summary_long(
            [],
            {"total": 0, "comments": []},
            profesor,
            cache_key=normalize_professor_name(profesor),
            cache_dir=settings.cache_dir,
            ttl_seconds=settings.reviews_summary_long_ttl_hours * 3600,
            require_groq=settings.require_groq_summary,
        )
        return ReviewsResponse(
            profesor=profesor,
            normalized=normalize_professor_name(profesor),
            match_confidence=0,
            tags={},
            bullets=[{"claim": "Sin reseñas disponibles.", "evidence_quote": "", "evidence_truncated": False}],
            summary_generated=summary_generated,
            summary_long=summary_long.get("summary_long", ""),
            reviews_total_count=0,
            reviews_unique_count=0,
            reviews=[],
            reviews_originales=None,
        )

    aliases = load_aliases(settings.aliases_path)
    reviews_names = reviews_df["norm_name"].unique().tolist()
    matched, confidence = match_professor(profesor, reviews_names, aliases)
    subset = reviews_df[reviews_df["norm_name"] == matched]
    features = compute_features(subset)
    tags = extract_tags(features)
    summary_generated = build_summary_generated(
        features.get("comments", []),
        features,
        tags,
        cache_key=matched,
        cache_dir=settings.cache_dir,
        require_groq=settings.require_groq_summary,
    )
    summary_long = build_summary_long(
        features.get("comments", []),
        features,
        matched,
        cache_key=matched,
        cache_dir=settings.cache_dir,
        ttl_seconds=settings.reviews_summary_long_ttl_hours * 3600,
        require_groq=settings.require_groq_summary,
    )
    bullets_raw = summarize_reviews(features.get("comments", [])[:20], features, use_groq=False)
    reviews_clean, total_count, unique_count = dedupe_and_clean_reviews(features.get("comments", []))
    reviews_list = reviews_clean[:limit] if include_raw else []

    bullets = []
    for bullet in bullets_raw:
        claim = str(bullet.get("claim", "")).strip()
        evidence = str(bullet.get("evidence_quote", "")).strip()
        truncated = False
        max_len = settings.evidence_quote_max_len
        if max_len and len(evidence) > max_len:
            evidence = evidence[:max_len].rstrip() + "..."
            truncated = True
        bullets.append(
            {
                "claim": claim,
                "evidence_quote": evidence,
                "evidence_truncated": truncated,
            }
        )

    return ReviewsResponse(
        profesor=profesor,
        normalized=matched,
        match_confidence=confidence,
        tags=tags,
        bullets=bullets,
        summary_generated=summary_generated,
        summary_long=summary_long.get("summary_long", ""),
        reviews_total_count=total_count,
        reviews_unique_count=unique_count,
        reviews=reviews_list,
        reviews_originales=reviews_list if include_raw else None,
    )
