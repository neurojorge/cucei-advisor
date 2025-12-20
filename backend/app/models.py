from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class MetaResponse(BaseModel):
    cycle: str | None
    source: str
    offer_rows: int
    professors: int
    plan_loaded: bool
    last_updated: str | None
    availability_quality: str
    offer_avail_known_pct: float
    offer_avail_nonzero_pct: float


class PlanMateria(BaseModel):
    clave_materia: str
    materia: str


class PlanSemestre(BaseModel):
    semestre: int
    materias: List[PlanMateria]


class PlanResponse(BaseModel):
    carrera: str
    semestres: List[PlanSemestre]


class GenerateRequest(BaseModel):
    carrera_clave: Literal["INFO", "ICOM"]
    semestre: int
    pref_horario: Literal["manana", "tarde", "mixto"]
    barco_exigente: float = Field(..., ge=0.0, le=1.0)
    aprender_pasar: float = Field(..., ge=0.0, le=1.0)
    solo_con_cupo: bool = False


class ScheduleStats(BaseModel):
    days_count: int
    gaps_minutes: int
    pref_match_pct: float
    reviews_coverage_pct: float


class ScheduleMeeting(BaseModel):
    day: str
    start_min: int
    end_min: int


class ScheduleClass(BaseModel):
    clave_materia: str
    materia: str
    nrc: str
    seccion: str
    profesor: str
    horario_raw: str
    dias: List[str]
    inicio: int
    fin: int
    meetings: List[ScheduleMeeting] = Field(default_factory=list)
    cupo: int | None
    disponibles: int | None
    tags: dict
    profile_confianza: int


class ScheduleItem(BaseModel):
    schedule_id: str
    score: float
    stats: ScheduleStats
    breakdown_bullets: List[str]
    classes: List[ScheduleClass]


class GenerateResponse(BaseModel):
    schedules: List[ScheduleItem]
    availability_quality: str
    offer_avail_nonzero_pct: float
    diagnostic_bullets: List[str]


class ReviewBullet(BaseModel):
    claim: str
    evidence_quote: str
    evidence_truncated: bool


class ReviewItem(BaseModel):
    text: str
    raw: str
    contexto_materia: str | None = None


class ReviewSummary(BaseModel):
    summary_text: str = ""
    pros: List[str] = Field(default_factory=list)
    contras: List[str] = Field(default_factory=list)
    confidence: int = 0
    tags: dict = Field(default_factory=dict)
    summary: str | None = None
    cons: List[str] | None = None


class ReviewsResponse(BaseModel):
    profesor: str
    normalized: str
    match_confidence: int
    tags: dict
    bullets: List[ReviewBullet]
    summary_generated: ReviewSummary | None = None
    reviews_total_count: int
    reviews_unique_count: int
    reviews: List[ReviewItem]
    reviews_originales: List[ReviewItem] | None = None
