export const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"

export type MetaResponse = {
  cycle: string | null
  source: string
  offer_rows: number
  professors: number
  plan_loaded: boolean
  last_updated: string | null
  availability_quality: "unknown" | "available"
  offer_avail_known_pct: number
  offer_avail_nonzero_pct: number
}

export type PlanMateria = {
  clave_materia: string
  materia: string
}

export type PlanSemestre = {
  semestre: number
  materias: PlanMateria[]
}

export type PlanResponse = {
  carrera: string
  semestres: PlanSemestre[]
}

export type GenerateRequest = {
  carrera_clave: "INFO" | "ICOM"
  semestre: number
  pref_horario: "manana" | "tarde" | "mixto"
  barco_exigente: number
  aprender_pasar: number
  solo_con_cupo: boolean
}

export type ScheduleStats = {
  days_count: number
  gaps_minutes: number
  pref_match_pct: number
  reviews_coverage_pct: number
}

export type ScheduleClass = {
  clave_materia: string
  materia: string
  nrc: string
  seccion: string
  profesor: string
  horario_raw: string
  dias: string[]
  inicio: number
  fin: number
  meetings?: ScheduleMeeting[]
  periodo?: string | null
  ubicacion?: string | null
  cupo: number | null
  disponibles: number | null
  tags: Record<string, number>
  profile_confianza: number
}

export type ScheduleMeeting = {
  day: string
  start_min: number
  end_min: number
}

export type Schedule = {
  schedule_id: string
  score: number
  stats: ScheduleStats
  breakdown_bullets: string[]
  classes: ScheduleClass[]
}

export type GenerateResponse = {
  schedules: Schedule[]
  availability_quality: "unknown" | "available"
  offer_avail_nonzero_pct: number
  diagnostic_bullets: string[]
}

export type ReviewBullet = {
  claim: string
  evidence_quote: string
  evidence_truncated: boolean
}

export type ReviewItem = {
  text: string
  raw: string
  contexto_materia?: string | null
}

export type ReviewsResponse = {
  profesor: string
  normalized: string
  match_confidence: number
  tags: Record<string, number>
  bullets: ReviewBullet[]
  summary_generated?: ReviewSummary | null
  reviews_total_count: number
  reviews_unique_count: number
  reviews: ReviewItem[]
  reviews_originales?: ReviewItem[] | null
}

export type ReviewSummary = {
  summary_text: string
  pros?: string[]
  contras?: string[]
  summary?: string
  cons?: string[]
  confidence: number
  tags: Record<string, number>
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(text || "Error de red")
  }
  return resp.json()
}

export function fetchMeta() {
  return http<MetaResponse>("/api/meta")
}

export function fetchPlan(carrera: string) {
  return http<PlanResponse>(`/api/plan?carrera=${encodeURIComponent(carrera)}`)
}

export function generateSchedules(payload: GenerateRequest) {
  return http<GenerateResponse>("/api/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export function fetchReviews(
  profesor: string,
  options: { limit?: number; includeRaw?: boolean } = {}
) {
  const limit = options.limit ?? 20
  const includeRaw = options.includeRaw ?? false
  return http<ReviewsResponse>(
    `/api/reviews?profesor=${encodeURIComponent(profesor)}&limit=${limit}&include_raw=${includeRaw}`
  )
}

export function schedulePngUrl(scheduleId: string) {
  return `${API_BASE}/api/schedule/${scheduleId}/png`
}
