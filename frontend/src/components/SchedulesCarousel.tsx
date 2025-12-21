import type { Schedule, ScheduleClass } from "../api/client"
import { cn } from "../lib/utils"
import { Badge } from "./ui/badge"
import { Button } from "./ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "./ui/card"

const DAY_ORDER = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab"] as const

type DayKey = (typeof DAY_ORDER)[number]

function normalizeDay(raw: string): DayKey | null {
  const cleaned = raw
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z]/g, "")

  if (!cleaned) return null
  if (cleaned.startsWith("lun")) return "Lun"
  if (cleaned.startsWith("mar")) return "Mar"
  if (cleaned.startsWith("mie")) return "Mie"
  if (cleaned.startsWith("jue")) return "Jue"
  if (cleaned.startsWith("vie")) return "Vie"
  if (cleaned.startsWith("sab")) return "Sab"
  return null
}

function normalizeDays(rawDays: string[]): DayKey[] {
  const found = new Set<DayKey>()
  rawDays.forEach((raw) => {
    raw
      .split(/[,\s/;-]+/)
      .map((token) => token.trim())
      .filter(Boolean)
      .forEach((token) => {
        const normalized = normalizeDay(token)
        if (normalized) found.add(normalized)
      })
  })
  return DAY_ORDER.filter((day) => found.has(day))
}

function formatTime(minutes: number) {
  const safeMinutes = Math.max(0, minutes)
  const hours = Math.floor(safeMinutes / 60)
  const mins = safeMinutes % 60
  return `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}`
}

function formatMeetingLabel(cls: ScheduleClass) {
  const meetings =
    cls.meetings
      ?.map((meeting) => {
        const day = normalizeDay(meeting.day)
        if (!day) return null
        if (meeting.start_min >= meeting.end_min) return null
        return `${day} ${formatTime(meeting.start_min)}-${formatTime(meeting.end_min)}`
      })
      .filter((item): item is string => Boolean(item)) || []

  if (meetings.length) {
    return meetings.join(" / ")
  }

  const days = normalizeDays(cls.dias || [])
  if (days.length && Number.isFinite(cls.inicio) && Number.isFinite(cls.fin) && cls.inicio < cls.fin) {
    return `${days.join(", ")} ${formatTime(cls.inicio)}-${formatTime(cls.fin)}`
  }

  return cls.horario_raw || "Horario por confirmar"
}

type SchedulesCarouselProps = {
  schedules: Schedule[]
  selectedId: string | null
  onSelect: (id: string) => void
}

function SchedulesCarousel({ schedules, selectedId, onSelect }: SchedulesCarouselProps) {
  if (!schedules.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Sin resultados</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-ink/60">
          No se encontraron combinaciones con los filtros actuales.
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="flex gap-4 overflow-x-auto pb-2">
      {schedules.map((schedule, index) => (
        <Card
          key={schedule.schedule_id}
          className={cn(
            "min-w-[280px] flex-1 text-left transition",
            selectedId === schedule.schedule_id && "border-ink shadow-soft"
          )}
        >
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Horario {index + 1}</CardTitle>
                <Badge variant={selectedId === schedule.schedule_id ? "solid" : "default"}>
                  score {schedule.score.toFixed(2)}
                </Badge>
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs text-ink/60">
                <Badge variant="ghost">{schedule.stats.days_count} dias</Badge>
                <Badge variant="ghost">{schedule.stats.gaps_minutes} min huecos</Badge>
                <Badge variant="ghost">{schedule.stats.pref_match_pct}% pref</Badge>
                <Badge variant="ghost">{schedule.stats.reviews_coverage_pct}% reseñas</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded-2xl border border-line bg-white p-3">
                <div className="text-[11px] uppercase tracking-[0.2em] text-ink/50">Vista rapida</div>
                <ul className="mt-2 space-y-2 text-xs text-ink/70">
                  {schedule.classes.slice(0, 3).map((cls) => (
                    <li key={`${cls.clave_materia}-${cls.nrc}`} className="space-y-1">
                      <div className="line-clamp-1 text-[13px] font-semibold text-ink">{cls.materia}</div>
                      <div className="line-clamp-2 text-[11px] text-ink/60">
                        {formatMeetingLabel(cls)}
                      </div>
                    </li>
                  ))}
                </ul>
                {schedule.classes.length > 3 && (
                  <div className="mt-2 text-[11px] text-ink/50">
                    +{schedule.classes.length - 3} materias mas
                  </div>
                )}
              </div>
              <ul className="space-y-1 text-xs text-ink/70">
                {schedule.breakdown_bullets.map((bullet) => (
                  <li key={bullet}>• {bullet}</li>
                ))}
              </ul>
            </CardContent>
            <CardFooter className="justify-end gap-2">
              <Button
                type="button"
                variant={selectedId === schedule.schedule_id ? "default" : "outline"}
                onClick={() => onSelect(schedule.schedule_id)}
              >
                Ver
              </Button>
            </CardFooter>
          </Card>
      ))}
    </div>
  )
}

export default SchedulesCarousel
