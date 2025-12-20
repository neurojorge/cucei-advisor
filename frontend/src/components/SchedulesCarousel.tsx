import { schedulePngUrl, type Schedule } from "../api/client"
import { cn } from "../lib/utils"
import ExportPngButton from "./ExportPngButton"
import { Badge } from "./ui/badge"
import { Button } from "./ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "./ui/card"

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
              <div className="h-40 overflow-hidden rounded-2xl border border-line bg-white">
                <img
                  src={schedulePngUrl(schedule.schedule_id)}
                  alt={`Horario ${index + 1}`}
                  className="h-full w-full object-cover"
                />
              </div>
              <ul className="space-y-1 text-xs text-ink/70">
                {schedule.breakdown_bullets.map((bullet) => (
                  <li key={bullet}>• {bullet}</li>
                ))}
              </ul>
            </CardContent>
            <CardFooter className="justify-between gap-2">
              <Button
                type="button"
                variant={selectedId === schedule.schedule_id ? "default" : "outline"}
                onClick={() => onSelect(schedule.schedule_id)}
              >
                Ver
              </Button>
              <ExportPngButton scheduleId={schedule.schedule_id} size="sm" />
            </CardFooter>
          </Card>
      ))}
    </div>
  )
}

export default SchedulesCarousel
