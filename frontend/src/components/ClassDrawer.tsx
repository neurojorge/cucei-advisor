import { useEffect, useMemo, useState } from "react"

import { fetchReviews, type ReviewItem, type ReviewsResponse, type ScheduleClass } from "../api/client"
import { cn } from "../lib/utils"
import { Badge, badgeVariants } from "./ui/badge"
import { Button } from "./ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "./ui/collapsible"
import { ScrollArea } from "./ui/scroll-area"
import { Separator } from "./ui/separator"
import { Sheet, SheetClose, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "./ui/sheet"
import { Skeleton } from "./ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs"

const TAG_ORDER = ["barco", "exigente", "claridad", "tareas", "examenes"]
const TAG_KEYWORDS: Record<string, string[]> = {
  barco: ["barco", "facil", "flexible"],
  exigente: ["exigente", "estricto", "pesado"],
  claridad: ["claro", "claridad", "explica"],
  tareas: ["tarea", "tareas", "trabajo"],
  examenes: ["examen", "examenes", "prueba"],
}
const DAY_ORDER = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab"] as const

type DayKey = (typeof DAY_ORDER)[number]

type ClassDrawerProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  selectedClass: ScheduleClass | null
}

function ClassDrawer({ open, onOpenChange, selectedClass }: ClassDrawerProps) {
  const [reviews, setReviews] = useState<ReviewsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [originals, setOriginals] = useState<ReviewItem[]>([])
  const [originalsOpen, setOriginalsOpen] = useState(false)
  const [originalsLoading, setOriginalsLoading] = useState(false)
  const [activeTag, setActiveTag] = useState<string | null>(null)
  const [openReviewIndex, setOpenReviewIndex] = useState<number | null>(null)
  const [pointsOpen, setPointsOpen] = useState(false)

  useEffect(() => {
    if (!open || !selectedClass) return
    setReviews(null)
    setLoading(true)
    setError(null)
    setOriginals([])
    setOriginalsOpen(false)
    setOriginalsLoading(false)
    setActiveTag(null)
    setOpenReviewIndex(null)
    setPointsOpen(false)
    fetchReviews(selectedClass.profesor, { limit: 20 })
      .then((data) => setReviews(data))
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [open, selectedClass])

  const summary = reviews?.summary_generated
  const summaryTags = summary?.tags ?? reviews?.tags ?? {}
  const summaryTagKeys = TAG_ORDER.filter((tag) => (summaryTags[tag] ?? 0) > 0)
  const summaryText = summary?.summary_text || summary?.summary || ""
  const summaryPros = summary?.pros ?? []
  const summaryContras = summary?.contras ?? summary?.cons ?? []

  const cupoLabel = formatCupo(selectedClass?.cupo ?? null, selectedClass?.disponibles ?? null)
  const meetings = useMemo(() => buildMeetings(selectedClass), [selectedClass])

  const filteredOriginals = useMemo(() => {
    if (!activeTag) return originals
    return originals.filter((review) => matchesTag(review, activeTag))
  }, [activeTag, originals])

  const handleToggleOriginals = async (nextOpen: boolean) => {
    setOriginalsOpen(nextOpen)
    if (!nextOpen || originals.length || !selectedClass) return
    setOriginalsLoading(true)
    try {
      const data = await fetchReviews(selectedClass.profesor, { limit: 40, includeRaw: true })
      setOriginals(data.reviews_originales ?? data.reviews ?? [])
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setOriginalsLoading(false)
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="space-y-6 overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{selectedClass?.materia || "Detalle"}</SheetTitle>
          <SheetDescription>
            {selectedClass?.clave_materia} - NRC {selectedClass?.nrc} - Seccion {selectedClass?.seccion}
          </SheetDescription>
          <div className="text-sm text-ink/70">{selectedClass?.profesor}</div>
        </SheetHeader>

        <Tabs defaultValue="detalle" className="space-y-4">
          <TabsList>
            <TabsTrigger value="detalle">Detalle</TabsTrigger>
            <TabsTrigger value="reseñas">Reseñas</TabsTrigger>
          </TabsList>

          <TabsContent value="detalle" className="space-y-4">
            <div className="grid gap-3 text-sm">
              {[
                { label: "NRC", value: selectedClass?.nrc || "—" },
                { label: "Seccion", value: selectedClass?.seccion || "—" },
                { label: "Profesor", value: selectedClass?.profesor || "—" },
                { label: "Cupo", value: cupoLabel },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between">
                  <span className="text-ink/60">{row.label}</span>
                  <span className="font-medium text-ink">{row.value}</span>
                </div>
              ))}
            </div>

            <div className="space-y-2">
              <div className="text-[11px] uppercase tracking-[0.2em] text-ink/40">Horarios</div>
              {meetings.length ? (
                <div className="flex flex-wrap gap-2">
                  {meetings.map((meeting, idx) => (
                    <div
                      key={`${meeting.day}-${meeting.start}-${idx}`}
                      className="flex items-center gap-2 rounded-full border border-line bg-white px-3 py-1 text-xs"
                    >
                      <Badge variant="ghost" className="text-[10px] uppercase tracking-[0.2em]">
                        {meeting.day}
                      </Badge>
                      <span className="text-ink/70">
                        {formatTime(meeting.start)} - {formatTime(meeting.end)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-ink/50">No disponible.</div>
              )}
            </div>

            {(selectedClass?.periodo || selectedClass?.ubicacion) && (
              <div className="space-y-1 text-xs text-ink/60">
                {selectedClass?.periodo && <div>Periodo: {selectedClass.periodo}</div>}
                {selectedClass?.ubicacion && <div>Ubicacion: {selectedClass.ubicacion}</div>}
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              {TAG_ORDER.map((tag) => (
                <Badge key={tag} variant={selectedClass?.tags?.[tag] ? "solid" : "ghost"}>
                  {tag}
                </Badge>
              ))}
              <Badge variant="default">confianza: {selectedClass?.profile_confianza ?? 0}</Badge>
            </div>
          </TabsContent>

          <TabsContent value="reseñas" className="space-y-4">
            {loading && (
              <div className="space-y-3">
                <Skeleton className="h-6 w-1/2" />
                <Skeleton className="h-24 w-full" />
              </div>
            )}
            {error && <div className="text-sm text-red-500">{error}</div>}
            {!loading && reviews && (
              <div className="space-y-4">
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="text-sm font-semibold">Resumen generado</h3>
                    {summary?.confidence !== undefined && summary && (
                      <Badge variant="default">confianza {summary.confidence}%</Badge>
                    )}
                  </div>
                  {summaryText ? (
                    <p className="text-sm leading-relaxed text-ink/70 whitespace-pre-line">{summaryText}</p>
                  ) : (
                    <div className="rounded-xl border border-line bg-fog p-3 text-xs text-ink/60">
                      Sin resumen disponible.
                    </div>
                  )}

                  {summaryTagKeys.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {summaryTagKeys.map((tag) => {
                        const count = summaryTags[tag] ?? 0
                        const isActive = activeTag === tag
                        return (
                          <button
                            key={tag}
                            type="button"
                            onClick={() => setActiveTag(isActive ? null : tag)}
                            className={cn(
                              badgeVariants({ variant: isActive ? "solid" : "ghost" }),
                              "cursor-pointer"
                            )}
                          >
                            {tag} {count}
                          </button>
                        )
                      })}
                      {activeTag && (
                        <button
                          type="button"
                          onClick={() => setActiveTag(null)}
                          className={cn(badgeVariants({ variant: "default" }), "cursor-pointer")}
                        >
                          limpiar
                        </button>
                      )}
                    </div>
                  )}

                  {(summaryPros.length || summaryContras.length) && (
                    <Collapsible open={pointsOpen} onOpenChange={setPointsOpen}>
                      <CollapsibleTrigger asChild>
                        <Button type="button" variant="outline" size="sm">
                          {pointsOpen ? "Ocultar puntos" : "Ver puntos"}
                        </Button>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="mt-3 grid gap-3 text-xs text-ink/70 md:grid-cols-2">
                          <div className="space-y-1">
                            <div className="text-[11px] uppercase tracking-[0.2em] text-ink/40">Pros</div>
                            <ul className="space-y-1">
                              {summaryPros.map((item, idx) => (
                                <li key={`${item}-${idx}`}>• {item}</li>
                              ))}
                            </ul>
                          </div>
                          <div className="space-y-1">
                            <div className="text-[11px] uppercase tracking-[0.2em] text-ink/40">Contras</div>
                            <ul className="space-y-1">
                              {summaryContras.map((item, idx) => (
                                <li key={`${item}-${idx}`}>• {item}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                </div>

                <Separator />

                <Collapsible open={originalsOpen} onOpenChange={handleToggleOriginals}>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="text-xs text-ink/50">
                      {reviews.reviews_unique_count} unicas / {reviews.reviews_total_count} totales
                    </div>
                    <CollapsibleTrigger asChild>
                      <Button type="button" variant="outline" size="sm">
                        {originalsOpen ? "Ocultar originales" : "Ver originales"}
                      </Button>
                    </CollapsibleTrigger>
                  </div>
                  <CollapsibleContent>
                    <div className="mt-3 space-y-3">
                      {originalsLoading && (
                        <div className="space-y-2">
                          <Skeleton className="h-4 w-1/2" />
                          <Skeleton className="h-20 w-full" />
                        </div>
                      )}
                      {!originalsLoading && filteredOriginals.length === 0 && (
                        <div className="rounded-xl border border-line bg-fog p-3 text-xs text-ink/60">
                          No hay reseñas originales disponibles.
                        </div>
                      )}
                      {!originalsLoading && filteredOriginals.length > 0 && (
                        <ScrollArea className="h-56 rounded-2xl border border-line bg-white/80 p-3">
                          <div className="space-y-3">
                            {filteredOriginals.map((review, idx) => (
                              <ReviewItemRow
                                key={`${review.raw}-${idx}`}
                                review={review}
                                open={openReviewIndex === idx}
                                onOpenChange={(next) => setOpenReviewIndex(next ? idx : null)}
                              />
                            ))}
                          </div>
                        </ScrollArea>
                      )}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            )}
          </TabsContent>
        </Tabs>

        <SheetClose asChild>
          <Button variant="outline">Cerrar</Button>
        </SheetClose>
      </SheetContent>
    </Sheet>
  )
}

export default ClassDrawer

function ReviewItemRow({
  review,
  open,
  onOpenChange,
}: {
  review: ReviewItem
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Collapsible open={open} onOpenChange={onOpenChange} className="rounded-xl border border-line bg-white p-3 text-xs">
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-1">
          {review.contexto_materia && (
            <Badge variant="ghost" className="text-[10px] uppercase tracking-[0.2em]">
              {review.contexto_materia}
            </Badge>
          )}
          <p className={open ? "text-ink/70" : "line-clamp-2 text-ink/70"}>{open ? review.raw : review.text}</p>
        </div>
        <CollapsibleTrigger asChild>
          <Button type="button" variant="ghost" size="sm">
            {open ? "Cerrar" : "Ver completa"}
          </Button>
        </CollapsibleTrigger>
      </div>
    </Collapsible>
  )
}

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

function buildMeetings(cls: ScheduleClass | null) {
  if (!cls) return []
  const rawMeetings = cls.meetings?.length
    ? cls.meetings.map((meeting) => ({
        day: meeting.day,
        start: meeting.start_min,
        end: meeting.end_min,
      }))
    : (cls.dias || []).map((day) => ({
        day,
        start: cls.inicio,
        end: cls.fin,
      }))

  return rawMeetings
    .map((meeting) => ({
      ...meeting,
      day: normalizeDay(meeting.day) ?? null,
    }))
    .filter((meeting): meeting is { day: DayKey; start: number; end: number } => Boolean(meeting.day))
    .filter((meeting) => meeting.start < meeting.end)
    .sort(
      (a, b) =>
        DAY_ORDER.indexOf(a.day) - DAY_ORDER.indexOf(b.day) || a.start - b.start
    )
}

function formatTime(minutes: number) {
  const safeMinutes = Math.max(0, minutes)
  const hours = Math.floor(safeMinutes / 60)
  const mins = safeMinutes % 60
  return `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}`
}

function formatCupo(cupo: number | null, disponibles: number | null) {
  if (cupo === null || disponibles === null) return "No disponible"
  if (cupo === 0 && disponibles === 0) return "No disponible"
  return `${disponibles} / ${cupo}`
}

function normalizeText(value: string) {
  return String(value || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
}

function matchesTag(review: ReviewItem, tag: string) {
  const keywords = TAG_KEYWORDS[tag] || [tag]
  const haystack = normalizeText(`${review.text} ${review.raw}`)
  return keywords.some((keyword) => haystack.includes(keyword))
}
