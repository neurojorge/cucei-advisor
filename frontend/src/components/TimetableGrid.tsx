import { useMemo } from "react"

import type { ScheduleClass } from "../api/client"
import { cn } from "../lib/utils"
import { Badge } from "./ui/badge"

const DAY_ORDER = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab"] as const

const MIN_HOUR = 7
const MAX_HOUR = 22
const STEP = 30

const rowCount = ((MAX_HOUR - MIN_HOUR) * 60) / STEP

type TimetableGridProps = {
  classes: ScheduleClass[]
  onSelect: (cls: ScheduleClass) => void
  selectedKey?: string | null
}

type DayKey = (typeof DAY_ORDER)[number]

type NormalizedMeeting = {
  cls: ScheduleClass
  day: DayKey
  start: number
  end: number
}

type PositionedMeeting = NormalizedMeeting & {
  lane: number
  laneCount: number
  conflict: boolean
  conflictWith?: string
}

function toRowStart(minutes: number) {
  const clamped = Math.max(minutes, MIN_HOUR * 60)
  return Math.floor((clamped - MIN_HOUR * 60) / STEP) + 2
}

function toRowSpan(start: number, end: number) {
  return Math.max(1, Math.round((end - start) / STEP))
}

function overlaps(aStart: number, aEnd: number, bStart: number, bEnd: number) {
  return !(aEnd <= bStart || bEnd <= aStart)
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

function buildMeetingsForClass(cls: ScheduleClass): NormalizedMeeting[] {
  if (cls.meetings?.length) {
    return cls.meetings
      .map((meeting) => ({
        cls,
        day: normalizeDay(meeting.day) ?? null,
        start: meeting.start_min,
        end: meeting.end_min,
      }))
      .filter(
        (meeting): meeting is NormalizedMeeting =>
          Boolean(meeting.day) && meeting.start !== undefined && meeting.end !== undefined
      )
      .filter((meeting) => meeting.start < meeting.end)
  }

  const days = normalizeDays(cls.dias || [])
  return days
    .map((day) => ({
      cls,
      day,
      start: cls.inicio,
      end: cls.fin,
    }))
    .filter((meeting) => meeting.start < meeting.end)
}

function buildPositions(meetings: NormalizedMeeting[]): PositionedMeeting[] {
  const positions: PositionedMeeting[] = []

  DAY_ORDER.forEach((day) => {
    const dayMeetings = meetings
      .filter((item) => item.day === day)
      .map((item) => ({
        ...item,
        lane: 0,
        laneCount: 1,
        conflict: false,
      }))
      .sort((a, b) => a.start - b.start || a.end - b.end)

    const lanes: number[] = []
    for (const item of dayMeetings) {
      let laneIndex = lanes.findIndex((end) => end <= item.start)
      if (laneIndex === -1) {
        laneIndex = lanes.length
        lanes.push(item.end)
      } else {
        lanes[laneIndex] = item.end
      }
      item.lane = laneIndex
    }

    for (const item of dayMeetings) {
      let maxLane = 1
      let conflictWith: string | undefined
      for (const other of dayMeetings) {
        if (overlaps(item.start, item.end, other.start, other.end)) {
          maxLane = Math.max(maxLane, other.lane + 1)
          if (!conflictWith && other.cls.nrc !== item.cls.nrc) {
            conflictWith = other.cls.materia
          }
        }
      }
      item.laneCount = maxLane
      item.conflict = item.laneCount > 1
      positions.push({ ...item, conflictWith })
    }
  })

  return positions
}

function TimetableGrid({ classes, onSelect, selectedKey }: TimetableGridProps) {
  const dayIndex = useMemo(() => new Map(DAY_ORDER.map((day, idx) => [day, idx])), [])
  const normalizedMeetings = useMemo(
    () => classes.flatMap((cls) => buildMeetingsForClass(cls)),
    [classes]
  )
  const positions = useMemo(() => buildPositions(normalizedMeetings), [normalizedMeetings])

  const summaryRows = useMemo(
    () =>
      classes.map((cls) => {
        const meetings = normalizedMeetings
          .filter((item) => item.cls.nrc === cls.nrc)
          .sort(
            (a, b) =>
              DAY_ORDER.indexOf(a.day) - DAY_ORDER.indexOf(b.day) || a.start - b.start
          )
        if (!meetings.length) {
          return {
            ...cls,
            daysLabel: cls.dias?.join(", ") || "—",
            timeLabel: cls.inicio ? `${formatTime(cls.inicio)} - ${formatTime(cls.fin)}` : "—",
          }
        }
        const daysLabel = meetings.map((meeting) => meeting.day).join(", ")
        const timeLabel = meetings
          .map((meeting) => `${formatTime(meeting.start)}-${formatTime(meeting.end)}`)
          .join(" / ")
        return {
          ...cls,
          daysLabel,
          timeLabel,
        }
      }),
    [classes, normalizedMeetings]
  )

  return (
    <div className="space-y-4">
      <div className="grid-soft overflow-x-auto rounded-2xl border border-line bg-white">
        <div
          className="grid"
          style={{
            gridTemplateColumns: `88px repeat(${DAY_ORDER.length}, minmax(160px, 1fr))`,
            gridTemplateRows: `44px repeat(${rowCount}, 28px)`,
            minWidth: `${88 + DAY_ORDER.length * 160}px`,
          }}
        >
          <div className="sticky left-0 top-0 z-30 border-b border-line bg-white/95" />
          {DAY_ORDER.map((day, idx) => (
            <div
              key={day}
              className="flex items-center justify-center border-b border-line text-xs font-semibold text-ink/70"
              style={{ gridColumn: idx + 2, gridRow: 1 }}
            >
              {day}
            </div>
          ))}

          {Array.from({ length: MAX_HOUR - MIN_HOUR + 1 }).map((_, idx) => {
            const hour = MIN_HOUR + idx
            const row = (hour - MIN_HOUR) * (60 / STEP) + 2
            return (
              <div
                key={hour}
                className="sticky left-0 z-20 border-b border-line bg-white/95 px-2 text-[11px] text-ink/50"
                style={{ gridColumn: 1, gridRow: row }}
              >
                {String(hour).padStart(2, "0")}:00
              </div>
            )
          })}

          {positions.map((pos) => {
            const colIndex = dayIndex.get(pos.day)
            if (colIndex === undefined) return null
            const rowStart = toRowStart(pos.start)
            const rowSpan = toRowSpan(pos.start, pos.end)
            const key = `${pos.cls.clave_materia}-${pos.day}-${pos.cls.nrc}-${pos.start}`
            const isSelected = selectedKey === pos.cls.nrc
            const laneWidth = 100 / pos.laneCount
            const gap = pos.laneCount > 1 ? 6 : 0
            const left = `calc(${laneWidth * pos.lane}% + ${pos.lane * gap}px)`
            const width = `calc(${laneWidth}% - ${gap}px)`
            const baseTitle = `${pos.cls.materia} - ${pos.cls.profesor} - ${pos.day} ${formatTime(
              pos.start
            )}-${formatTime(pos.end)} - NRC ${pos.cls.nrc}`
            const conflictTitle = pos.conflictWith ? `Se empalma con ${pos.conflictWith}` : "Conflicto de horario"
            const title = pos.conflict ? `${baseTitle}. ${conflictTitle}` : baseTitle

            return (
              <button
                key={key}
                type="button"
                title={title}
                onClick={() => onSelect(pos.cls)}
                className={cn(
                  "relative rounded-xl border px-3 py-2 text-left text-xs shadow-sm transition hover:border-ink hover:shadow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink/30",
                  isSelected ? "border-ink bg-ink text-white" : "border-line bg-white/90",
                  pos.conflict && !isSelected && "border-ink/60"
                )}
                style={{
                  gridColumn: colIndex + 2,
                  gridRow: `${rowStart} / span ${rowSpan}`,
                  width,
                  left,
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <div
                    className={cn(
                      "text-[10px] uppercase tracking-[0.2em]",
                      isSelected ? "text-white/70" : "text-ink/60"
                    )}
                  >
                    {pos.cls.clave_materia}
                  </div>
                  {pos.conflict && (
                    <Badge
                      variant={isSelected ? "ghost" : "default"}
                      title={pos.conflictWith ? `Se empalma con ${pos.conflictWith}` : undefined}
                    >
                      conflicto
                    </Badge>
                  )}
                </div>
                <div className="mt-1 line-clamp-2 text-[13px] font-semibold leading-snug">
                  {pos.cls.materia}
                </div>
                <div className={cn("line-clamp-1 text-[11px]", isSelected ? "text-white/70" : "text-ink/60")}>
                  {pos.cls.profesor}
                </div>
                <div
                  className={cn(
                    "mt-1 text-[10px] uppercase tracking-[0.2em]",
                    isSelected ? "text-white/60" : "text-ink/50"
                  )}
                >
                  NRC {pos.cls.nrc} / Sec {pos.cls.seccion}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      <div className="rounded-2xl border border-line bg-white/80 p-4 text-xs text-ink/70">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-sm font-semibold text-ink">Materias en este horario: {classes.length}</div>
          <div className="text-[11px] text-ink/50 md:hidden">Desliza para ver todos los dias.</div>
        </div>
        {summaryRows.length > 0 ? (
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-left">
              <thead className="text-[11px] uppercase tracking-[0.2em] text-ink/40">
                <tr>
                  <th className="pb-2 pr-4">Materia</th>
                  <th className="pb-2 pr-4">Profesor</th>
                  <th className="pb-2 pr-4">Dias</th>
                  <th className="pb-2">Hora</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {summaryRows.map((cls) => (
                  <tr key={`${cls.clave_materia}-${cls.nrc}`}>
                    <td className="py-2 pr-4 align-top font-medium text-ink">
                      <span className="line-clamp-1">{cls.materia}</span>
                    </td>
                    <td className="py-2 pr-4 align-top">
                      <span className="line-clamp-1 text-ink/70">{cls.profesor}</span>
                    </td>
                    <td className="py-2 pr-4 align-top text-ink/60">{cls.daysLabel}</td>
                    <td className="py-2 align-top text-ink/60">{cls.timeLabel}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="mt-2 text-[11px] text-ink/50">No hay materias en este horario.</div>
        )}
      </div>
    </div>
  )
}

export default TimetableGrid

function formatTime(minutes: number) {
  const safeMinutes = Math.max(0, minutes)
  const hours = Math.floor(safeMinutes / 60)
  const mins = safeMinutes % 60
  return `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}`
}
