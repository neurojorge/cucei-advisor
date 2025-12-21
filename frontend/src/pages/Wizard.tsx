import { useEffect, useMemo, useState } from "react"

import {
  fetchMeta,
  fetchPlan,
  generateSchedules,
  type GenerateRequest,
  type MetaResponse,
  type PlanResponse,
  type Schedule,
  type ScheduleClass,
} from "../api/client"
import ClassDrawer from "../components/ClassDrawer"
import CopyNrcButton from "../components/CopyNrcButton"
import PreferencesForm, { type PreferencesValues } from "../components/PreferencesForm"
import SchedulesCarousel from "../components/SchedulesCarousel"
import Stepper from "../components/Stepper"
import TimetableGrid from "../components/TimetableGrid"
import { PreferencesSkeleton, ScheduleSkeleton } from "../components/Skeletons"
import { Button } from "../components/ui/button"
import { Card, CardContent } from "../components/ui/card"
import { Toast, ToastClose, ToastDescription, ToastProvider, ToastTitle, ToastViewport } from "../components/ui/toast"

const DEFAULT_VALUES: PreferencesValues = {
  carrera: "INFO",
  semestre: 3,
  prefHorario: "manana",
  barcoExigente: 0.3,
  aprenderPasar: 0.4,
  soloConCupo: false,
}

function Wizard() {
  const [step, setStep] = useState(0)
  const [meta, setMeta] = useState<MetaResponse | null>(null)
  const [plan, setPlan] = useState<PlanResponse | null>(null)
  const [loadingMeta, setLoadingMeta] = useState(true)
  const [loadingPlan, setLoadingPlan] = useState(false)
  const [loadingSchedules, setLoadingSchedules] = useState(false)
  const [values, setValues] = useState<PreferencesValues>(DEFAULT_VALUES)
  const [schedules, setSchedules] = useState<Schedule[]>([])
  const [selectedScheduleId, setSelectedScheduleId] = useState<string | null>(null)
  const [selectedClass, setSelectedClass] = useState<ScheduleClass | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [toastOpen, setToastOpen] = useState(false)
  const [toastMessage, setToastMessage] = useState("")
  const [availabilityQuality, setAvailabilityQuality] = useState<"unknown" | "available">("available")
  const [diagnosticBullets, setDiagnosticBullets] = useState<string[]>([])

  const pushError = (message: string) => {
    setToastMessage(message)
    setToastOpen(true)
  }

  useEffect(() => {
    setLoadingMeta(true)
    fetchMeta()
      .then((data) => {
        setMeta(data)
        setAvailabilityQuality(data.availability_quality)
      })
      .catch((err: Error) => pushError(err.message))
      .finally(() => setLoadingMeta(false))
  }, [])

  useEffect(() => {
    setLoadingPlan(true)
    fetchPlan(values.carrera)
      .then((data) => {
        setPlan(data)
        const semestres = data.semestres.map((item) => item.semestre)
        if (!semestres.includes(values.semestre)) {
          setValues((prev) => ({ ...prev, semestre: semestres[0] || prev.semestre }))
        }
      })
      .catch((err: Error) => pushError(err.message))
      .finally(() => setLoadingPlan(false))
  }, [values.carrera])

  useEffect(() => {
    if (availabilityQuality === "unknown" && values.soloConCupo) {
      setValues((prev) => ({ ...prev, soloConCupo: false }))
    }
  }, [availabilityQuality, values.soloConCupo])

  const semestresDisponibles = useMemo(() => {
    return plan?.semestres.map((item) => item.semestre) || []
  }, [plan])

  const selectedSchedule = useMemo(
    () => schedules.find((item) => item.schedule_id === selectedScheduleId) || null,
    [schedules, selectedScheduleId]
  )

  const handleGenerate = async () => {
    setLoadingSchedules(true)
    try {
      const payload: GenerateRequest = {
        carrera_clave: values.carrera,
        semestre: values.semestre,
        pref_horario: values.prefHorario,
        barco_exigente: values.barcoExigente,
        aprender_pasar: values.aprenderPasar,
        solo_con_cupo: values.soloConCupo,
      }
      const response = await generateSchedules(payload)
      setSchedules(response.schedules)
      setSelectedScheduleId(response.schedules[0]?.schedule_id || null)
      setSelectedClass(null)
      setAvailabilityQuality(response.availability_quality)
      setDiagnosticBullets(response.diagnostic_bullets || [])
      setStep(1)
    } catch (err) {
      pushError((err as Error).message)
    } finally {
      setLoadingSchedules(false)
    }
  }

  const handleNext = () => {
    if (step === 0) {
      void handleGenerate()
      return
    }
    setStep((prev) => Math.min(prev + 1, 2))
  }

  const handleBack = () => setStep((prev) => Math.max(prev - 1, 0))

  const handleSelectClass = (cls: ScheduleClass) => {
    setSelectedClass(cls)
    setDrawerOpen(true)
  }

  const handleSelectSchedule = (id: string) => {
    setSelectedScheduleId(id)
    setSelectedClass(null)
  }

  return (
    <ToastProvider>
      <div className="space-y-8">
        <div className="flex flex-col gap-4 rounded-2xl border border-line bg-white/80 p-6 shadow-soft">
          <Stepper current={step} />
          {loadingMeta ? (
            <div className="text-xs text-ink/50">Cargando meta...</div>
          ) : (
            meta && (
              <div className="flex flex-wrap gap-2 text-xs text-ink/60">
                <span>Ciclo: {meta.cycle || "N/D"}</span>
                <span>Oferta: {meta.offer_rows} filas</span>
                <span>Profesores: {meta.professors}</span>
                <span>Plan: {meta.plan_loaded ? "listo" : "no"}</span>
              </div>
            )
          )}
        </div>

        {step === 0 && (
          <Card className="glass-panel fade-up">
            <CardContent className="space-y-6">
              <div className="space-y-1">
                <h2 className="text-lg font-semibold">Paso 1: Preferencias</h2>
                <p className="text-sm text-ink/60">Elige carrera, semestre y filtros.</p>
              </div>
            {loadingPlan ? (
              <PreferencesSkeleton />
            ) : (
              <PreferencesForm
                values={values}
                semestres={semestresDisponibles}
                availabilityQuality={availabilityQuality}
                onChange={(next) => setValues((prev) => ({ ...prev, ...next }))}
              />
            )}
          </CardContent>
        </Card>
      )}

        {step === 1 && (
          <div className="space-y-6 fade-up">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">Paso 2: Horarios</h2>
                <p className="text-sm text-ink/60">Elige tu horario.</p>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" onClick={handleBack}>
                  Volver
                </Button>
                <Button onClick={handleNext}>Continuar</Button>
              </div>
            </div>
            {loadingSchedules ? (
              <ScheduleSkeleton />
            ) : (
              <div className="space-y-4">
                {diagnosticBullets.length > 0 && (
                  <div className="rounded-2xl border border-line bg-fog p-4 text-xs text-ink/70">
                    {diagnosticBullets.map((bullet) => (
                      <p key={bullet}>• {bullet}</p>
                    ))}
                  </div>
                )}
                <SchedulesCarousel
                  schedules={schedules}
                  selectedId={selectedScheduleId}
                  onSelect={handleSelectSchedule}
                />
              </div>
            )}
            {selectedSchedule && (
              <TimetableGrid
                schedule={selectedSchedule}
                onSelect={handleSelectClass}
                selectedKey={selectedClass?.nrc || null}
              />
            )}
          </div>
        )}

        {step === 2 && (
          <div className="space-y-6 fade-up">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">Paso 3: Confirmacion</h2>
                <p className="text-sm text-ink/60">Copia los NRCs para inscribir.</p>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" onClick={handleBack}>
                  Volver
                </Button>
              </div>
            </div>

            <Card className="glass-panel">
              <CardContent className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-base font-semibold">Acciones rapidas</h3>
                  <p className="text-xs text-ink/60">Copia los NRCs en un clic.</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <CopyNrcButton classes={selectedSchedule?.classes || []} />
                </div>
              </CardContent>
            </Card>

            {selectedSchedule && (
              <TimetableGrid
                schedule={selectedSchedule}
                onSelect={handleSelectClass}
                selectedKey={selectedClass?.nrc || null}
              />
            )}
          </div>
        )}

        <div className="flex items-center justify-between">
          {step === 0 && (
            <Button onClick={handleNext} disabled={loadingPlan || loadingSchedules}>
              {loadingSchedules ? "Generando..." : "Generar horarios"}
            </Button>
          )}
        </div>

        <ClassDrawer open={drawerOpen} onOpenChange={setDrawerOpen} selectedClass={selectedClass} />
      </div>

      <Toast open={toastOpen} onOpenChange={setToastOpen} duration={5000}>
        <div className="space-y-1">
          <ToastTitle>Ocurrió un error</ToastTitle>
          <ToastDescription>{toastMessage}</ToastDescription>
        </div>
        <ToastClose>Cerrar</ToastClose>
      </Toast>
      <ToastViewport />
    </ToastProvider>
  )
}

export default Wizard
