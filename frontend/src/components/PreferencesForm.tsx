import { Label } from "./ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select"
import { Slider } from "./ui/slider"
import { Switch } from "./ui/switch"

export type PreferencesValues = {
  carrera: "INFO" | "ICOM"
  semestre: number
  prefHorario: "manana" | "tarde" | "mixto"
  barcoExigente: number
  aprenderPasar: number
  soloConCupo: boolean
}

type PreferencesFormProps = {
  values: PreferencesValues
  semestres: number[]
  onChange: (next: Partial<PreferencesValues>) => void
  availabilityQuality?: "unknown" | "available"
}

const horarios = [
  { value: "manana", label: "Mañana" },
  { value: "tarde", label: "Tarde" },
  { value: "mixto", label: "Mixto" },
] as const

function PreferencesForm({ values, semestres, onChange, availabilityQuality }: PreferencesFormProps) {
  const availabilityUnknown = availabilityQuality === "unknown"

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div className="space-y-2">
        <Label>Carrera</Label>
        <Select value={values.carrera} onValueChange={(value) => onChange({ carrera: value as "INFO" | "ICOM" })}>
          <SelectTrigger>
            <SelectValue placeholder="Selecciona" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="INFO">INFO</SelectItem>
            <SelectItem value="ICOM">ICOM</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Semestre</Label>
        <Select
          value={String(values.semestre)}
          onValueChange={(value) => onChange({ semestre: Number(value) })}
        >
          <SelectTrigger>
            <SelectValue placeholder="Selecciona" />
          </SelectTrigger>
          <SelectContent>
            {semestres.map((sem) => (
              <SelectItem key={sem} value={String(sem)}>
                {sem}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Horario preferido</Label>
        <Select
          value={values.prefHorario}
          onValueChange={(value) => onChange({ prefHorario: value as PreferencesValues["prefHorario"] })}
        >
          <SelectTrigger>
            <SelectValue placeholder="Selecciona" />
          </SelectTrigger>
          <SelectContent>
            {horarios.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Solo con cupo</Label>
        <div className="flex items-center gap-3">
          <Switch
            checked={values.soloConCupo}
            disabled={availabilityUnknown}
            onCheckedChange={(checked) => onChange({ soloConCupo: checked })}
          />
          <span className="text-sm text-ink/60">
            {availabilityUnknown ? "Cupo no disponible en este ciclo." : "Filtra secciones sin cupo."}
          </span>
        </div>
        <HintDetails
          hint="Usa cupo oficial cuando existe."
          details="Si el ciclo no trae disponibilidad confiable, el filtro se desactiva."
        />
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label>Barco ↔ Exigente</Label>
          <span className="text-xs text-ink/50">{values.barcoExigente.toFixed(2)}</span>
        </div>
        <Slider
          value={[values.barcoExigente]}
          min={0}
          max={1}
          step={0.05}
          onValueChange={(value) => onChange({ barcoExigente: value[0] })}
        />
        <div className="flex justify-between text-xs text-ink/50">
          <span>Barco</span>
          <span>Exigente</span>
        </div>
        <HintDetails hint="Balancea facilidad vs exigencia." details="0 es más barco, 1 es más exigente." />
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label>Aprender ↔ Pasar</Label>
          <span className="text-xs text-ink/50">{values.aprenderPasar.toFixed(2)}</span>
        </div>
        <Slider
          value={[values.aprenderPasar]}
          min={0}
          max={1}
          step={0.05}
          onValueChange={(value) => onChange({ aprenderPasar: value[0] })}
        />
        <div className="flex justify-between text-xs text-ink/50">
          <span>Aprender</span>
          <span>Pasar</span>
        </div>
        <HintDetails hint="Enfoca aprender o pasar." details="0 prioriza aprender, 1 prioriza pasar." />
      </div>
    </div>
  )
}

export default PreferencesForm

function HintDetails({ hint, details }: { hint: string; details: string }) {
  return (
    <div className="space-y-1 text-xs text-ink/50">
      <div>{hint}</div>
      <details className="group">
        <summary className="cursor-pointer text-[11px] text-ink/40 hover:text-ink/60">¿Qué es esto?</summary>
        <div className="mt-1 text-[11px] text-ink/60">{details}</div>
      </details>
    </div>
  )
}
