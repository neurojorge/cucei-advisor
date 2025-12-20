import { cn } from "../lib/utils"

const steps = ["Preferencias", "Horarios", "Confirmacion"]

type StepperProps = {
  current: number
}

function Stepper({ current }: StepperProps) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      {steps.map((label, index) => (
        <div key={label} className="flex items-center gap-2">
          <div
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-full border text-xs font-semibold",
              index <= current ? "border-ink bg-ink text-white" : "border-line bg-white text-ink/50"
            )}
          >
            {index + 1}
          </div>
          <span className={cn("text-sm", index <= current ? "text-ink" : "text-ink/40")}>
            {label}
          </span>
          {index < steps.length - 1 && <span className="mx-2 hidden h-px w-8 bg-line md:block" />}
        </div>
      ))}
    </div>
  )
}

export default Stepper
