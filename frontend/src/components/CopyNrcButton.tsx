import { useState } from "react"

import type { ScheduleClass } from "../api/client"
import { Button } from "./ui/button"

function CopyNrcButton({ classes }: { classes: ScheduleClass[] }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    const text = classes.map((cls) => cls.nrc).filter(Boolean).join(", ")
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <Button type="button" variant="outline" onClick={handleCopy} disabled={!classes.length}>
      {copied ? "NRCs copiados" : "Copiar NRCs"}
    </Button>
  )
}

export default CopyNrcButton
