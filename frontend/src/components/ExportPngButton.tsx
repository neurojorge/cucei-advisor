import { useState } from "react"

import { API_BASE } from "../api/client"
import { Button, type ButtonProps } from "./ui/button"

type ExportPngButtonProps = {
  scheduleId: string | null
  size?: ButtonProps["size"]
  variant?: ButtonProps["variant"]
}

function ExportPngButton({ scheduleId, size = "default", variant = "secondary" }: ExportPngButtonProps) {
  const [loading, setLoading] = useState(false)

  const handleExport = async () => {
    if (!scheduleId) return
    setLoading(true)
    try {
      const resp = await fetch(`${API_BASE}/api/schedule/${scheduleId}/png`)
      if (!resp.ok) throw new Error("No se pudo generar PNG")
      const blob = await resp.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = `horario-${scheduleId}.png`
      link.click()
      URL.revokeObjectURL(url)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button
      type="button"
      variant={variant}
      size={size}
      onClick={handleExport}
      disabled={!scheduleId || loading}
    >
      {loading ? "Exportando..." : "Exportar PNG"}
    </Button>
  )
}

export default ExportPngButton
