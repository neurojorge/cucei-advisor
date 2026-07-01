import { cn } from "../../lib/utils"

function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-xl bg-line/60", className)} />
}

export { Skeleton }
