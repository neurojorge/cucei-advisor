import { Skeleton } from "./ui/skeleton"

export function PreferencesSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-12 w-full" />
      <Skeleton className="h-12 w-full" />
    </div>
  )
}

export function ScheduleSkeleton() {
  return (
    <div className="flex gap-4">
      <Skeleton className="h-72 w-72" />
      <Skeleton className="h-72 w-72" />
      <Skeleton className="h-72 w-72" />
    </div>
  )
}
