function Header() {
  return (
    <header className="mb-8 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-ink/50">CUCEI Advisor</p>
          <h1 className="text-3xl font-semibold">Tu horario con datos reales</h1>
        </div>
        <div className="hidden md:flex items-center gap-2 text-xs text-ink/60">
          <span className="rounded-full border border-line px-3 py-1">INFO / ICOM</span>
          <span className="rounded-full border border-line px-3 py-1">SIIAU</span>
        </div>
      </div>
      <p className="max-w-2xl text-sm text-ink/60">
        Preferencias claras, reseñas reales y un grid semanal para decidir rápido.
      </p>
    </header>
  )
}

export default Header
