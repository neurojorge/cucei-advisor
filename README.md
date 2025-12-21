# CUCEI Advisor — Scheduler App

Migracion a arquitectura full-stack:
- **Frontend:** Vite + React + Tailwind + shadcn/ui
- **Backend:** FastAPI (Python)

La app genera hasta 3 horarios sin choques para INFO/ICOM, muestra grid semanal interactivo y despliega reseñas con evidencia real.

## Estructura
- `backend/` API FastAPI + motor de horarios
- `frontend/` UI tipo app con wizard de 3 pasos
- `semestres_materias_INFO_ICOM.csv` plan por semestre (CORE)
- `evaluaciones_con_departamentos.csv` reseñas base

## Requisitos
- Python 3.10+
- Node.js 18+
- Windows (scripts `.ps1` incluidos)

## Datos
La oferta se carga en este orden:
1) `OFFER_DATA_PARQUET`
2) `OFFER_DATA_CSV`
3) `backend/app/data/oferta_cucei_*` o `webscraping/data/oferta_cucei_*`

Las columnas reales se detectan por heuristica y se adaptan a un esquema canonico antes de generar horarios.

## Backend (FastAPI)
Run backend:
```powershell
powershell -File backend/scripts/bootstrap.ps1
powershell -File backend/scripts/run_backend.ps1
```

Endpoints principales:
- `GET /api/health`
- `GET /api/meta`
- `GET /api/plan?carrera=INFO`
- `POST /api/generate`
- `GET /api/reviews?profesor=...`

`/api/meta` incluye `availability_quality` y porcentajes de disponibilidad para saber si el ciclo trae cupo/disponibles confiables.

## Frontend (Vite)
Run frontend:
```powershell
powershell -File frontend/scripts/bootstrap.ps1
powershell -File frontend/scripts/run_frontend.ps1
```

Luego abre: `http://localhost:5173`

## Variables de entorno
Copia `backend/.env.example` a `backend/.env` si necesitas overrides:
- `OFFER_DATA_PARQUET`
- `OFFER_DATA_CSV`
- `PLAN_CSV_PATH`
- `REVIEWS_CSV_PATH`
- `PROFESSOR_ALIASES_PATH`
- `GROQ_API_KEY` / `GROQ_MODEL`
- `EVIDENCE_QUOTE_MAX_LEN`

Frontend (opcional):
- `VITE_API_BASE_URL` (default: `http://127.0.0.1:8000`)

## Pruebas
```powershell
py -m pytest backend/tests
```

## Flujo esperado
1) Selecciona carrera, semestre y preferencias.
2) Genera 3 horarios en carrusel.
3) Explora en grid semanal y abre el drawer con reseñas.
4) Copia NRCs para inscribir.
