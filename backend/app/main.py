from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .settings import Settings

logging.basicConfig(level=logging.INFO)

settings = Settings()

app = FastAPI(title="CUCEI Advisor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.api_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
