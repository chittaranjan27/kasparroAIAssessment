"""FastAPI application entry-point.

Startup sequence:
  1. Create all DB tables (idempotent).
  2. Mount the /classify and /evaluation-runs routers.
  3. Expose GET /healthz for liveness checks.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.classify import router as classify_router
from app.api.evaluation_runs import router as eval_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup (no-op if they already exist)
    from app.db.session import create_tables
    create_tables()
    yield


app = FastAPI(
    title="Kasparro FSE I – Claim Extraction API",
    version="1.0.0",
    description=(
        "Extract and classify factual claims from AI-generated speaker descriptions. "
        "POST /classify for a single answer; POST /evaluation-runs for a batch experiment."
    ),
    lifespan=lifespan,
)

app.include_router(classify_router, tags=["classification"])
app.include_router(eval_router, tags=["evaluation"])


@app.get("/healthz", tags=["meta"], summary="Liveness check")
def healthz():
    from app.db.session import ping
    db_ok = ping()
    return JSONResponse(
        status_code=200 if db_ok else 503,
        content={"status": "ok" if db_ok else "degraded", "db": db_ok},
    )
