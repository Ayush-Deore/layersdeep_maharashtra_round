"""
TrustLayer — AI Analysis Backend
FastAPI entry point with startup model warmup.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from routers import analyze

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models in a background thread so the event loop isn't blocked."""
    logger.info("Starting model warmup …")
    loop = asyncio.get_event_loop()
    status = await loop.run_in_executor(None, _warmup)
    logger.info("Models ready: %s", status)
    yield
    # Shutdown — nothing to clean up for now


def _warmup():
    from models.loader import warmup_all
    return warmup_all()


app = FastAPI(
    title="TrustLayer Analysis API",
    description=(
        "Analyzes digital artifacts (images, videos) and emits structured Evidence JSON "
        "for cross-modal reasoning. Upload real files to **POST /analyze/** or test with "
        "**POST /analyze/mock**."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    from models.loader import model_status
    return {
        "status": "ok",
        "service": "trustlayer-analysis",
        "models": model_status(),
    }
