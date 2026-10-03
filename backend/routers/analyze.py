"""
POST /analyze — main endpoint (real-life processing)

Changes in this revision:
- FaceRegistry instantiated per request so person_ids are consistent
  across all artifacts in a single upload batch
- POST /analyze/url — analyze images/videos directly from a public URL
- Better filename-based MIME detection with content-type fallback
- Unique per-file temp paths to avoid collisions on concurrent requests
"""

import logging
import os
import tempfile
import uuid
from typing import Annotated, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from analyzers import image_analyzer, metadata_analyzer, video_analyzer
from utils.face_registry import new_registry
from utils.file_utils import artifact_id, artifact_type_from_mime, guess_mime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])

MAX_FILE_SIZE_MB = 500
DOWNLOAD_TIMEOUT_S = 60


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _save_upload(file: UploadFile, dest_path: str) -> None:
    """Stream an UploadFile to disk."""
    with open(dest_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)


async def _download_url(url: str, dest_path: str, content_type_hint: str = "") -> str:
    """Download a URL to disk. Returns detected MIME type."""
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", content_type_hint).split(";")[0].strip()
            with open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes(1024 * 1024):
                    f.write(chunk)
    return content_type


def _dispatch(path: str, mime: str, filename: str, registry) -> dict:
    """
    Route to analyzer, assemble Evidence JSON for one artifact.
    `registry` is a shared FaceRegistry for the current request batch.
    """
    art_type = artifact_type_from_mime(mime)
    art_id = artifact_id(filename)
    evidence: list[dict] = []
    errors: list[str] = []

    if art_type == "image":
        img_ev, img_err = image_analyzer.analyze(path, registry=registry)
        evidence.extend(img_ev)
        errors.extend(img_err)
        evidence.extend(metadata_analyzer.analyze(path))

    elif art_type == "video":
        vid_ev, vid_err = video_analyzer.analyze(path, registry=registry)
        evidence.extend(vid_ev)
        errors.extend(vid_err)

    else:
        evidence.extend(metadata_analyzer.analyze(path))
        errors.append(f"Modality '{art_type}' not fully supported yet — metadata only.")

    return {
        "artifact_id": art_id,
        "type": art_type,
        "filename": filename,
        "mime_type": mime,
        "evidence": evidence,
        "analysis_errors": errors,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/", summary="Analyze uploaded files")
async def analyze_files(
    files: Annotated[list[UploadFile], File(description="One or more image or video files")],
):
    """
    Upload one or more artifacts for analysis.
    All artifacts in a single request share a **FaceRegistry** so the same
    person gets the same `person_id` across images/video frames.

    Returns an array of Evidence JSON objects — one per artifact.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    registry = new_registry()
    results = []

    with tempfile.TemporaryDirectory(prefix="trustlayer_") as tmp_dir:
        for file in files:
            filename = file.filename or f"upload_{uuid.uuid4().hex[:8]}"
            mime = guess_mime(filename)
            dest_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{os.path.splitext(filename)[1]}")

            logger.info("Received: %s (%s)", filename, mime)

            try:
                await _save_upload(file, dest_path)
            except Exception as exc:
                results.append(_error_artifact(filename, mime, f"upload_failed: {exc}"))
                continue

            try:
                artifact = _dispatch(dest_path, mime, filename, registry)
            except Exception as exc:
                logger.exception("Analysis crashed for %s", filename)
                artifact = _error_artifact(filename, mime, f"analysis_crashed: {exc}")

            results.append(artifact)

    return JSONResponse(content=results)


@router.post("/url", summary="Analyze artifact from a public URL")
async def analyze_from_url(
    url: Annotated[str, Form(description="Publicly accessible image or video URL")],
    filename: Annotated[Optional[str], Form(description="Override filename (optional)")] = None,
):
    """
    Download a file from a URL and run the full analysis pipeline.
    Useful for demos — paste a public image/video URL and get Evidence JSON.

    Example URLs that work:
    - `https://upload.wikimedia.org/wikipedia/commons/thumb/.../file.jpg`
    - Any direct-link MP4/JPG/PNG
    """
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Derive filename from URL if not provided
    if not filename:
        filename = url.split("?")[0].split("/")[-1] or "artifact"
        if "." not in filename:
            filename += ".jpg"  # safe fallback

    registry = new_registry()

    with tempfile.TemporaryDirectory(prefix="trustlayer_url_") as tmp_dir:
        ext = os.path.splitext(filename)[1] or ".bin"
        dest_path = os.path.join(tmp_dir, f"download{ext}")

        try:
            detected_mime = await _download_url(url, dest_path)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Download error: {exc}")

        # Prefer filename-based MIME; fall back to Content-Type from server
        mime = guess_mime(filename) or detected_mime or "application/octet-stream"

        try:
            artifact = _dispatch(dest_path, mime, filename, registry)
        except Exception as exc:
            logger.exception("Analysis crashed for URL %s", url)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    artifact["source_url"] = url
    return JSONResponse(content=artifact)


@router.post("/mock", summary="Return mock Evidence JSON (no upload needed)")
async def analyze_mock():
    """
    Returns a realistic mock Evidence JSON for Participant 2 integration.
    Covers all 4 demo scenarios: AUTHENTIC, MANIPULATED, COORDINATED_SYNTHETIC, INSUFFICIENT.
    """
    import json
    mock_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "mock_data", "sample_evidence.json")
    )
    if os.path.exists(mock_path):
        with open(mock_path) as f:
            return JSONResponse(content=json.load(f))
    raise HTTPException(status_code=404, detail="mock_data/sample_evidence.json not found")


@router.get("/status", summary="Model load status")
async def analyze_status():
    """Check which AI models are currently loaded and ready."""
    from models.loader import model_status
    status = model_status()
    all_ready = all(status.values())
    return JSONResponse(content={
        "ready": all_ready,
        "models": status,
        "note": "Models load on first use if not pre-warmed." if not all_ready else "All models loaded.",
    })


# ── Internal helpers ───────────────────────────────────────────────────────────

def _error_artifact(filename: str, mime: str, error: str) -> dict:
    return {
        "artifact_id": artifact_id(filename),
        "type": artifact_type_from_mime(mime),
        "filename": filename,
        "mime_type": mime,
        "evidence": [],
        "analysis_errors": [error],
    }
