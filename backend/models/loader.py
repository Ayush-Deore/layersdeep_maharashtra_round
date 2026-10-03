"""
Model loader — lazy singleton with warmup support.

Call warmup_all() at startup to pre-download and load models so the first
real request doesn't take 30+ seconds.
"""

import logging
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_face_lock = Lock()
_deepfake_lock = Lock()

_face_app = None
_deepfake_pipe = None


def get_face_app():
    """Return a lazily-loaded InsightFace FaceAnalysis instance."""
    global _face_app
    if _face_app is not None:
        return _face_app
    with _face_lock:
        if _face_app is not None:
            return _face_app
        try:
            from insightface.app import FaceAnalysis
            logger.info("Loading InsightFace (buffalo_l) …")
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            _face_app = app
            logger.info("InsightFace loaded ✓")
        except Exception as exc:
            logger.warning("InsightFace unavailable (%s) — face analysis disabled.", exc)
            _face_app = None
    return _face_app


def get_deepfake_pipe():
    """Return a lazily-loaded HuggingFace deepfake-detection pipeline."""
    global _deepfake_pipe
    if _deepfake_pipe is not None:
        return _deepfake_pipe
    with _deepfake_lock:
        if _deepfake_pipe is not None:
            return _deepfake_pipe
        try:
            from transformers import pipeline
            logger.info("Loading deepfake-detection model (ViT) …")
            _deepfake_pipe = pipeline(
                "image-classification",
                model="dima806/deepfake_vs_real_image_detection",
                device=-1,  # CPU
            )
            logger.info("Deepfake model loaded ✓")
        except Exception as exc:
            logger.warning("Deepfake model unavailable (%s) — disabled.", exc)
            _deepfake_pipe = None
    return _deepfake_pipe


def warmup_all() -> dict:
    """
    Pre-load all models. Call once at server startup.
    Returns a status dict of what loaded successfully.
    """
    status = {}

    face = get_face_app()
    status["insightface"] = face is not None

    pipe = get_deepfake_pipe()
    status["deepfake_vit"] = pipe is not None

    logger.info("Model warmup complete: %s", status)
    return status


def model_status() -> dict:
    """Return current load state of each model (non-blocking)."""
    return {
        "insightface": _face_app is not None,
        "deepfake_vit": _deepfake_pipe is not None,
    }
