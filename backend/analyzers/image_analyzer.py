"""
Image Analyzer — Real-life Processing (Phase 2 revised)

Key changes vs prototype:
- Images are resized before deepfake detection (ViT expects ≤1024px)
- Face embeddings are resolved to stable person_ids via FaceRegistry
- Label normalization handles any HuggingFace model's label vocabulary
- Authentic signal is also emitted (low fake score) for downstream reasoning
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Max dimension fed to the deepfake ViT (larger → OOM on CPU)
MAX_IMAGE_DIM = 1024


def _load_cv2_image(path: str):
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cv2 returned None for {path}")
        return img
    except Exception as exc:
        logger.warning("cv2 imread failed for %s: %s", path, exc)
        return None


def _pil_open_resized(path: str):
    """Open image, convert to RGB, downscale if needed. Returns PIL Image."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        ratio = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        logger.debug("Resized %s from (%d,%d) → (%d,%d)", path, w, h, *img.size)
    return img


# ── Deepfake Detection ──────────────────────────────────────────────────────────

# All label vocabularies observed across HuggingFace deepfake models
_FAKE_LABELS = {"fake", "deepfake", "synthetic", "ai-generated", "manipulated", "forged", "0"}
_REAL_LABELS = {"real", "authentic", "genuine", "original", "1"}


def _classify_label(label: str) -> str:
    """Normalize a model label to 'fake' | 'real' | 'unknown'."""
    l = label.lower().strip()
    if l in _FAKE_LABELS:
        return "fake"
    if l in _REAL_LABELS:
        return "real"
    # Fallback: label contains keyword
    if any(k in l for k in _FAKE_LABELS):
        return "fake"
    if any(k in l for k in _REAL_LABELS):
        return "real"
    return "unknown"


def _run_deepfake_detection(path: str, crop_bbox: list[int] = None) -> list[dict]:
    """
    Runs ViT-based deepfake detector.
    If crop_bbox is provided, runs detection on the specific cropped region.
    Always emits an evidence item — either 'synthetic_visual' or 'authentic_visual'
    so the reasoning layer has a definitive signal.
    """
    from models.loader import get_deepfake_pipe
    pipe = get_deepfake_pipe()
    if pipe is None:
        return []
    try:
        pil_img = _pil_open_resized(path)
        if crop_bbox:
            # Crop the bounding box (bbox coordinates match because both use MAX_IMAGE_DIM resizing)
            x1, y1, x2, y2 = [int(v) for v in crop_bbox]
            # Ensure within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
            if x2 > x1 and y2 > y1:
                pil_img = pil_img.crop((x1, y1, x2, y2))
                
        results = pipe(pil_img)  # [{"label": str, "score": float}, ...]

        fake_score = 0.0
        real_score = 0.0
        raw_label = ""

        for item in results:
            cls = _classify_label(item["label"])
            if cls == "fake" and item["score"] > fake_score:
                fake_score = item["score"]
                raw_label = item["label"]
            elif cls == "real" and item["score"] > real_score:
                real_score = item["score"]

        logger.debug("Deepfake scores — fake=%.3f real=%.3f label=%s", fake_score, real_score, raw_label)

        if fake_score >= 0.5:
            return [{
                "type": "manipulation",
                "value": "synthetic_visual",
                "confidence": round(fake_score, 4),
                "source_model": "dima806/deepfake_vs_real_image_detection",
                "raw_label": raw_label,
            }]
        elif real_score >= 0.5:
            return [{
                "type": "manipulation",
                "value": "authentic_visual",
                "confidence": round(real_score, 4),
                "source_model": "dima806/deepfake_vs_real_image_detection",
                "raw_label": raw_label,
            }]
        else:
            # Model uncertain
            return [{
                "type": "manipulation",
                "value": "uncertain",
                "confidence": round(max(fake_score, real_score), 4),
                "source_model": "dima806/deepfake_vs_real_image_detection",
            }]

    except Exception as exc:
        logger.warning("Deepfake detection failed: %s", exc)
        return []


# ── Face Detection & ID Resolution ─────────────────────────────────────────────

def _run_face_analysis(path: str, registry=None) -> list[dict]:
    """
    Detects faces via InsightFace, resolves person_ids via FaceRegistry.
    Returns evidence items with person_id instead of raw embedding.
    """
    from models.loader import get_face_app
    face_app = get_face_app()
    if face_app is None:
        return []
    try:
        img = _load_cv2_image(path)
        if img is None:
            return []

        # Resize for face detector too (avoids slow inference on 4K frames)
        h, w = img.shape[:2]
        if max(h, w) > MAX_IMAGE_DIM:
            import cv2
            scale = MAX_IMAGE_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        faces = face_app.get(img)
        evidence = []

        for i, face in enumerate(faces):
            det_score = float(face.det_score) if hasattr(face, "det_score") else 0.8
            embedding = face.normed_embedding.tolist() if hasattr(face, "normed_embedding") else None

            # Resolve to stable person_id
            if registry is not None:
                person_id = registry.register(embedding)
            else:
                person_id = f"person_{i + 1:02d}"

            evidence.append({
                "type": "person",
                "value": {
                    "person_id": person_id,
                    "face_index": i,
                    "bbox": [round(x) for x in face.bbox.tolist()] if hasattr(face, "bbox") else None,
                    "age": int(face.age) if hasattr(face, "age") and face.age else None,
                    "gender": int(face.gender) if hasattr(face, "gender") and face.gender is not None else None,
                },
                "confidence": round(det_score, 4),
                "source_model": "insightface/buffalo_l",
            })

        logger.info("Detected %d face(s) in %s", len(faces), Path(path).name)
        return evidence

    except Exception as exc:
        logger.warning("Face analysis failed: %s", exc)
        return []



def _run_background_noise_detection(path: str) -> list[dict]:
    """Detect whether the background appears overly smooth (synthetic) or contains natural camera noise.
    Returns an evidence item with type 'background' and value either 'synthetic_background' or 'authentic_background'.
    """
    try:
        import cv2
        img = _load_cv2_image(path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute variance of Laplacian – higher means more texture/noise.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Threshold empirically chosen: < 30 -> smooth synthetic background
        if laplacian_var < 30:
            label = "synthetic_background"
            confidence = max(0.0, 1 - laplacian_var / 30)  # higher confidence when very smooth
        else:
            label = "authentic_background"
            confidence = min(1.0, laplacian_var / 100)  # cap confidence
        return [{
            "type": "background",
            "value": label,
            "confidence": round(confidence, 4),
            "source_model": "camera_noise_estimator",
        }]
    except Exception as exc:
        logger.warning("Background noise detection failed: %s", exc)
        return []


def analyze(path: str, registry=None) -> tuple[list[dict], list[str]]:
    """
    Run all image analyzers. Pass a FaceRegistry instance for cross-artifact
    person_id resolution. Returns (evidence_list, error_list).
    """
    evidence = []
    errors = []

    # 1. Run face analysis first
    face_evidence = []
    try:
        face_evidence = _run_face_analysis(path, registry=registry)
        evidence.extend(face_evidence)
    except Exception as exc:
        errors.append(f"face_analysis: {exc}")

    # 2. Run background noise detection
    try:
        evidence.extend(_run_background_noise_detection(path))
    except Exception as exc:
        errors.append(f"background_noise_detection: {exc}")

    # 3. Run deepfake detection on full image
    try:
        evidence.extend(_run_deepfake_detection(path))
    except Exception as exc:
        errors.append(f"deepfake_detection: {exc}")

    # 3. Run deepfake detection on each face crop
    for f_ev in face_evidence:
        bbox = f_ev.get("value", {}).get("bbox")
        person_id = f_ev.get("value", {}).get("person_id")
        if bbox:
            try:
                crop_ev = _run_deepfake_detection(path, crop_bbox=bbox)
                for item in crop_ev:
                    item.setdefault("detail", {})
                    item["detail"]["face_crop"] = True
                    item["detail"]["person_id"] = person_id
                evidence.extend(crop_ev)
            except Exception as exc:
                errors.append(f"deepfake_detection_crop: {exc}")

    return evidence, errors
