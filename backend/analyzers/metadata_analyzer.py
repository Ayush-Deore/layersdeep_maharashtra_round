"""
Metadata Analyzer
Extracts EXIF and file-level metadata from images (and other files via hachoir).
Emits evidence items of type "metadata".
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _exif_from_pillow(path: str) -> Optional[dict]:
    try:
        from PIL import Image, ExifTags
        img = Image.open(path)
        raw_exif = img._getexif()
        if not raw_exif:
            return None
        readable = {}
        for tag_id, value in raw_exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            # Skip binary blobs
            if isinstance(value, bytes):
                continue
            readable[tag] = value
        return readable
    except Exception as exc:
        logger.debug("Pillow EXIF extraction failed: %s", exc)
        return None


def _file_stats(path: str) -> dict:
    stat = os.stat(path)
    return {
        "size_bytes": stat.st_size,
        "filename": Path(path).name,
    }


def analyze(path: str) -> list[dict]:
    """
    Returns a list of evidence dicts for metadata signals found in the file.
    """
    evidence = []
    errors = []

    # ── File stats (always available) ─────────────────────────────────────────
    stats = _file_stats(path)

    # ── EXIF via Pillow ────────────────────────────────────────────────────────
    exif = _exif_from_pillow(path)

    meta_value: dict[str, Any] = {**stats}

    camera_fields = ["Make", "Model", "Software"]
    gps_fields = ["GPSInfo"]

    if exif:
        for field in camera_fields:
            if field in exif:
                meta_value[field.lower()] = str(exif[field])
        if "GPSInfo" in exif:
            meta_value["gps_present"] = True
        else:
            meta_value["gps_present"] = False

        # Suspicious signals: software edited
        software = str(exif.get("Software", "")).lower()
        manipulation_signals = ["photoshop", "gimp", "lightroom", "pixelmator", "affinity"]
        if any(sig in software for sig in manipulation_signals):
            evidence.append({
                "type": "manipulation",
                "value": f"editing_software_detected:{exif.get('Software')}",
                "confidence": 0.72,
                "source_model": "metadata_analyzer",
            })

        # DateTime vs DateTimeOriginal mismatch = potential manipulation
        dt = exif.get("DateTime")
        dt_orig = exif.get("DateTimeOriginal")
        if dt and dt_orig and dt != dt_orig:
            evidence.append({
                "type": "manipulation",
                "value": "datetime_mismatch",
                "confidence": 0.55,
                "source_model": "metadata_analyzer",
            })

    evidence.append({
        "type": "metadata",
        "value": meta_value,
        "confidence": 1.0,
        "source_model": "metadata_analyzer",
    })

    return evidence
