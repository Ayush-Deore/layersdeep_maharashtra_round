"""
Utility helpers for file handling and artifact ID generation.
"""

import hashlib
import mimetypes
import os
import uuid
from pathlib import Path


def artifact_id(filename: str, prefix: str = "") -> str:
    """Generate a short deterministic artifact ID from filename."""
    short = hashlib.md5(filename.encode()).hexdigest()[:6]
    tag = prefix or _guess_prefix(filename)
    return f"{tag}_{short}"


def _guess_prefix(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        if mime.startswith("image"):
            return "img"
        if mime.startswith("video"):
            return "vid"
        if mime.startswith("audio"):
            return "aud"
    return "doc"


def guess_mime(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def artifact_type_from_mime(mime: str) -> str:
    if mime.startswith("image"):
        return "image"
    if mime.startswith("video"):
        return "video"
    if mime.startswith("audio"):
        return "audio"
    return "document"
