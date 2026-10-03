"""
Face Registry — session-scoped cross-artifact person matching.

When a face is detected, its embedding is compared against all known embeddings.
If cosine similarity > MATCH_THRESHOLD, the same person_id is reused.
Otherwise, a new person_id is minted.

This allows Participant 2 to do entity matching purely from person_ids,
without handling raw float arrays.

Usage:
    registry = FaceRegistry()
    person_id = registry.register(embedding_array)
"""

import threading
from typing import Optional

import numpy as np

MATCH_THRESHOLD = 0.45  # cosine similarity threshold (tuned for InsightFace buffalo_l)


class FaceRegistry:
    """Thread-safe registry of face embeddings → person IDs within one analysis session."""

    def __init__(self):
        self._lock = threading.Lock()
        self._embeddings: list[np.ndarray] = []
        self._ids: list[str] = []
        self._counter = 0

    def register(self, embedding: Optional[list | np.ndarray]) -> str:
        """
        Register a face embedding and return its stable person_id.
        If a matching identity already exists, returns the existing ID.
        """
        if embedding is None:
            return self._new_id()

        emb = np.array(embedding, dtype=np.float32)
        # Normalize just in case (InsightFace already normed, but be safe)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        with self._lock:
            for known_emb, known_id in zip(self._embeddings, self._ids):
                similarity = float(np.dot(emb, known_emb))
                if similarity >= MATCH_THRESHOLD:
                    return known_id
            # New identity
            pid = self._new_id()
            self._embeddings.append(emb)
            self._ids.append(pid)
            return pid

    def _new_id(self) -> str:
        self._counter += 1
        return f"person_{self._counter:02d}"

    @property
    def known_count(self) -> int:
        return len(self._ids)


# ── Per-request registry factory ──────────────────────────────────────────────
# Each call to /analyze gets its own registry so person IDs are
# scoped to a single analysis session (set of uploaded artifacts).

def new_registry() -> FaceRegistry:
    return FaceRegistry()
