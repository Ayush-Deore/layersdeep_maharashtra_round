"""
TrustLayer — Contradiction & Conflict Detector

Identifies:
1. Single-artifact manipulation signals (editing software, EXIF timestamp mismatch).
2. Cross-artifact conflicts (same person in authentic vs synthetic media, coordinated synthetic manipulation).
3. Assigns relevant severity ratings ("high", "medium", "low").
"""

from typing import Any


def detect_contradictions(
    artifacts: list[dict[str, Any]],
    artifact_scores: dict[str, dict[str, Any]],
    entity_matches: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Returns a list of contradiction objects conforming to verdict.schema.json:
    [
        {
            "artifact_a": "img_01",
            "artifact_b": "img_02",
            "description": "...",
            "severity": "high" | "medium" | "low"
        }
    ]
    """
    contradictions: list[dict[str, Any]] = []

    person_map = entity_matches.get("person_map", {})

    # 1. Single-artifact manipulation signals
    for art_id, score_info in artifact_scores.items():
        for sig in score_info.get("signals", []):
            val_str = str(sig.get("value", ""))
            if "editing_software_detected" in val_str:
                sw_name = val_str.split(":", 1)[-1] if ":" in val_str else val_str
                contradictions.append({
                    "artifact_a": art_id,
                    "artifact_b": art_id,
                    "description": f"EXIF metadata reveals digital editing software ({sw_name}) used on {art_id}.",
                    "severity": "medium",
                })
            elif "datetime_mismatch" in val_str:
                contradictions.append({
                    "artifact_a": art_id,
                    "artifact_b": art_id,
                    "description": f"Metadata anomaly: EXIF DateTime differs from DateTimeOriginal in {art_id}.",
                    "severity": "medium",
                })

    # 2. Cross-artifact conflicts (shared person_id)
    for pid, art_ids in person_map.items():
        unique_arts = list(dict.fromkeys(art_ids))
        synthetic_arts = [aid for aid in unique_arts if artifact_scores.get(aid, {}).get("has_synthetic")]
        authentic_arts = [
            aid for aid in unique_arts
            if not artifact_scores.get(aid, {}).get("has_synthetic") and
            any(s.get("value") == "authentic_visual" for s in artifact_scores.get(aid, {}).get("signals", []))
        ]

        # Conflict: Same person in authentic artifact AND synthetic artifact
        if synthetic_arts and authentic_arts:
            for s_art in synthetic_arts:
                for a_art in authentic_arts:
                    contradictions.append({
                        "artifact_a": a_art,
                        "artifact_b": s_art,
                        "description": (
                            f"Cross-modal contradiction: Entity '{pid}' appears in authentic artifact "
                            f"'{a_art}' but is synthetically manipulated in '{s_art}'."
                        ),
                        "severity": "high",
                    })

        # Conflict: Coordinated synthetic manipulation of same person across multiple artifacts
        if len(synthetic_arts) >= 2:
            contradictions.append({
                "artifact_a": synthetic_arts[0],
                "artifact_b": synthetic_arts[1],
                "description": (
                    f"Coordinated synthetic manipulation: Entity '{pid}' is manipulated across "
                    f"multiple media files ({', '.join(synthetic_arts)})."
                ),
                "severity": "high",
            })

    return contradictions
