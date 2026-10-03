"""
TrustLayer — Artifact-Level Manipulation Scorer

Evaluates manipulation evidence per artifact independently of final verdict logic.
Does NOT average unrelated confidence values (e.g. face detection confidence vs EXIF confidence).
Only aggregates type == "manipulation" signals.
"""

from typing import Any


def score_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Computes artifact-level manipulation scores for each artifact.
    Returns dict: artifact_id -> {
        "manipulation_score": float (0.0 to 1.0),
        "has_synthetic": bool,
        "has_metadata_manipulation": bool,
        "signals": list[dict],
        "explanation": str
    }
    """
    results = {}

    for artifact in artifacts:
        art_id = artifact.get("artifact_id", "unknown")
        evidence_list = artifact.get("evidence", [])

        manip_signals = []
        synthetic_confidences = []
        authentic_confidences = []
        metadata_manip_flags = []

        for ev in evidence_list:
            if ev.get("type") == "manipulation":
                conf = float(ev.get("confidence", 0.0))
                val_str = str(ev.get("value", ""))
                src_model = ev.get("source_model", "")

                signal_item = {
                    "value": val_str,
                    "confidence": conf,
                    "source_model": src_model,
                    "detail": ev.get("detail"),
                    "frame_timestamp": ev.get("frame_timestamp"),
                }
                manip_signals.append(signal_item)

                if "synthetic_visual" in val_str:
                    synthetic_confidences.append(conf)
                elif "authentic_visual" in val_str:
                    authentic_confidences.append(conf)
                elif "editing_software_detected" in val_str or "datetime_mismatch" in val_str:
                    metadata_manip_flags.append(signal_item)

        has_synthetic = len(synthetic_confidences) > 0 and max(synthetic_confidences) >= 0.5
        has_metadata_manip = len(metadata_manip_flags) > 0

        # Compute artifact-level manipulation score
        if has_synthetic:
            # Derived from peak synthetic visual confidence
            art_score = round(max(synthetic_confidences), 4)
            explanation = f"Synthetic visual manipulation detected with {int(art_score * 100)}% confidence."
        elif has_metadata_manip:
            # Derived from highest metadata manipulation confidence
            max_meta_conf = max(f["confidence"] for f in metadata_manip_flags)
            art_score = round(max_meta_conf, 4)
            explanation = f"Metadata anomaly / editing software detected with {int(art_score * 100)}% confidence."
        elif authentic_confidences:
            # Authentic visual score
            max_auth = max(authentic_confidences)
            art_score = round(1.0 - max_auth, 4)  # Low manipulation score for authentic visual
            explanation = f"Authentic visual signal verified ({int(max_auth * 100)}% authentic)."
        else:
            art_score = 0.0
            explanation = "No explicit manipulation signals detected."

        results[art_id] = {
            "artifact_id": art_id,
            "filename": artifact.get("filename", ""),
            "manipulation_score": art_score,
            "has_synthetic": has_synthetic,
            "has_metadata_manipulation": has_metadata_manip,
            "signals": manip_signals,
            "explanation": explanation,
        }

    return results
