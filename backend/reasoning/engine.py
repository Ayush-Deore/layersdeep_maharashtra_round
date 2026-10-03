"""
TrustLayer — Main Reasoning Engine

Orchestrates:
- Scorer (artifact-level manipulation scoring)
- Matcher (entity & metadata matching)
- Contradictions (single-artifact & cross-artifact conflict detection)
- Graph (relationship builder for Evidence Graph)
- LLM Explainer (rationale & free-text explanation synthesis)

Conforms strictly to /schemas/verdict.schema.json.
"""

import logging
from typing import Any

from .scorer import score_artifacts
from .matcher import match_entities
from .contradictions import detect_contradictions
from .graph import build_graph_relationships
from .llm_explainer import generate_explanation

logger = logging.getLogger(__name__)


def analyze_evidence(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Main entry point for Participant 2's reasoning engine.
    Input: List of artifact dicts conforming to evidence.schema.json.
    Output: Verdict dict conforming to verdict.schema.json.
    """
    if not artifacts:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.0,
            "evidence_coverage": 0.0,
            "reasons": ["No artifacts provided for analysis."],
            "contradictions": [],
            "relationships": [],
            "artifact_scores": {},
            "llm_explanation": "No artifacts were provided for cross-modal reasoning.",
        }

    total_artifacts = len(artifacts)
    usable_artifacts = sum(
        1 for a in artifacts
        if a.get("evidence") or not a.get("analysis_errors")
    )
    evidence_coverage = round(usable_artifacts / total_artifacts, 2) if total_artifacts > 0 else 0.0

    # 1. Artifact-Level Manipulation Scoring (Separate from Verdict)
    artifact_scores = score_artifacts(artifacts)

    # 2. Entity & Metadata Matching
    entity_matches = match_entities(artifacts)

    # 3. Contradiction Detection (Single-artifact & Cross-artifact)
    contradictions = detect_contradictions(artifacts, artifact_scores, entity_matches)

    # 4. Evidence Graph Relationships
    relationships = build_graph_relationships(entity_matches)

    # 5. Verdict Classification Logic
    high_severity_contradictions = [c for c in contradictions if c.get("severity") == "high"]
    medium_severity_contradictions = [c for c in contradictions if c.get("severity") == "medium"]

    synthetic_artifacts = [aid for aid, s in artifact_scores.items() if s.get("has_synthetic")]
    manipulated_artifacts = [
        aid for aid, s in artifact_scores.items()
        if s.get("has_synthetic") or s.get("has_metadata_manipulation") or s.get("manipulation_score", 0) >= 0.5
    ]

    reasons: list[str] = []

    # Check for entity matches
    person_map = entity_matches.get("person_map", {})
    shared_persons = {pid: arts for pid, arts in person_map.items() if len(dict.fromkeys(arts)) >= 2}

    for pid, arts in shared_persons.items():
        reasons.append(f"Entity '{pid}' recognized across {len(dict.fromkeys(arts))} artifacts ({', '.join(dict.fromkeys(arts))}).")

    # RULE 3 ENFORCEMENT:
    # COORDINATED_SYNTHETIC requires meaningful cross-artifact evidence (shared entity conflicts or cross-artifact contradictions).
    # Multiple manipulated artifacts without cross-artifact coordination are classified as MANIPULATED.

    has_cross_artifact_coordination = len(high_severity_contradictions) >= 1 or (
        len(synthetic_artifacts) >= 1 and len(shared_persons) >= 1 and total_artifacts >= 2
    )

    if usable_artifacts == 0 or evidence_coverage < 0.20:
        verdict = "INSUFFICIENT_EVIDENCE"
        confidence = 0.35
        reasons.append("Insufficient evidence coverage across submitted artifacts.")

    elif has_cross_artifact_coordination:
        verdict = "COORDINATED_SYNTHETIC"
        peak_synth = max([artifact_scores[aid]["manipulation_score"] for aid in synthetic_artifacts] or [0.85])
        confidence = round(min(0.98, peak_synth + 0.05), 2)
        reasons.append(
            f"Coordinated synthetic campaign detected across {len(synthetic_artifacts)} artifact(s) "
            f"involving cross-artifact entity/media relationships."
        )

    elif len(manipulated_artifacts) >= 1:
        verdict = "MANIPULATED"
        peak_score = max([artifact_scores[aid]["manipulation_score"] for aid in manipulated_artifacts] or [0.75])
        confidence = round(min(0.95, peak_score), 2)
        reasons.append(f"Localized digital manipulation detected in artifact(s): {', '.join(manipulated_artifacts)}.")

    else:
        verdict = "AUTHENTIC"
        confidence = round(min(0.96, 0.90 + (evidence_coverage * 0.06)), 2)
        reasons.append("All analyzed artifacts pass authenticity checks with clean metadata and consistent signals.")

    dedup_reasons = list(dict.fromkeys(reasons))

    # 6. LLM / Template Explanation Synthesis
    llm_explanation = generate_explanation(
        verdict=verdict,
        confidence=confidence,
        evidence_coverage=evidence_coverage,
        reasons=dedup_reasons,
        contradictions=contradictions,
        relationships=relationships,
        artifacts=artifacts,
    )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence_coverage": evidence_coverage,
        "reasons": dedup_reasons,
        "contradictions": contradictions,
        "relationships": relationships,
        "artifact_scores": artifact_scores,
        "llm_explanation": llm_explanation,
    }
