"""
TrustLayer — LLM Explanation & Rationale Synthesis Module

Generates human-readable reason strings and detailed free-text 'Why?' explanations
conforming to verdict.schema.json.
Supports optional Gemini API synthesis if key is present, with robust local fallback.
"""

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_explanation(
    verdict: str,
    confidence: float,
    evidence_coverage: float,
    reasons: list[str],
    contradictions: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> str:
    """
    Synthesize a free-text explanation for the verdict.
    Falls back to a structured template synthesis if Gemini API call is unavailable.
    """
    # Attempt Gemini API if configured
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            explanation = _generate_with_gemini(
                api_key=api_key,
                verdict=verdict,
                confidence=confidence,
                reasons=reasons,
                contradictions=contradictions,
                relationships=relationships,
                artifacts=artifacts,
            )
            if explanation:
                return explanation
        except Exception as exc:
            logger.warning("Gemini LLM explanation call failed: %s. Using local synthesis.", exc)

    # Local fallback synthesis
    return _build_local_explanation(
        verdict=verdict,
        confidence=confidence,
        evidence_coverage=evidence_coverage,
        reasons=reasons,
        contradictions=contradictions,
        relationships=relationships,
        artifacts=artifacts,
    )


def _build_local_explanation(
    verdict: str,
    confidence: float,
    evidence_coverage: float,
    reasons: list[str],
    contradictions: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> str:
    lines = []
    
    # Verdict summary headline
    conf_pct = int(confidence * 100)
    cov_pct = int(evidence_coverage * 100)
    
    if verdict == "COORDINATED_SYNTHETIC":
        lines.append(
            f"TrustLayer has determined with {conf_pct}% confidence that this set of artifacts represents a "
            f"COORDINATED SYNTHETIC campaign. Across {len(artifacts)} submitted artifacts ({cov_pct}% evidence coverage), "
            f"multiple cross-modal signals and entity contradictions were detected."
        )
    elif verdict == "MANIPULATED":
        lines.append(
            f"TrustLayer analyzed {len(artifacts)} artifact(s) with {conf_pct}% confidence and found evidence of "
            f"digital MANIPULATION localized within specific file elements."
        )
    elif verdict == "AUTHENTIC":
        lines.append(
            f"TrustLayer verified {len(artifacts)} artifact(s) as AUTHENTIC with {conf_pct}% confidence. "
            f"No deepfake signatures, metadata tampering, or entity contradictions were observed."
        )
    else: # INSUFFICIENT_EVIDENCE
        lines.append(
            f"TrustLayer evaluated the submitted artifacts and issued a verdict of INSUFFICIENT EVIDENCE ({conf_pct}% confidence). "
            f"The evidence coverage ({cov_pct}%) or signal strength was too low to make a conclusive authenticity determination."
        )

    # Key findings / Reasons
    if reasons:
        lines.append("\nKey Reasoning Factors:")
        for r in reasons:
            lines.append(f"• {r}")

    # Contradictions
    if contradictions:
        lines.append("\nDetected Contradictions:")
        for c in contradictions:
            sev = c.get("severity", "medium").upper()
            art_a = c.get("artifact_a", "")
            art_b = c.get("artifact_b", "")
            desc = c.get("description", "")
            if art_a and art_b and art_a != art_b:
                lines.append(f"• [{sev}] Conflict between {art_a} and {art_b}: {desc}")
            else:
                lines.append(f"• [{sev}] Artifact {art_a}: {desc}")

    # Entity relationships
    person_rels = [r for r in relationships if "shared_person" in r.get("relation", "")]
    if person_rels:
        lines.append("\nCross-Artifact Entity Matches:")
        for pr in person_rels:
            person_id = pr["relation"].split(":")[-1]
            lines.append(f"• Entity {person_id} tracked across artifacts {pr['artifact_a']} and {pr['artifact_b']}.")

    return "\n".join(lines)


def _generate_with_gemini(
    api_key: str,
    verdict: str,
    confidence: float,
    reasons: list[str],
    contradictions: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> str | None:
    """Optional call to Google Gemini API if library & key are present."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""You are the explanation engine for TrustLayer, a digital artifact authenticity system.
Synthesize a concise, professional 2-3 paragraph 'Why?' explanation for an investigation report.

Verdict: {verdict}
Confidence: {confidence:.2f}
Artifacts Analyzed: {len(artifacts)}
Key Reasons: {reasons}
Contradictions Found: {contradictions}
Entity Relationships: {relationships}

Explain clearly whether the pieces of the digital story agree with each other or show evidence of synthetic manipulation / coordination. Keep the tone analytical and precise.
"""
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else None
    except Exception as exc:
        logger.debug("Gemini API generation exception: %s", exc)
        return None
