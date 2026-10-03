"""
POST /investigate — reasoning & verdict endpoint for Participant 2

Accepts Evidence JSON array (output of /analyze/ or mock data) and executes:
- Scorer (artifact-level manipulation scoring)
- Matcher (entity matching & relationship mapping)
- Contradictions (single-artifact & cross-artifact conflict detection)
- Graph (relationship builder for Evidence Graph)
- Verdict classification & confidence scoring
- Explanation synthesis

Returns Verdict JSON conforming to /schemas/verdict.schema.json.
"""

import logging
import os
import json
from typing import Any, Union
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from reasoning.engine import analyze_evidence

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/investigate", tags=["Reasoning"])


@router.post("/", summary="Generate trust verdict from Evidence JSON")
async def investigate(
    payload: Union[list[dict[str, Any]], dict[str, Any]] = Body(
        ...,
        description="Evidence JSON array (from POST /analyze/) or object containing 'artifacts' array",
    )
):
    """
    Consumes Evidence JSON for multiple digital artifacts and produces a complete **Trust Verdict**.
    
    Output matches `/schemas/verdict.schema.json`.
    """
    if isinstance(payload, dict):
        artifacts = payload.get("artifacts", [payload])
    elif isinstance(payload, list):
        artifacts = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be a list of Evidence objects or a dict.")

    try:
        verdict_result = analyze_evidence(artifacts)
        return JSONResponse(content=verdict_result)
    except Exception as exc:
        logger.exception("Reasoning engine execution failed")
        raise HTTPException(status_code=500, detail=f"Reasoning engine error: {exc}")


@router.post("/mock", summary="Run reasoning engine on sample_evidence.json")
async def investigate_mock():
    """
    Runs the cross-modal reasoning engine directly against `mock_data/sample_evidence.json`.
    """
    mock_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "mock_data", "sample_evidence.json")
    )
    if not os.path.exists(mock_path):
        raise HTTPException(status_code=404, detail="mock_data/sample_evidence.json not found")
    
    with open(mock_path, "r", encoding="utf-8") as f:
        sample_evidence = json.load(f)

    verdict_result = analyze_evidence(sample_evidence)
    return JSONResponse(content=verdict_result)


@router.post("/fixture/{fixture_name}", summary="Run reasoning engine on a specific demo fixture")
async def investigate_fixture(fixture_name: str):
    """
    Run reasoning engine on demo fixtures: 'authentic', 'manipulated', 'insufficient', 'evidence' (coordinated).
    """
    fname = fixture_name.lower().strip()
    file_map = {
        "authentic": "sample_authentic.json",
        "manipulated": "sample_manipulated.json",
        "insufficient": "sample_insufficient.json",
        "coordinated": "sample_evidence.json",
        "evidence": "sample_evidence.json",
    }
    filename = file_map.get(fname)
    if not filename:
        raise HTTPException(status_code=400, detail=f"Unknown fixture '{fixture_name}'. Valid: {list(file_map.keys())}")

    path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "mock_data", filename)
    )
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Fixture file '{filename}' not found")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    verdict_result = analyze_evidence(data)
    return JSONResponse(content=verdict_result)
