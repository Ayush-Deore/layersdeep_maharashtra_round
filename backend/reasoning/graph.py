"""
TrustLayer — Evidence Graph Relationship Builder

Generates relationships between artifacts suitable for Evidence Graph visualization
and verdict.schema.json output.
"""

from typing import Any


def build_graph_relationships(entity_matches: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Returns list of relationship dicts:
    [
        {
            "artifact_a": "img_01",
            "artifact_b": "img_02",
            "relation": "shared_person:person_01",
            "confidence": 0.95
        }
    ]
    """
    relationships: list[dict[str, Any]] = []
    seen_rel_keys: set[tuple[str, str, str]] = set()

    person_map = entity_matches.get("person_map", {})
    camera_map = entity_matches.get("camera_map", {})

    # Person relationships
    for pid, art_ids in person_map.items():
        unique_arts = list(dict.fromkeys(art_ids))
        if len(unique_arts) >= 2:
            for i in range(len(unique_arts)):
                for j in range(i + 1, len(unique_arts)):
                    a1, a2 = unique_arts[i], unique_arts[j]
                    rel_key = tuple(sorted([a1, a2])) + (pid,)
                    if rel_key not in seen_rel_keys:
                        seen_rel_keys.add(rel_key)
                        relationships.append({
                            "artifact_a": a1,
                            "artifact_b": a2,
                            "relation": f"shared_person:{pid}",
                            "confidence": 0.95,
                        })

    # Camera relationships
    for cam_key, art_ids in camera_map.items():
        unique_arts = list(dict.fromkeys(art_ids))
        if len(unique_arts) >= 2:
            for i in range(len(unique_arts)):
                for j in range(i + 1, len(unique_arts)):
                    a1, a2 = unique_arts[i], unique_arts[j]
                    rel_key = tuple(sorted([a1, a2])) + (cam_key,)
                    if rel_key not in seen_rel_keys:
                        seen_rel_keys.add(rel_key)
                        relationships.append({
                            "artifact_a": a1,
                            "artifact_b": a2,
                            "relation": f"shared_camera:{cam_key}",
                            "confidence": 0.90,
                        })

    return relationships
