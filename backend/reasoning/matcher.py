"""
TrustLayer — Entity & Metadata Matcher

Extracts and clusters shared entities across digital artifacts using stable person_ids
and camera make/model metadata.
"""

from typing import Any


def match_entities(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Returns matched entity groupings:
    - person_map: person_id -> list of artifact_ids
    - camera_map: camera_key -> list of artifact_ids
    - entity_clusters: summary list of entity match objects
    """
    person_map: dict[str, list[str]] = {}
    camera_map: dict[str, list[str]] = {}

    for artifact in artifacts:
        art_id = artifact.get("artifact_id", "unknown")
        evidence_list = artifact.get("evidence", [])

        for ev in evidence_list:
            ev_type = ev.get("type")
            ev_val = ev.get("value")

            if ev_type == "person" and isinstance(ev_val, dict):
                pid = ev_val.get("person_id")
                if not pid and "face_index" in ev_val:
                    pid = f"person_{int(ev_val['face_index']) + 1:02d}"

                if pid:
                    if art_id not in person_map.setdefault(pid, []):
                        person_map[pid].append(art_id)

            elif ev_type == "metadata" and isinstance(ev_val, dict):
                make = ev_val.get("make")
                model = ev_val.get("model")
                if make or model:
                    cam_key = f"{make or ''} {model or ''}".strip()
                    if cam_key:
                        if art_id not in camera_map.setdefault(cam_key, []):
                            camera_map[cam_key].append(art_id)

    entity_clusters = []
    for pid, art_ids in person_map.items():
        if len(art_ids) >= 2:
            entity_clusters.append({
                "entity_type": "person",
                "entity_id": pid,
                "artifacts": art_ids,
                "count": len(art_ids),
            })

    for cam_key, art_ids in camera_map.items():
        if len(art_ids) >= 2:
            entity_clusters.append({
                "entity_type": "camera",
                "entity_id": cam_key,
                "artifacts": art_ids,
                "count": len(art_ids),
            })

    return {
        "person_map": person_map,
        "camera_map": camera_map,
        "entity_clusters": entity_clusters,
    }
