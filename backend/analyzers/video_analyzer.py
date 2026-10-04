"""
Video Analyzer — Real-life Processing (Phase 3 revised)

Key changes vs prototype:
- Frame cap: max MAX_FRAMES sampled regardless of video length
- FaceRegistry passed through so person_ids are consistent across frames
- Manipulation evidence deduplication replaced with per-frame timeline
- Duration-adaptive sampling interval
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_FRAMES = 12          # hard cap to keep CPU inference tractable
BASE_INTERVAL = 3        # default seconds between frames
MIN_INTERVAL = 2         # minimum even for short clips


def _get_video_duration(video_path: str) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _compute_interval(duration: Optional[float]) -> int:
    """Choose frame sample interval so we get ≤ MAX_FRAMES total."""
    if duration is None:
        return BASE_INTERVAL
    ideal = max(MIN_INTERVAL, int(duration / MAX_FRAMES))
    return ideal


def _extract_frames_ffmpeg(video_path: str, output_dir: str, interval: int) -> list[tuple[str, float]]:
    """
    Extract frames at `interval` seconds via FFmpeg.
    Returns list of (frame_path, timestamp_seconds).
    """
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        output_pattern,
        "-y", "-loglevel", "error",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    except FileNotFoundError:
        logger.warning("FFmpeg not found — falling back to OpenCV")
        return _extract_frames_opencv(video_path, output_dir, interval)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timed out extracting frames")
        return []

    frame_paths = sorted(Path(output_dir).glob("frame_*.jpg"))
    # frame_%04d starts at 1, so frame i (1-indexed) is at (i-1)*interval seconds
    result = []
    for j, fp in enumerate(frame_paths[:MAX_FRAMES]):
        ts = j * interval
        result.append((str(fp), float(ts)))
    return result


def _extract_frames_opencv(video_path: str, output_dir: str, interval: int) -> list[tuple[str, float]]:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(fps * interval))
        frames = []
        frame_idx = 0
        saved = 0
        while saved < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
                cv2.imwrite(out_path, frame)
                ts = frame_idx / fps
                frames.append((out_path, ts))
                saved += 1
            frame_idx += 1
        cap.release()
        return frames
    except Exception as exc:
        logger.warning("OpenCV frame extraction failed: %s", exc)
        return []


def analyze(path: str, registry=None) -> tuple[list[dict], list[str]]:
    """
    Full video analysis pipeline.
    - registry: FaceRegistry instance for cross-artifact person_id consistency
    Returns (evidence_list, error_list).
    """
    from analyzers import image_analyzer, metadata_analyzer

    all_evidence: list[dict] = []
    errors: list[str] = []

    duration = _get_video_duration(path)
    interval = _compute_interval(duration)
    logger.info("Video duration=%.1fs, sampling every %ds (max %d frames)",
                duration or 0, interval, MAX_FRAMES)

    with tempfile.TemporaryDirectory(prefix="trustlayer_video_") as tmp_dir:
        frames = _extract_frames_ffmpeg(path, tmp_dir, interval)

        if not frames:
            errors.append("video_analyzer: no frames could be extracted")
        else:
            logger.info("Analyzing %d frames from %s", len(frames), Path(path).name)

        fake_scores = []
        for frame_path, timestamp in frames:
            frame_ev, frame_err = image_analyzer.analyze(frame_path, registry=registry)
            for ev in frame_ev:
                ev["frame_timestamp"] = round(timestamp, 1)
                # Collect fake scores for aggregation
                if ev["type"] == "manipulation" and ev["value"] == "synthetic_visual":
                    fake_scores.append(ev["confidence"])
            all_evidence.extend(frame_ev)
            errors.extend(frame_err)

    # ── Manipulation summary ────────────────────────────────────────────────────
    if fake_scores:
        # Remove per-frame synthetic_visual items, replace with a summary
        all_evidence = [
            e for e in all_evidence
            if not (e["type"] == "manipulation" and e.get("value") == "synthetic_visual")
        ]
        all_evidence.append({
            "type": "manipulation",
            "value": "synthetic_visual",
            "confidence": round(max(fake_scores), 4),
            "source_model": "dima806/deepfake_vs_real_image_detection",
            "detail": {
                "frames_flagged": len(fake_scores),
                "frames_analyzed": len(frames),
                "avg_confidence": round(sum(fake_scores) / len(fake_scores), 4),
                "peak_confidence": round(max(fake_scores), 4),
            },
        })

    # ── Video metadata ──────────────────────────────────────────────────────────
    meta_evidence = metadata_analyzer.analyze(path)
    for ev in meta_evidence:
        if ev["type"] == "metadata" and duration is not None:
            ev["value"]["duration_seconds"] = round(duration, 2)
            ev["value"]["frames_analyzed"] = len(frames)
            ev["value"]["sample_interval_seconds"] = interval
    all_evidence.extend(meta_evidence)

    return all_evidence, errors
