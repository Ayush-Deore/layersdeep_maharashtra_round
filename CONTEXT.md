# TrustLayer — AI Analysis Backend · Context File

> **Purpose**: Short living context file for future agents/models to quickly understand project state.
> **Last updated**: Phase 2 revision — real-life image/video processing

---

## What This Project Is

**TrustLayer** analyzes digital artifacts (image, video, audio, doc) to determine authenticity.
Core thesis: _"Does the digital story add up?"_ — cross-modal evidence fusion, not single-file detection.

**This repo is Participant 1's track** — AI Analysis Backend only.

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| API | FastAPI (Python 3.14) |
| Image deepfake | `dima806/deepfake_vs_real_image_detection` (ViT, HuggingFace) |
| Face detection | InsightFace `buffalo_l` (ONNX, CPU) |
| Video frames | FFmpeg (subprocess) → frame-by-frame image analysis |
| Metadata | Pillow EXIF |
| URL download | httpx async client |
| Output contract | Evidence JSON (`/schemas/evidence.schema.json`) |

---

## Project Structure

```
bnb26-layersdeep-internalrounds/
├── PRD.md
├── CONTEXT.md                    ← this file
├── README.md
├── schemas/
│   ├── evidence.schema.json      ← shared output contract
│   └── verdict.schema.json       ← Participant 2's output contract
├── backend/
│   ├── main.py                   ← FastAPI app (lifespan warmup)
│   ├── requirements.txt
│   ├── routers/
│   │   └── analyze.py            ← POST /analyze/, /analyze/url, /analyze/mock, /analyze/status
│   ├── analyzers/
│   │   ├── image_analyzer.py     ← deepfake detection + face detection (real-life)
│   │   ├── video_analyzer.py     ← FFmpeg frame sampling → image analyzer loop
│   │   └── metadata_analyzer.py  ← EXIF, editing software signals, datetime mismatch
│   ├── models/
│   │   └── loader.py             ← lazy singleton + warmup_all() + model_status()
│   └── utils/
│       ├── file_utils.py         ← MIME detection, artifact IDs
│       └── face_registry.py      ← cosine-similarity person_id registry (cross-artifact)
├── frontend/
│   └── index.html                ← single-file validation UI (drag & drop, visualization)
└── mock_data/
    └── sample_evidence.json      ← 4-artifact mock for Participant 2
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | → redirect to `/docs` |
| `GET` | `/health` | Health + model load status |
| `POST` | `/analyze/` | **Upload files** — real-life analysis |
| `POST` | `/analyze/url` | **URL input** — download + analyze |
| `POST` | `/analyze/mock` | Mock evidence JSON (no upload) |
| `GET` | `/analyze/status` | Which models are loaded |

---

## Evidence JSON Shape (output)

```json
{
  "artifact_id": "img_a1b2c3",
  "type": "image",
  "filename": "photo.jpg",
  "mime_type": "image/jpeg",
  "evidence": [
    {
      "type": "person",
      "value": { "person_id": "person_01", "age": 38, "gender": "M", "bbox": [112,45,380,420] },
      "confidence": 0.97,
      "source_model": "insightface/buffalo_l"
    },
    {
      "type": "manipulation",
      "value": "synthetic_visual",
      "confidence": 0.88,
      "source_model": "dima806/deepfake_vs_real_image_detection"
    },
    {
      "type": "metadata",
      "value": { "size_bytes": 2847219, "make": "Canon", "gps_present": false },
      "confidence": 1.0,
      "source_model": "metadata_analyzer"
    }
  ],
  "analysis_errors": []
}
```

---

## Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **FaceRegistry per request** | All artifacts in one `/analyze/` call share a registry so `person_01` means the same person across images and video frames |
| **Cosine similarity face matching** | Threshold = 0.45 on InsightFace `normed_embedding`. Same identity → same `person_id` |
| **No raw embeddings in output** | 512 floats per face is unusable JSON. Registry maps embedding → stable ID |
| **Image resize to ≤1024px** | ViT runs on CPU; large images would OOM. Pillow LANCZOS downscale before inference |
| **Frame cap = 12** | Keeps CPU inference under ~2 min for long videos. Interval adapts to duration |
| **Robust label normalization** | HF model labels vary ("Fake"/"Real", "0"/"1", etc.) — all normalized to `fake`/`real`/`uncertain` |
| **Lifespan warmup** | Models pre-load in a thread at startup. First request is fast |
| **Graceful degradation** | If any model fails to load, that evidence type is skipped (not a crash) |

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold, schemas, context | ✅ Done |
| 1 | FastAPI app + upload + metadata | ✅ Done |
| 2 | Image analysis (deepfake + face) | ✅ Done (real-life revised) |
| 3 | Video analysis (FFmpeg frames) | ✅ Done (real-life revised) |
| 4 | Evidence assembly + /analyze | ✅ Done (real-life revised) |
| 5 | Mock data + README | ✅ Done |
| 6 | Real-life processing revision | ✅ Done |
| 7 | Frontend Validator (UI) | ✅ Done |

---

## Modalities

| Modality | Status | Notes |
|----------|--------|-------|
| Image deepfake detection | ✅ Active | ViT, CPU, images resized to ≤1024px |
| Face detection + person_id | ✅ Active | InsightFace buffalo_l, cosine similarity registry |
| Video frame sampling | ✅ Active | FFmpeg, max 12 frames, duration-adaptive interval |
| Metadata / EXIF | ✅ Active | Editing software detection, datetime mismatch |
| URL-based input | ✅ Active | httpx async download → same pipeline |
| Speech transcription | ⏳ P1 | Whisper |
| Speaker embeddings | ⏳ P1 | ECAPA-TDNN |
| Synthetic audio | ⏳ P1 | Wav2Vec2 |
| OCR | ⏳ P1 | PaddleOCR |

---

## Participant 2 Integration

- `POST /analyze/mock` → returns `mock_data/sample_evidence.json` immediately
- Person matching: same `person_id` across artifacts = same physical person
- Manipulation signal values: `synthetic_visual`, `authentic_visual`, `uncertain`, `editing_software_detected:*`, `datetime_mismatch`
- Video items include `frame_timestamp` (seconds) and `detail.frames_flagged`

---

## Decision Log (Internal)
* **2026-10-03 (Frontend Validator)**: Built a lightweight HTML/JS frontend to validate the API responses visually.
* **2026-10-03 (Bug Fix)**: Fixed a silent 500 JSON serialization error where `insightface` gender was returning `np.int64`. Casted it to `int`.
* **2026-10-03 (Model Limitation Mitigated)**: User reported that the deepfake detector (`dima806/ViT`) failed to catch a face-swap manipulation on a full image. **Decision**: Update `image_analyzer.py` to crop bounding boxes detected by InsightFace, and run the deepfake model specifically on those isolated face crops to prevent the face swap artifacts from being destroyed by the Vision Transformer's downscaling.
* **2026-10-04 (Fine-tuning infrastructure)**: Added a custom training script (`backend/training/train_vit.py`) using the HuggingFace `Trainer` API to allow fine-tuning the deepfake ViT model on custom, organization-specific datasets.
* **2026-10-04 (UI Previews & Video Sampling)**: Enabled inline image/video rendering in the UI via `URL.createObjectURL` to allow for side-by-side visual and metadata comparisons. Reduced video sampling interval to 3 seconds for higher granularity.
* **2026-10-04 (Compare Mode)**: Added **🖼️ Compare Images** and **📋 Compare Metadata** toggle buttons to the results panel (visible when 2+ artifacts are loaded). Clicking them reveals: (1) a 2-column visual comparison with per-image verdicts and top signals, and (2) a full metadata diff table where differing fields are highlighted in amber.
