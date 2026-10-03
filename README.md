# TrustLayer — AI Analysis Backend

> **Participant 1** · Branch: `feature/ai-analysis`

Analyzes digital artifacts (images, videos, documents) and emits structured **Evidence JSON** for the cross-modal reasoning layer.

---

## Architecture

```
Upload (multipart)
       ↓
POST /analyze
       ↓
   [per-file dispatcher]
       ↓
  ┌─────────────────────────────────────┐
  │  image: deepfake_detection + face   │
  │  video: FFmpeg frames → image loop  │
  │  all:   metadata extraction         │
  └─────────────────────────────────────┘
       ↓
Evidence JSON  ──→  Participant 2 (reasoning + dashboard)
```

---

## Setup

```bash
cd backend
pip install -r requirements.txt
```

> **FFmpeg** must be installed and on PATH for video frame extraction.
> Download: https://ffmpeg.org/download.html

---

## Run

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze/` | Analyze one or more files |
| `POST` | `/analyze/mock` | Return mock evidence (no upload needed) |

---

## POST /analyze — Usage

```bash
# Single image
curl -X POST http://localhost:8000/analyze/ \
  -F "files=@photo.jpg"

# Multiple files
curl -X POST http://localhost:8000/analyze/ \
  -F "files=@photo.jpg" \
  -F "files=@clip.mp4"
```

### Response (Evidence JSON)

```json
[
  {
    "artifact_id": "img_a1b2c3",
    "type": "image",
    "filename": "photo.jpg",
    "mime_type": "image/jpeg",
    "evidence": [
      { "type": "person", "value": { "face_index": 0, "age": 38 }, "confidence": 0.97, "source_model": "insightface/buffalo_l" },
      { "type": "manipulation", "value": "synthetic_visual", "confidence": 0.88, "source_model": "dima806/deepfake_vs_real_image_detection" },
      { "type": "metadata", "value": { "size_bytes": 2847219, "make": "Canon" }, "confidence": 1.0, "source_model": "metadata_analyzer" }
    ],
    "analysis_errors": []
  }
]
```

---

## Modalities

| Modality | Status | Models |
|----------|--------|--------|
| Image deepfake detection | ✅ P0 | `dima806/deepfake_vs_real_image_detection` (ViT) |
| Face detection & embedding | ✅ P0 | InsightFace `buffalo_l` |
| Video (frame sampling) | ✅ P0 | FFmpeg → image analyzer |
| Metadata (EXIF) | ✅ P0 | Pillow |
| Speech transcription | ⏳ P1 | Whisper |
| Speaker embeddings | ⏳ P1 | ECAPA-TDNN |
| Synthetic audio | ⏳ P1 | Wav2Vec2 |
| OCR | ⏳ P1 | PaddleOCR |

---

## Schemas

Shared contract lives in `/schemas/`:
- `evidence.schema.json` — output of this repo
- `verdict.schema.json` — input to Participant 2

**Do not break these schemas without coordinating with Participant 2.**

---

## For Participant 2 (Mock Integration)

Use `POST /analyze/mock` to get realistic Evidence JSON without needing real files.
Or directly load `mock_data/sample_evidence.json`.

---

## Project Structure

```
backend/
├── main.py                  # FastAPI app
├── requirements.txt
├── routers/
│   └── analyze.py           # POST /analyze
├── analyzers/
│   ├── image_analyzer.py    # ViT deepfake + InsightFace
│   ├── video_analyzer.py    # FFmpeg frames → image analyzer
│   └── metadata_analyzer.py # EXIF + file stats
├── models/
│   └── loader.py            # Lazy singleton model loading
└── utils/
    └── file_utils.py        # MIME detection, artifact IDs
```
# layersdeep_maharashtra_round
