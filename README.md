# TrustLayer — Multi-Modal Deepfake Analysis & Cross-Modal Reasoning Platform

> **Participant 1 & Participant 2** · Branch: `anushka-work`

TrustLayer analyzes digital artifacts (images, videos, documents) and evaluates whether a multi-artifact digital story is **AUTHENTIC, MANIPULATED, COORDINATED SYNTHETIC, or INSUFFICIENT EVIDENCE**.

---

## Architecture

```
Upload (files or URL)
       ↓
POST /analyze
       ↓
  ┌─────────────────────────────────────────┐
  │  image: ViT deepfake + face detection   │
  │  video: FFmpeg frames → image loop      │
  │  metadata: EXIF, software & timestamps  │
  └─────────────────────────────────────────┘
       ↓
Structured Evidence JSON
       ↓
POST /investigate
       ↓
  ┌───────────────────────────────────────────────────────────┐
  │  Modular Reasoning Package (backend/reasoning/)           │
  │  • scorer.py        Artifact manipulation scoring         │
  │  • matcher.py       Entity & camera matching (person_id)  │
  │  • contradictions.py Single-artifact & cross-media conflicts│
  │  • graph.py         Evidence graph relationships          │
  │  • engine.py        Verdict classification & orchestration│
  │  • llm_explainer.py Rationale & "Why?" synthesis          │
  └───────────────────────────────────────────────────────────┘
       ↓
Trust Verdict JSON & Interactive Dashboard
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

## Run Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Backend status & model pre-warm status |
| `POST` | `/analyze/` | Upload one or more image/video files for analysis |
| `POST` | `/analyze/url` | Download and analyze an artifact from a public URL |
| `POST` | `/analyze/mock` | Return realistic mock Evidence JSON |
| `GET` | `/analyze/status` | Check AI model load state |
| `POST` | `/investigate/` | Run cross-modal reasoning on Evidence JSON → returns Verdict JSON |
| `POST` | `/investigate/mock` | Run reasoning directly on `sample_evidence.json` |
| `POST` | `/investigate/fixture/{name}` | Run reasoning on demo fixtures: `authentic`, `manipulated`, `insufficient`, `coordinated` |

---

## Usage

### 1. Extract Evidence (`POST /analyze/`)

```bash
curl -X POST http://localhost:8000/analyze/ \
  -F "files=@photo.jpg" \
  -F "files=@clip.mp4"
```

### 2. Generate Trust Verdict (`POST /investigate/`)

```bash
curl -X POST http://localhost:8000/investigate/ \
  -H "Content-Type: application/json" \
  -d '@mock_data/sample_evidence.json'
```

### Verdict Response Example

```json
{
  "verdict": "COORDINATED_SYNTHETIC",
  "confidence": 0.93,
  "evidence_coverage": 1.0,
  "reasons": [
    "Entity 'person_01' recognized across 3 artifacts (img_a1b2c3, img_d4e5f6, vid_g7h8i9).",
    "Coordinated synthetic campaign detected across 3 artifact(s) involving cross-artifact entity/media relationships."
  ],
  "contradictions": [
    {
      "artifact_a": "img_d4e5f6",
      "artifact_b": "img_d4e5f6",
      "description": "EXIF metadata reveals digital editing software (Adobe Photoshop 25.0) used on img_d4e5f6.",
      "severity": "medium"
    },
    {
      "artifact_a": "vid_g7h8i9",
      "artifact_b": "img_d4e5f6",
      "description": "Coordinated synthetic manipulation: Entity 'person_01' is manipulated across multiple media files (vid_g7h8i9, img_d4e5f6).",
      "severity": "high"
    }
  ],
  "relationships": [
    {
      "artifact_a": "img_a1b2c3",
      "artifact_b": "img_d4e5f6",
      "relation": "shared_person:person_01",
      "confidence": 0.95
    }
  ],
  "artifact_scores": {
    "img_d4e5f6": {
      "manipulation_score": 0.88,
      "has_synthetic": true,
      "explanation": "Synthetic visual manipulation detected with 88% confidence."
    }
  },
  "llm_explanation": "TrustLayer has determined with 93% confidence that this set of artifacts represents a COORDINATED SYNTHETIC campaign..."
}
```

---

## Modalities & AI Models

| Modality | Status | Models |
|----------|--------|--------|
| Image deepfake detection | ✅ Active | `dima806/deepfake_vs_real_image_detection` (ViT) |
| Face detection & person_id | ✅ Active | InsightFace `buffalo_l` + session `FaceRegistry` |
| Face-crop deepfake scan | ✅ Active | Bounding-box crop → ViT deepfake scan |
| Video frame sampling | ✅ Active | FFmpeg (max 12 frames) → image analyzer |
| Metadata & EXIF analysis | ✅ Active | Pillow EXIF, software & datetime mismatch detection |
| Cross-modal reasoning | ✅ Active | Modular `backend/reasoning/` (scorer, matcher, contradictions, graph, engine) |
| Explanation synthesis | ✅ Active | Local fallback engine & optional Gemini API |

---

## Demo Fixtures

Deterministic fixtures available for demonstrating all 4 verdicts:
- `sample_evidence.json` → `COORDINATED_SYNTHETIC`
- `sample_authentic.json` → `AUTHENTIC`
- `sample_manipulated.json` → `MANIPULATED`
- `sample_insufficient.json` → `INSUFFICIENT_EVIDENCE`

---

## Schemas

Shared data contracts in `/schemas/`:
- `evidence.schema.json` — output of `/analyze/`
- `verdict.schema.json` — output of `/investigate/`

---

## Frontend Trust Dashboard

Open `frontend/index.html` in any web browser (or via HTTP server at `http://localhost:5500`) to access the redesigned SaaS UI dashboard:
- **Visual Design**: Premium SaaS AI aesthetics with soft blue-gray background (`#EEF2F7`), clean white surfaces (`#FFFFFF`), navy typography (`#101828`), and muted periwinkle blue primary accents (`#5B6FD8`).
- **Batch Upload & URL Analysis**: Drag-and-drop file upload zone and public URL input tabs.
- **Model Status Indicator**: Real-time polling of backend `/health` showing model readiness.
- **Trust Verdict Banner**: Prominent verdict badge (`AUTHENTIC`, `MANIPULATED`, `COORDINATED_SYNTHETIC`, `INSUFFICIENT_EVIDENCE`), confidence score, and evidence coverage metrics.
- **Artifact Manipulation Scores**: Independent per-artifact manipulation scores breakdown.
- **Reasoning & "Why?" Explanation**: Detailed natural language synthesis and rationale factors.
- **Contradiction Cards**: Severity badges (`HIGH`, `MEDIUM`, `LOW`) with artifact conflict descriptions.
- **Cross-Artifact Evidence Graph**: Interactive network graph widget displaying shared entities (e.g. `SHARED PERSON: person_01`) at the apex with smooth Bezier curve connections to individual artifact cards (`press_photo.jpg`, `rally_crowd.jpg`, `speech_clip.mp4`), featuring hover highlight triggers for connected nodes and relationship links.
- **Evidence Cards & Raw JSON**: Fine-grained manipulation scores, face badges, EXIF accordions, and dark-themed raw JSON toggle.


---

## Project Structure

```
backend/
├── main.py                  # FastAPI app (warmup & router setup)
├── requirements.txt
├── routers/
│   ├── analyze.py           # POST /analyze/, /analyze/url, /analyze/mock
│   └── investigate.py       # POST /investigate/, /investigate/mock, /investigate/fixture/{name}
├── reasoning/
│   ├── __init__.py
│   ├── scorer.py            # Artifact manipulation scoring
│   ├── matcher.py           # Entity & metadata matching (person_id)
│   ├── contradictions.py    # Single-artifact & cross-media conflict detection
│   ├── graph.py             # Evidence graph relationship builder
│   ├── engine.py            # Main verdict decision & orchestration
│   └── llm_explainer.py     # Rationale & "Why?" synthesis
├── analyzers/
│   ├── image_analyzer.py    # ViT deepfake + InsightFace + face crops
│   ├── video_analyzer.py    # FFmpeg frame sampling → image analyzer loop
│   └── metadata_analyzer.py # EXIF, software & datetime mismatch
├── models/
│   └── loader.py            # Lazy singleton model loader
└── utils/
    ├── face_registry.py     # Cosine similarity person_id registry
    └── file_utils.py        # MIME detection & artifact IDs
frontend/
└── index.html               # Single-file HTML/CSS/JS Trust Dashboard
mock_data/
├── sample_evidence.json     # Coordinated synthetic sample dataset
├── sample_authentic.json    # Authentic sample dataset
├── sample_manipulated.json  # Manipulated sample dataset
└── sample_insufficient.json # Insufficient evidence sample dataset
schemas/
├── evidence.schema.json     # Output contract for analysis
└── verdict.schema.json      # Output contract for reasoning
```
