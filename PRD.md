# TrustLayer — Ultra-Lean PRD

## 1. Product

**TrustLayer** analyzes multiple digital artifacts—images, videos, audio, and documents—to determine whether they are **authentic, manipulated, coordinated synthetic, or inconclusive**.

### Core differentiator

```
Files
 ↓
Modality Analysis
 ↓
Evidence
 ↓
Cross-Modal Reasoning
 ↓
Verdict + Explanation
```

We are **not training new models**. Pretrained models generate evidence; our core product is **evidence fusion and cross-modal reasoning**.

---

## 2. MVP Components

### Input

- Image
- Video
- Audio
- Document

### Analysis

- Image/video manipulation signals
- Face detection / matching
- OCR
- Metadata
- Speech transcription
- Speaker matching
- Synthetic audio signals

### Reasoning

- Entity matching
- Time/location consistency
- Audio-video consistency
- Contradiction detection
- Coordinated manipulation detection
- LLM-assisted reasoning

### Output

```
Verdict:
AUTHENTIC
MANIPULATED
COORDINATED_SYNTHETIC
INSUFFICIENT_EVIDENCE

+ confidence
+ evidence coverage
+ reasons
+ relationships
```

---

# 3. Pipeline

```
                UPLOAD
                  │
                  ▼
          ┌───────────────┐
          │   ANALYSIS    │
          └───────┬───────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
    IMAGE       VIDEO      AUDIO/DOC
      │           │           │
      └───────────┼───────────┘
                  ▼
            EVIDENCE JSON
                  │
                  ▼
         CROSS-MODAL ENGINE
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
    Matching  Conflicts  Coordination
        │         │         │
        └─────────┼─────────┘
                  ▼
            TRUST VERDICT
                  │
                  ▼
          DASHBOARD + WHY
```

---

# 4. Shared Data Contract

This is the **only major contract between the two participants**.

### Evidence input

```
{
  "artifact_id": "img_01",
  "type": "image",
  "evidence": [
    {
      "type": "person",
      "value": "person_01",
      "confidence": 0.94
    },
    {
      "type": "location",
      "value": "Pune",
      "confidence": 0.82
    },
    {
      "type": "manipulation",
      "value": "synthetic_visual",
      "confidence": 0.81
    }
  ]
}
```

### Verdict output

```
{
  "verdict": "COORDINATED_SYNTHETIC",
  "confidence": 0.87,
  "evidence_coverage": 0.78,
  "reasons": [],
  "contradictions": [],
  "relationships": []
}
```

---

# 5. Participant 1 — Analysis + AI Backend

### Branch

```
feature/ai-analysis
```

### Owns

```
Upload
  ↓
Preprocessing
  ↓
Image / Video / Audio / Document analysis
  ↓
Evidence JSON
```

### Build

- FastAPI
- FFmpeg/OpenCV
- OCR
- Face analysis
- Whisper
- Speaker embeddings
- Deepfake/synthetic-content detectors
- Metadata extraction
- `/analyze` API

### Suggested models

- **InsightFace** → faces
- **PaddleOCR** → OCR
- **Whisper** → transcription
- **ECAPA-TDNN** → speaker embeddings
- **ViT/deepfake detector** → image/video manipulation
- **Pella/Wav2Vec2 detector** → synthetic audio

### Definition of Done

```
Input file
   ↓
POST /analyze
   ↓
Structured Evidence JSON
```

Participant 1 **does not build the final verdict or frontend**.

---

# 6. Participant 2 — Reasoning + Frontend

### Branch

```
feature/trust-dashboard
```

### Owns

```
Evidence JSON
      ↓
Cross-modal reasoning
      ↓
Verdict
      ↓
Dashboard
```

### Build

**Reasoning**

- Evidence normalization
- Entity matching
- Contradiction detection
- Temporal consistency
- Location consistency
- Cross-modal reasoning
- LLM explanation
- Confidence/evidence coverage

**Frontend**

- Upload interface
- Artifact cards
- Evidence graph
- Contradiction display
- Verdict
- "Why?" explanation

### Definition of Done

```
Evidence JSON
     ↓
POST /investigate
     ↓
Verdict JSON
     ↓
Dashboard
```

Participant 2 should use **mock Evidence JSON immediately**, so development doesn't wait for Participant 1.

---

# 7. GitHub

```
main
│
├── feature/ai-analysis
└── feature/trust-dashboard
```

Shared:

```
/schemas
    evidence.schema.json
    verdict.schema.json
```

### Workflow

```
Agree schemas
      ↓
Work independently
      ↓
Mock APIs
      ↓
First integration
      ↓
End-to-end testing
      ↓
main
```

Don't continuously merge each other's branches.

---

# 8. 24-Hour Priority

### P0 — Absolutely required

- Upload
- At least 2–3 modalities working reliably
- Evidence JSON
- Cross-modal contradiction detection
- Trust verdict
- Explanation
- Basic dashboard

### P1 — If time permits

- All 4 modalities
- Evidence graph
- Speaker matching
- Better visualizations
- Generalization test

### P2 — Ignore for hackathon

- Training custom foundation models
- Real-time monitoring
- Blockchain
- Social-media crawling
- Enterprise authentication
- Perfect forensic guarantees

---

# 9. Demo

Prepare **4 fixed cases**:

```
1. AUTHENTIC
   Everything agrees.

2. MANIPULATED
   One artifact shows suspicious evidence.

3. COORDINATED SYNTHETIC
   Multiple artifacts look plausible individually,
   but their evidence conflicts.

4. INSUFFICIENT EVIDENCE
   System explicitly refuses to overclaim.
```

### The final demo story

```
Upload 4 artifacts
       ↓
Individual analysis
       ↓
Evidence extracted
       ↓
Relationships discovered
       ↓
Contradiction detected
       ↓
🔴 COORDINATED SYNTHETIC
       ↓
"Why?"
       ↓
Show the exact evidence
```

## Product thesis

> **TrustLayer doesn't decide whether one file is fake. It determines whether the pieces of a digital story agree with each other.**

---

# 10. Update — Real-Life Processing Revision (2026-10-03)

**Participant 1 backend upgraded from prototype to real-life processing.**

### Changes

**Image analysis**
- Images auto-resized to ≤1024px before ViT inference (prevents OOM on 4K inputs)
- Label normalization layer handles all HuggingFace model label vocabularies (`Fake`/`Real`, `0`/`1`, `deepfake`/`genuine`, etc.)
- Emits `authentic_visual` evidence (not just `synthetic_visual`) so the reasoning layer has both sides of the signal
- Emits `uncertain` when model confidence < 50% on either class

**Face detection & cross-artifact matching**
- Raw 512-float embeddings no longer appear in the Evidence JSON
- Introduced **FaceRegistry**: cosine-similarity (threshold 0.45) maps embeddings → stable `person_id` strings (`person_01`, `person_02`, …)
- All artifacts in a single `/analyze/` request share one registry → same person gets same ID across images and video frames
- Participant 2 can directly use `person_id` for entity matching without touching embeddings

**Video analysis**
- Hard cap: maximum 12 frames per video (duration-adaptive interval keeps this constant)
- Fake-frame aggregation: per-frame signals collapsed into one summary with `detail.frames_flagged`, `detail.avg_confidence`, `detail.peak_confidence`
- Video metadata includes `frames_analyzed`, `sample_interval_seconds`

**New endpoints**
- `POST /analyze/url` — paste a public image/video URL, get Evidence JSON (uses httpx async download)
- `GET /analyze/status` — check which models are currently loaded
- `GET /health` — now includes `models` dict with per-model load state

**Startup**
- Models pre-load via `asynccontextmanager` lifespan on server start (non-blocking thread pool)
- First real request no longer triggers 30s cold-start download

---

# 11. Update — Frontend Validator (2026-10-03)

**Added a lightweight, single-file frontend to visually validate the AI analysis backend.**

### Features
- **File Upload & URL Support**: Drag-and-drop zone for multiple files and a URL input tab.
- **Model Status Indicator**: Polls `/health` to show realtime model loading status (red/yellow/green).
- **Evidence Visualization**:
  - Displays manipulation bars with confidence (red/yellow/green).
  - Person badges with stable cross-artifact IDs, bounding boxes, age, and gender.
  - Expandable accordion for full EXIF/file metadata.
- **Mock Integration**: One-click "Load Mock Evidence" button to test the UI against `POST /analyze/mock`.
- **Raw JSON Toggle**: Easily inspect the raw Evidence JSON emitted by the backend.
- **Summary Dashboard**: High-level artifact count, total evidence, unique persons, and synthetic flags summary.

---

# 12. Update — Face-Cropping for Deepfake Detection (2026-10-03)

**Addressed a limitation where the `dima806/ViT` model failed to detect face-swaps on full images due to extreme downscaling.**

### Changes
- Modified `image_analyzer.py` to extract bounding boxes from `InsightFace`.
- Instead of just running deepfake detection on the full image, the pipeline now crops each detected face and runs the ViT model specifically on the face crop.
- This ensures that pixel-level face-swap artifacts are preserved and not destroyed when the ViT model internally downscales the image to 224x224.