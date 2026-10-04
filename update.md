
# TrustLayer — Forensic Engine Roadmap

### Baseline truths we do **not** change

1. **TrustLayer is not simply a deepfake classifier.**
2. It analyzes **multiple modalities**: image, video, audio, document.
3. Individual models are **evidence sensors**, not ground truth.
4. The final decision comes from **evidence fusion + cross-modal reasoning**.
5. Final states remain:
   - `AUTHENTIC`
   - `MANIPULATED`
   - `COORDINATED_SYNTHETIC`
   - `INSUFFICIENT_EVIDENCE`
6. Every verdict must have **evidence + explanation**.
7. **Confidence ≠ Evidence Coverage.**
8. The system must be able to say **“I don't have enough evidence.”**
9. Production inference should **not require the original image**.
10. Paired real/manipulated data is primarily for **training, calibration and evaluation**.

---

# Phase 1 — Establish the Forensic Evidence Layer

Replace:

```text
Image
 ↓
Deepfake model
 ↓
Fake %
```

with:

```text
Image
 ↓
Forensic Analysis
 ↓
Multiple independent signals
```

Initial signals:

```text
Face manipulation
Background manipulation
Synthetic-generation likelihood
Image integrity
Metadata anomalies
```

Output:

```json
{
  "face_manipulation": 0.04,
  "background_manipulation": 0.91,
  "synthetic_visual": 0.83,
  "metadata_anomaly": 0.72
}
```

**Goal:** never allow one detector's score to represent the entire artifact.

---

# Phase 2 — Build a Small Purpose-Built Dataset

Start with paired examples:

```text
original
   ↕
manipulated version
```

Cover manipulation **types**, not just fake/real:

```text
AUTHENTIC
BACKGROUND_REPLACEMENT
FACE_SWAP
OBJECT_INPAINTING
IMAGE_GENERATION
COMPOSITE_IMAGE
```

For each example store:

```json
{
  "source_id": "img_001",
  "label": "BACKGROUND_REPLACEMENT",
  "regions": ["background"],
  "severity": 0.9
}
```

Your own original + Eiffel Tower manipulation becomes one of the **golden test cases**, not the entire training dataset.

---

# Phase 3 — Evaluate Existing Models Before Fine-Tuning

For every candidate detector:

```text
Dataset
 ↓
Model
 ↓
Prediction
 ↓
Precision / Recall / F1
 ↓
Calibration
```

Create a simple benchmark:

| Manipulation | Detector performance |
|---|---:|
| Face swap | ? |
| Background replacement | ? |
| Inpainting | ? |
| Fully synthetic | ? |
| Authentic | ? |

**Decision rule:**

If an existing model performs poorly on your target manipulation, **don't blindly fine-tune it**. First determine whether it was designed for that manipulation.

---

# Phase 4 — Add Region-Aware Analysis

This is probably the biggest upgrade.

Instead of:

```text
IMAGE → 87% FAKE
```

aim for:

```text
IMAGE
 │
 ├── Face → authentic
 │
 ├── Body → authentic
 │
 ├── Background → highly suspicious
 │
 └── Metadata → suspicious
```

Then TrustLayer can explain:

> **The subject appears consistent with an authentic photograph, but the surrounding environment shows strong manipulation indicators.**

This is much more defensible.

---

# Phase 5 — Evidence Fusion

Create a dedicated fusion layer.

```text
Face evidence
Background evidence
Metadata evidence
Generation evidence
       ↓
 Evidence Fusion
       ↓
 Manipulation Assessment
```

Do **not** use:

```text
(face + background + metadata) / 3
```

Instead, make the fusion layer understand:

- signal reliability
- conflicting signals
- manipulation type
- evidence coverage
- severity

Example:

```json
{
  "assessment": {
    "manipulation_probability": 0.92,
    "confidence": 0.89,
    "evidence_coverage": 0.76
  }
}
```

---

# Phase 6 — Preserve the Multimodal Architecture

Once image analysis works:

```text
IMAGE
 ↓
Image Evidence
```

becomes:

```text
IMAGE ──→ Image Evidence
VIDEO ──→ Video Evidence
AUDIO ──→ Audio Evidence
DOC ────→ Document Evidence
              ↓
        Evidence Graph
              ↓
       Cross-modal reasoning
```

Example:

```text
Image:
Person = Ayush
Location = Paris
       │
       ├─────────────┐
       ↓             ↓
Audio           Document
Location=Pune   Location=Paris
       │
       ↓
CONTRADICTION
```

This preserves your original **TrustLayer differentiator**.

---

# Phase 7 — Calibration + “Insufficient Evidence”

Before finalizing the verdict system, explicitly calibrate confidence.

Your system should be able to produce:

```text
MANIPULATED
Confidence: 93%
Evidence Coverage: 81%
```

but also:

```text
INSUFFICIENT EVIDENCE
Confidence: 41%
Evidence Coverage: 19%
```

This prevents the model from confidently hallucinating certainty.

---

# Phase 8 — Final TrustLayer API

Keep your existing integration contract.

### `/analyze`

```text
artifact
 ↓
modality-specific forensic analysis
 ↓
Evidence JSON
```

### `/investigate`

```text
Evidence JSON
 ↓
fusion
 ↓
cross-modal reasoning
 ↓
Verdict JSON
```

The frontend shouldn't care which detector produced the evidence.

That means you can swap:

```text
Detector A
```

for:

```text
Detector B
```

without rebuilding TrustLayer.

---

# Final architecture

```text
                 ARTIFACTS
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        Image      Audio       Video
          │          │          │
          ↓          ↓          ↓
    Forensic      Forensic    Forensic
     Signals       Signals     Signals
          │          │          │
          └──────────┼──────────┘
                     ↓
              EVIDENCE JSON
                     ↓
              EVIDENCE FUSION
                     ↓
             EVIDENCE GRAPH
                     ↓
          CROSS-MODAL REASONING
                     ↓
             TRUST ASSESSMENT
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
    VERDICT      CONFIDENCE     COVERAGE
       │
       ↓
   EXPLANATION
```

## Hackathon priority order

**P0 — Must work**

1. Background manipulation detection
2. Face manipulation detection
3. Evidence JSON
4. Evidence fusion
5. Correct `MANIPULATED` verdict
6. Explanation

**P1 — Strong differentiator**

7. Region-aware evidence
8. Metadata/provenance
9. Audio/video analysis
10. Cross-modal contradictions
11. Evidence graph

**P2 — Only if time remains**

12. Fine-tuning
13. More manipulation classes
14. Advanced calibration
15. Unseen-generator evaluation

### The guiding principle

> **Don't train TrustLayer to recognize “fake.” Train the system to recognize and combine evidence of manipulation.**

That keeps everything you've already designed intact while fixing the exact weakness you're seeing now.