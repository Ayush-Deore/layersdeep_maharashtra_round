
# TrustLayer — Current Capabilities

TrustLayer's AI Analysis Backend is currently operational and actively processing multi-modal digital artifacts. Here is a summary of what the system can do right now:

## 1. Supported Modalities & Models

*   **Image Analysis (ViT Deepfake Detection)**: 
    *   Powered by `dima806/deepfake_vs_real_image_detection` (HuggingFace).
    *   **Face-Crop Analysis**: Extracts bounding boxes via InsightFace and runs the deepfake detector specifically on isolated face crops to prevent artifacts from being destroyed by downscaling.
    *   **Background Noise Detection**: Uses Laplacian variance to detect synthetically smooth backgrounds versus natural camera noise (`synthetic_background` vs `authentic_background`).
    *   Automatically resizes large images to prevent out-of-memory errors.
    *   Normalizes labels across different vocabularies to definitively output `synthetic_visual`, `authentic_visual`, or `uncertain`.
*   **Face Detection & Recognition**:
    *   Powered by InsightFace (`buffalo_l`).
    *   Extracts bounding boxes, age, and gender.
    *   **Cross-Artifact Matching**: Uses a session-scoped `FaceRegistry` with cosine similarity to assign stable IDs (e.g., `person_01`). If the same person appears in a photo and a video, they get the exact same ID.
*   **Video Processing**:
    *   Intelligently samples frames using `ffmpeg` (with an OpenCV fallback).
    *   Applies a hard cap of 12 frames per video (using duration-adaptive intervals) to keep inference fast.
    *   Aggregates frame-by-frame deepfake signals into a single video-level confidence score.
*   **Metadata & EXIF Extraction**:
    *   Extracts file size, type, camera make/model, and GPS presence.
    *   Flags suspicious signals, such as known photo editing software or mismatched datetime signatures.

## 2. API & Infrastructure

*   **File Uploads (`POST /analyze/`)**: Accepts drag-and-drop batch uploads (images and videos) and processes them simultaneously under a single session.
*   **URL Processing (`POST /analyze/url`)**: Can download and analyze public images and videos directly from a provided link via async `httpx`.
*   **Zero Cold-Start**: Models are pre-warmed in a background thread on server startup, ensuring the first request is immediately responsive.
*   **Custom Fine-Tuning**: Provides a PyTorch/HuggingFace script (`train_vit.py`) to easily fine-tune the baseline ViT deepfake model on organization-specific or highly sophisticated deepfake datasets.

## 3. Frontend Validator UI

*   A sleek, dark-mode, glassmorphism UI running locally to visualize the API's JSON output.
*   Displays real-time server health and model load status.
*   Visualizes manipulation confidence bars (Red/Yellow/Green), person detection badges, and expandable metadata accordions.
*   Includes a mock-data injection mode for instant UI testing.
