# Changelog
All notable changes to the SIGNBRIDGE project are documented here.

## [2.0.0] - Ultra Legendary Edition - 2026-11-11

### Architectural Overhaul & Triple-Mode Input
- **Mode A (Video Upload)**: Implemented end-to-end video ingestion supporting MP4, WebM, MOV, and AVI up to 50MB. Added uniform temporal keyframe sampling mapping variable length video sequences into 30 canonical frames.
- **Mode B (Live Camera Fallback)**: Added hardware detection via `navigator.mediaDevices.enumerateDevices()`. Gracefully identifies missing camera hardware on desktop workstations and guides users to alternative modes.
- **Mode C (Interactive Demonstration)**: Built a simulation mode badged as `DEMONSTRATION / SIMULATED` for exhibition and lab demonstration without hardware requirements.

### Machine Learning & Computer Vision
- **MediaPipe Tasks Upgrade**: Transitioned from legacy `mp.solutions` to modern `mediapipe.tasks.python.vision` (`HandLandmarker` and `PoseLandmarker`) with pre-packaged neural task models (`hand_landmarker.task`, `pose_landmarker.task`).
- **Feature Pipeline**: Refined 153-dimensional coordinate feature normalization using shoulder-span Euclidean distance scaling and anatomical origin translation.
- **Practice Video Evaluation**: Added server-side `/api/practice/video` endpoint allowing users to record or upload attempts and receive BiLSTM similarity grades.

### Backend & API
- **New Endpoints**:
  - `POST /api/predict/video`: Processes uploaded video files with transient storage and instant purging.
  - `POST /api/practice/video`: Evaluates user practice video against canonical ISL templates.
  - `GET /api/research/metrics`: Exposes real-time benchmark KPIs, synthetic vs. prospective metrics, and per-class performance tables.
  - `GET /api/health`: Enhanced with V2 engine status, task models verification, and supported input modes.
- **Automated Test Suite**: Added comprehensive test coverage in `backend/tests/test_api.py` covering health, vocabulary, models, frame prediction, video prediction, practice evaluation, and research metrics.

### Frontend & UX
- **Translator Workspace**: Built dedicated 3-column unified workspace with mode switching tabs, video upload drop zone, live camera canvas, and demo simulator.
- **Research Dashboard**: Added interactive science dashboard with live KPI cards, confusion matrix visualization, and methodology documentation.
- **Privacy & Sovereignty Page**: Authored privacy charter detailing zero cloud retention, instant file purging, and local-only session storage.
- **History Enhancements**: Added mode origin badges (`Upload`, `Camera`, `Demo`) and individual record deletion.

### Infrastructure & Deployment
- Added production `Dockerfile` with multi-stage build.
- Added `docker-compose.yml` for unified single-port deployment.
- Created `docs/DEPLOYMENT.md`, `docs/API_REFERENCE.md`, `docs/PRIVACY.md`, and `research/REAL_WORLD_VALIDATION.md`.

---

## [1.0.0] - Initial Exhibition Baseline - 2026-11-11
- Initial release featuring isolated ISL vocabulary classifier (10 classes).
- 1D-Conv + 2-layer BiLSTM + Temporal Attention architecture (796,426 parameters).
- Tri-lingual speech synthesis in English, Hindi, and Telugu.
- Initial interactive Practice Studio and Vocabulary Explorer.
