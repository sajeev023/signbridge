# Changelog
All notable changes to the SIGNBRIDGE project are documented here.

## [2.1.0-rc1] - Release Candidate - 2026-11-11

### Reliability & Concurrency Hardening
- **Async Threadpool Offloading**: Offloaded CPU-bound OpenCV decoding, MediaPipe landmark extraction, and PyTorch inference to background threadpools via `starlette.concurrency.run_in_threadpool`, ensuring the FastAPI event loop remains responsive to healthchecks and concurrent requests.
- **Concurrency Rate Limiting**: Added `asyncio.Semaphore(4)` on video upload endpoints to prevent worker starvation and memory exhaustion under concurrent upload bursts.
- **Path Traversal Protection**: Sanitized upload filenames with `os.path.basename` and assigned collision-resistant UUID-based temp filenames.
- **Consistent Size Limits**: Enforced `MAX_VIDEO_SIZE_BYTES` (50MB) streaming chunk validation on both `/api/predict/video` and `/api/practice/video` endpoints, returning `413 Payload Too Large`.

### Privacy Transparency & Disclosures
- **Operational Disclosures**: Corrected privacy statements across `PrivacyPage.tsx`, `App.tsx`, and `docs/PRIVACY.md` to explicitly differentiate between client-side RAM processing (Live Camera mode) and transient server-side decoding with sub-500ms automated file deletion (Video Upload mode).
- **Zero Disk Persistence Guarantee**: Added explicit verification testing confirming zero lingering temporary files after both successful and failed upload attempts.

### Quality Assurance & Benchmarking
- **Comprehensive Video Testing Suite**: Authored `backend/tests/test_video_pipeline.py` testing MP4, WebM, MOV, and AVI containers, blank frame rejection, duration boundaries (5 frames to 60 seconds), corrupted headers, and empty files (23 total automated tests passing).
- **Performance Profiling Suite**: Created `backend/benchmark_performance.py` measuring cold-start import latency, model loading latency (21.56ms), pure CPU inference (1.78ms mean, 563 inferences/sec), and peak process memory (415.59MB).

### Cloud & Container Infrastructure
- **UTF-8 Requirements Normalization**: Converted `backend/requirements.txt` from Windows UTF-16LE with BOM to clean UTF-8.
- **Render Deployment Manifest**: Added `render.yaml` defining Docker web service specifications, health check paths, and port configurations.
- **Dynamic CORS**: Configured `CORS_ORIGINS` to load from environment variables with fallback for local prototyping.

---

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
