# SIGNBRIDGE 2.0 — ULTRA LEGENDARY EDITION
### Autonomous AI Indian Sign Language Recognition, Translation & Voice Communication Platform

SIGNBRIDGE 2.0 is an assistive communication and research platform engineered for public exhibition, classroom learning, and medical accessibility. It translates dynamic Indian Sign Language (ISL) gestures into real-time text and speech across English, Hindi, and Telugu using modern MediaPipe Tasks vision pipelines and a PyTorch BiLSTM neural network.

---

## 🚀 Key V2 Innovations & Highlights

- **Triple-Mode Input Architecture**:
  - **Mode A (Video Upload)**: Upload recorded MP4, WebM, MOV, or AVI video files (up to 50MB). The backend performs uniform temporal sampling across variable video lengths, extracts landmarks, runs BiLSTM inference, and instantly deletes temporary uploads. **No local webcam required.**
  - **Mode B (Live Camera)**: Real-time browser webcam tracking with automated hardware presence detection (`enumerateDevices`). When no webcam is detected, clear notifications and 1-click mode switching are provided.
  - **Mode C (Interactive Demonstration)**: Built-in exhibition simulation streaming canonical ISL kinematics with an explicit `DEMONSTRATION / SIMULATED` watermark.
- **MediaPipe Tasks Vision Models**: Powered by official Google MediaPipe Tasks vision packages (`hand_landmarker.task` and `pose_landmarker.task`) for robust 3D hand and upper-body joint tracking.
- **Spatio-Temporal Deep Learning**: 153 normalized spatial features $\times$ 30-frame temporal window processed by 1D temporal convolution, a 2-layer Bidirectional LSTM, and Temporal Self-Attention (796,426 parameters).
- **Epistemic Scientific Transparency**: Rigorously separates synthetic benchmarks (100% on 180 controlled sequences) from the prospective real-world ISLRTC protocol ($\geq 91.5\%$ target) in a dedicated **Research Dashboard** and `research/REAL_WORLD_VALIDATION.md`.
- **Zero Hallucination Tolerance**: Calibrated rejection floor ($\tau = 0.75$) outputs explicit repetition prompts instead of random guesses.
- **Sub-3ms Edge Inference**: 2.07 ms average latency on standard consumer CPUs without GPU requirements.
- **Tri-Lingual Localized Voice**: Instant speech synthesis in English (`en-IN`), Hindi (`hi-IN`), and Telugu (`te-IN`).
- **Interactive Practice Studio**: Biomechanical evaluation supporting both interactive kinematics and direct video upload scoring.
- **Data Sovereignty by Design**: In-memory frame processing, transient file deletion, and client-only browser translation logs.

---

## 📋 Supported ISL Vocabulary

| Sign ID | Canonical Name | Category | Hand Config | Kinematic Action |
|---|---|---|---|---|
| `namaste` | Namaste / Welcome | Greetings | Two-handed | Palms joined at chest, slight forward nod |
| `thank_you` | Thank You | Social | Single-handed | Flat hand touches chin, sweeps forward-downward |
| `help` | Help / Assistance | Emergency | Two-handed | Dominant fist on open non-dominant palm, lifting |
| `yes` | Yes / Affirmation | Social | Single-handed | Closed fist nodding vertically at wrist |
| `no` | No / Negation | Social | Single-handed | Open flat hand waving horizontally side-to-side |
| `water` | Water / Drink | Daily Needs | Single-handed | Curved hand elevates to mouth, tilts backward |
| `food` | Food / Eat | Daily Needs | Single-handed | Bunched fingertips tapping lips repeatedly |
| `emergency` | Emergency / Urgent | Emergency | Two-handed | Both hands crossing and waving at chest height |
| `doctor` | Doctor / Hospital | Medical | Two-handed | Dominant fingers tap non-dominant wrist pulse |
| `please` | Please / Request | Social | Single-handed | Open hand circular clockwise rub on center chest |

---

## 📊 Empirical Metrics & Scientific Validation

Evaluated on 180 held-out test sequences (`backend/evaluate.py`):

- **In-Silico Test Accuracy**: 100.00%
- **Macro Precision / Recall / F1**: 1.000 / 1.000 / 1.000
- **Mean Inference Latency**: 2.07 ms (Intel/AMD CPU)
- **Safety Rejection Threshold**: $\tau = 0.75$
- **Prospective Real-World Protocol**: See [research/REAL_WORLD_VALIDATION.md](research/REAL_WORLD_VALIDATION.md) for cross-signer split protocol.

---

## 🛠️ Quick Start & Execution

### Option 1: Docker (Single-Port Production)
```bash
docker compose up --build
```
Access the application at `http://localhost:8000`.

### Option 2: Local Python & Node Environment

#### Backend Setup
```bash
cd backend
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run build
```

When `frontend/dist` is built, FastAPI automatically serves the complete web application at:
**`http://127.0.0.1:8000`**

---

## 🧪 Automated Testing

Run the full pytest suite (10 automated unit & integration test suites):
```bash
cd backend
.\.venv\Scripts\python.exe -m pytest tests/
```

---

## 📁 Repository Structure

```
signbridge/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   ├── architecture.py       # PyTorch Conv1D + BiLSTM + Attention
│   │   │   └── tasks/                # MediaPipe Tasks vision model files
│   │   ├── services/
│   │   │   ├── landmark_extractor.py # Normalization & MediaPipe Tasks
│   │   │   ├── video_processor.py    # Video decode, uniform sampling, infer
│   │   │   ├── inference_engine.py   # Sliding window queue
│   │   │   ├── practice_evaluator.py # Biomechanical feedback
│   │   │   └── dataset_generator.py  # ISLRTC kinematics generator
│   │   ├── config.py                 # Hyperparameters & vocabulary
│   │   └── main.py                   # FastAPI V2 REST & WebSocket endpoints
│   ├── saved_models/                 # Serialized PyTorch weights
│   └── tests/test_api.py             # 10 PyTest automated test suites
├── frontend/
│   ├── src/
│   │   ├── components/               # UI Cards, confidence meter, navbar
│   │   ├── pages/                    # Workspace, Practice, History, Research, Privacy
│   │   ├── services/                 # API client, speech synthesis, camera
│   │   └── types/                    # TypeScript interfaces
│   └── dist/                         # Compiled production bundle
├── docs/
│   ├── ARCHITECTURE.md               # Spatio-temporal pipeline spec
│   ├── API_REFERENCE.md              # REST & WebSocket endpoint docs
│   ├── DEPLOYMENT.md                 # Cloud & Docker deployment guide
│   ├── PRIVACY.md                    # Data minimization & sovereignty
│   └── USER_GUIDE.md                 # Interactive user instructions
├── research/
│   ├── REAL_WORLD_VALIDATION.md      # Cross-signer scientific protocol
│   ├── EXISTING_SOLUTIONS.md         # Comparative prior art review
│   ├── ORIGINALITY_REPORT.md         # Novelty and differentiation
│   └── EXPERIMENTS.md                # Quantitative benchmarks
├── Dockerfile                        # Multi-stage production container
├── docker-compose.yml                # Docker compose configuration
└── CHANGELOG.md                      # Complete V1 to V2 revision record
```

---

## 📄 License & Attribution
Designed and engineered for assistive social impact and competitive science exhibitions, in compliance with the Indian Sign Language Research and Training Centre (ISLRTC) linguistic standards.
