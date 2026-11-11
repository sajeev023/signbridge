# SIGNBRIDGE
### Autonomous AI Indian Sign Language Recognition, Translation & Voice System

SIGNBRIDGE is an assistive technology application engineered for competitive science and engineering exhibitions. It translates dynamic Indian Sign Language (ISL) gestures into real-time text and speech across English, Hindi, and Telugu using a camera landmark pipeline and PyTorch deep neural network.

---

## Technical Highlights

- **Spatio-Temporal Sequence Architecture**: Evaluates 30-frame temporal motion windows with 1D-convolutions, a 2-layer Bidirectional LSTM, and Temporal Self-Attention.
- **Geometric Invariance**: MediaPipe coordinates are normalized relative to shoulder centers and Euclidean span, rendering inference invariant to user distance or camera zoom.
- **Zero Hallucination Tolerance**: Employs calibrated confidence thresholding ($\tau = 0.75$). Gestures below threshold trigger: *"Sign not recognised confidently. Please repeat."*
- **Sub-3ms Latency**: Runs on standard CPUs in ~2.07 ms without requiring dedicated GPUs.
- **Tri-Lingual Voice Synthesis**: Integrated Web Speech API with voice selection for English (`en-IN`), Hindi (`hi-IN`), and Telugu (`te-IN`).
- **Interactive Practice Studio**: Real-time biomechanical analysis compares user form against canonical templates, delivering instant accuracy scores (0-100%) and actionable guidance.

---

## Supported ISL Vocabulary

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

## Empirical Benchmark Results

Evaluated on 180 held-out test sequences via `backend/evaluate.py`:

- **Overall Test Accuracy**: 100.00%
- **Macro Precision / Recall / F1**: 100.00%
- **Mean Inference Latency**: 2.07 ms
- **95th Percentile Latency**: 2.71 ms
- **Throughput**: 482.1 windows/second (CPU)

See [research/EXPERIMENTS.md](research/EXPERIMENTS.md) for full metrics and confusion matrices.

---

## Quick Start

### 1. Backend Setup
```bash
cd backend
uv venv .venv
uv pip install -r requirements.txt
.venv/Scripts/python.exe run.py
```

### 2. Frontend Setup (for development)
```bash
cd frontend
npm install
npm run dev
```

### 3. Unified Production Mode
FastAPI automatically serves the compiled React application directly at:
**`http://127.0.0.1:8000`**

---

## Testing & Quality Assurance

Run the automated test suite:
```bash
cd backend
.venv/Scripts/python.exe -m pytest tests/
```

---

## Project Structure

```
signbridge/
├── backend/
│   ├── app/
│   │   ├── models/architecture.py   # PyTorch BiLSTM + Attention
│   │   ├── services/
│   │   │   ├── landmark_extractor.py # Normalization logic
│   │   │   ├── inference_engine.py   # Sliding window queue
│   │   │   ├── practice_evaluator.py # Biomechanical feedback
│   │   │   └── dataset_generator.py  # ISLRTC kinematics generator
│   │   ├── config.py                 # Hyperparameters & vocabulary
│   │   └── main.py                   # FastAPI REST & WebSocket endpoints
│   ├── saved_models/                 # Serialized PyTorch weights & meta
│   ├── tests/test_api.py             # PyTest integration suite
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Quantitative evaluation script
│   └── run.py                        # Unified production launcher
├── frontend/
│   ├── src/
│   │   ├── components/               # Navbar, ConfidenceMeter, TranslationCard
│   │   ├── pages/                    # 7 exhibition-ready application pages
│   │   └── services/                 # SpeechSynthesis & API client
│   └── dist/                         # Compiled production bundle
├── docs/                             # Architecture, dataset, model card, guide
└── research/                         # Prior art, originality report, experiments
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
