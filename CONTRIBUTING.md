# Contributing to SIGNBRIDGE

Thank you for your interest in contributing to **SIGNBRIDGE**! We welcome contributions that expand authentic Indian Sign Language accessibility, improve model accuracy, or enhance application usability.

---

## Code of Conduct & Ground Rules

1. **Linguistic Authenticity**: All contributed sign definitions must align with official standards defined by the Indian Sign Language Research and Training Centre (ISLRTC) or verified deaf community conventions. Do not contribute American Sign Language (ASL) gestures labeled as ISL.
2. **Scientific Honesty**: Never report fabricated accuracy metrics. All performance claims must be verifiable via reproducible evaluation scripts.
3. **Privacy First**: Never commit private personal recordings or unconsented video datasets.

---

## Development Setup

```bash
git clone https://github.com/sajeev023/signbridge.git
cd signbridge

# Backend setup
cd backend
uv venv .venv
uv pip install -r requirements.txt

# Run backend tests
.venv/Scripts/python.exe -m pytest tests/

# Frontend setup
cd ../frontend
npm install
npm run build
```

---

## How to Add a New ISL Sign

1. Add the sign definition to `backend/app/config.py` in `VOCABULARY_LIST` with English, Hindi, and Telugu translations.
2. Define its canonical kinematic trajectory in `backend/app/services/dataset_generator.py`.
3. Add the sign to `frontend/src/services/apiClient.ts` in `FALLBACK_VOCABULARY`.
4. Re-run training: `python backend/train.py`.
5. Run evaluation: `python backend/evaluate.py`.
6. Run the test suite: `pytest backend/tests/`.
7. Submit a clean pull request with documentation of the added sign.
