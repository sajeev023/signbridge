# Installation and Setup Guide

This guide walks you through setting up and running **SIGNBRIDGE** on your local machine.

---

## 1. Prerequisites

Ensure your system has the following installed:
- **Python**: Version 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- **Node.js**: Version 18.0.0 or higher (v24 tested)
- **Package Managers**: `npm` and `uv` (or standard `python -m venv`)
- **Webcam**: Standard USB or built-in laptop camera
- **Browser**: Modern Chromium browser (Chrome, Edge, Brave) or Firefox with SpeechSynthesis support

---

## 2. Quick Start (Unified Single-Port Launch)

```bash
# 1. Clone repository
git clone https://github.com/sajeev023/signbridge.git
cd signbridge

# 2. Setup Python environment and install backend dependencies
cd backend
uv venv .venv
uv pip install -r requirements.txt

# 3. Setup and build React frontend
cd ../frontend
npm install
npm run build

# 4. Launch unified application server
cd ../backend
.venv/Scripts/python.exe run.py
```

Open your browser to:
**`http://127.0.0.1:8000`**

FastAPI serves both the complete React web application and the real-time AI inference endpoints on this single URL!

---

## 3. Development Mode (Hot-Reloading)

If you wish to modify frontend React components with instant hot module replacement (HMR):

### Terminal 1: Backend Server
```bash
cd backend
.venv/Scripts/python.exe run.py
# Running on http://127.0.0.1:8000
```

### Terminal 2: Frontend Vite Dev Server
```bash
cd frontend
npm run dev
# Running on http://127.0.0.1:5173
```

Open `http://localhost:5173` to view the development frontend connected to the local FastAPI backend.

---

## 4. Running Automated Tests & Evaluation

To verify backend endpoints and AI model performance:

```bash
cd backend

# Run the 7 backend API & inference test suites
.venv/Scripts/python.exe -m pytest tests/

# Run the experimental evaluation benchmark on the 180 test samples
.venv/Scripts/python.exe evaluate.py
```

---

## 5. Troubleshooting

- **Camera Access Denied**: Check browser address bar permissions icon (lock/camera) and ensure permissions are granted for `localhost`.
- **Port 8000 Already in Use**: Change the port in `backend/app/config.py` and `backend/run.py` to `8080` or another free port.
- **Microphone / Audio Muted**: Ensure your device audio output is active and verify that the in-app mute toggle in the top navbar is set to "Sound Active".
