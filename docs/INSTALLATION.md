# SIGNBRIDGE 2.0 Installation & Setup Guide

This document describes local installation procedures for SIGNBRIDGE 2.0.

---

## Prerequisites

- **Python**: 3.10 or 3.11 (Python 3.11 recommended)
- **Node.js**: v18.0.0 or higher (v20 LTS recommended)
- **Git**
- **Docker** (Optional, for containerized execution)

---

## Method 1: Docker (Recommended)

To run the entire system (backend + frontend) in a single container:
```bash
# Clone the repository
git clone https://github.com/sajeev023/signbridge.git
cd signbridge

# Run with docker-compose
docker compose up -d
```
Open your browser to `http://localhost:8000`.

---

## Method 2: Manual Local Setup

### 1. Clone & Directory Structure
```bash
git clone https://github.com/sajeev023/signbridge.git
cd signbridge
```

### 2. Backend Environment
```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (cmd):
# .\.venv\Scripts\activate.bat
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Verify that the MediaPipe task files are present:
- `backend/app/models/tasks/hand_landmarker.task`
- `backend/app/models/tasks/pose_landmarker.task`

Run backend tests:
```bash
python -m pytest tests/
```

### 3. Frontend Environment
```bash
cd ../frontend

# Install dependencies
npm install

# Build static production assets
npm run build
```

### 4. Running the Unified Server
FastAPI automatically serves the built frontend from `frontend/dist`.
```bash
cd ../backend
python run.py
```
Open **`http://127.0.0.1:8000`** in your browser.
