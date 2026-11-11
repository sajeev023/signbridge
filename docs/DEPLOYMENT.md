# Production Deployment Guide
## SIGNBRIDGE 2.0 Web Platform

This guide outlines containerized and cloud deployment architectures for SIGNBRIDGE 2.0.

---

## 1. Single-Container Unified Architecture

In production, SIGNBRIDGE is packaged as a single container image where:
1. The React/Vite frontend is built into static assets (`frontend/dist`).
2. FastAPI mounts and serves `frontend/dist` at `/`, providing static file hosting and SPA routing.
3. FastAPI provides all REST and WebSocket endpoints under `/api`.
4. The container runs on a single exposed HTTP port (default `8000`).

```
Internet / User Browser
         │  (Port 8000 / 443)
         ▼
┌───────────────────────────────────────────────┐
│ Docker Container                              │
│                                               │
│  FastAPI (Uvicorn / Gunicorn)                 │
│  ├── /api/*          REST & WebSocket Engine  │
│  ├── /assets/*       Vite Static Bundles      │
│  └── /*              SPA Index HTML Route     │
│                                               │
│  Python 3.11 Runtime + PyTorch CPU + MediaPipe│
│  Models: signbridge_isl_bilstm.pt             │
│          hand_landmarker.task                 │
│          pose_landmarker.task                 │
└───────────────────────────────────────────────┘
```

---

## 2. Docker Deployment

### 2.1 Build the Image
```bash
docker build -t signbridge:2.0.0 .
```

### 2.2 Run with Docker
```bash
docker run -d \
  -p 8000:8000 \
  -e PORT=8000 \
  -e ENVIRONMENT=production \
  --name signbridge-v2 \
  signbridge:2.0.0
```

### 2.3 Run with Docker Compose
```bash
docker compose up -d
```
Verify the container status:
```bash
docker compose ps
curl http://localhost:8000/api/health
```

---

## 3. Cloud Platform Deployment

### 3.1 Render.com
1. Create a new **Web Service** pointing to your repository `https://github.com/sajeev023/signbridge`.
2. Configure settings:
   - **Environment**: `Docker`
   - **Branch**: `main` or `feature/signbridge-v2`
   - **Instance Type**: Starter / Standard (at least 1GB RAM recommended for PyTorch + MediaPipe)
   - **Health Check Path**: `/api/health`
3. Environment variables:
   - `PORT=8000`
   - `ENVIRONMENT=production`
4. Deploy service. Render will automatically issue a free managed SSL certificate.

### 3.2 Railway.app
1. Create a new project from your GitHub repo.
2. Railway detects the `Dockerfile` in the root directory.
3. Under **Variables**, add:
   - `PORT=8000`
4. Under **Settings**, generate a domain (e.g., `signbridge-production.up.railway.app`).
5. Deployment will build the frontend, package the Python backend, and serve immediately.

---

## 4. Security & SSL Requirements

- **Browser Camera Permissions**: Modern web browsers restrict `getUserMedia()` (camera access) exclusively to secure origins (`https://` or `localhost`). When hosting on public domains, always ensure TLS/SSL is active via Cloudflare, Let's Encrypt, or managed cloud platform proxies.
- **Video Upload Limits**: Max payload size is bounded to 50MB by default in `video_processor.py`. Adjust via environment variable `MAX_UPLOAD_SIZE_MB` if necessary.
- **Transient File Cleanup**: All uploaded videos are isolated in temporary directories and deleted in a `finally` block post-inference.
