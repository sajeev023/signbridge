# API Reference & Endpoint Specification
## SIGNBRIDGE 2.0 Engine API

Base URL in local development: `http://localhost:8000`  
Interactive OpenAPI Swagger Docs: `http://localhost:8000/docs`

---

## 1. System Health & Metadata

### `GET /api/health`
Checks backend engine status, PyTorch model loading, and MediaPipe Tasks model readiness.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "mediapipe_tasks_ready": true,
  "device": "cpu",
  "active_vocabulary_size": 10,
  "supported_modes": ["video_upload", "live_camera", "demonstration"]
}
```

---

### `GET /api/vocabulary`
Returns the 10 official ISL vocabulary signs with multi-lingual translations and kinematic descriptions.

**Response `200 OK`**:
```json
{
  "count": 10,
  "vocabulary": [
    {
      "id": "namaste",
      "name": "Namaste",
      "label": "Namaste / Greeting",
      "category": "social",
      "translations": {
        "en": "Hello / Greetings / Namaste",
        "hi": "नमस्ते / अभिवादन",
        "te": "నమస్కారం"
      },
      "description": "Both palms flat and pressed together vertically in front of chest...",
      "hand_config": "two_handed",
      "movement": "Upward join at chest line.",
      "difficulty": "Beginner"
    }
  ]
}
```

---

### `GET /api/model/info`
Exposes the PyTorch BiLSTM neural network architecture and feature pipeline metadata.

**Response `200 OK`**:
```json
{
  "model_name": "SignBridge-BiLSTM-ISL",
  "version": "2.0.0",
  "architecture": "1D-Conv + 2-layer BiLSTM + Temporal Attention",
  "parameters": 796426,
  "input_features": 153,
  "sequence_length": 30,
  "classes": 10,
  "confidence_threshold": 0.75,
  "latency_cpu_ms": 2.07
}
```

---

### `GET /api/research/metrics`
Returns empirical validation metrics, synthetic benchmark KPIs, and per-class precision/recall metrics.

**Response `200 OK`**:
```json
{
  "synthetic_accuracy": 1.0,
  "synthetic_test_samples": 180,
  "real_world_target_accuracy": 0.915,
  "average_latency_ms": 2.07,
  "confidence_threshold": 0.75,
  "class_metrics": [
    {
      "sign_id": "namaste",
      "name": "Namaste",
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "samples": 18
    }
  ]
}
```

---

## 2. Inference Endpoints

### `POST /api/predict/video`
Infers sign language gesture from an uploaded video file (Mode A).

- **Content-Type**: `multipart/form-data`
- **Supported Formats**: MP4, WebM, MOV, AVI (max 50MB)
- **Parameters**: `file`: Binary video file

**Response `200 OK`**:
```json
{
  "status": "recognized",
  "is_confident": true,
  "sign_id": "namaste",
  "label": "Namaste / Greeting",
  "category": "social",
  "confidence": 0.985,
  "translations": {
    "en": "Hello / Greetings / Namaste",
    "hi": "नमस्ते / अभिवादन",
    "te": "నమస్కారం"
  },
  "description": "Both palms flat and pressed together...",
  "processing_duration_ms": 185.4,
  "input_mode": "video_upload",
  "model_version": "2.0.0",
  "video_metadata": {
    "duration_seconds": 2.4,
    "total_frames": 72,
    "fps": 30.0,
    "sampled_frames": 30
  }
}
```

---

### `POST /api/predict/frame`
Streams individual landmark sets into the 30-frame temporal buffer for live webcam recognition (Mode B).

- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "left_hand": [{"x": 0.5, "y": 0.5, "z": 0.0}],
  "right_hand": [{"x": 0.6, "y": 0.5, "z": 0.0}],
  "pose": [{"x": 0.5, "y": 0.2, "z": 0.0}]
}
```

**Response `200 OK`**: Returns `PredictionResult` with `buffer_fill` indicator (0–30).

---

### `POST /api/predict/reset`
Flushes the live temporal buffer.

---

### `POST /api/practice/video`
Evaluates user signing accuracy in an uploaded video against a canonical target sign.

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Video file
  - `target_sign`: Target sign ID (e.g., `namaste`)

**Response `200 OK`**:
```json
{
  "target_sign": "namaste",
  "target_label": "Namaste / Greeting",
  "category": "social",
  "accuracy_score": 98.5,
  "grade": "Excellent",
  "status": "mastered",
  "feedback": [
    "Hand configuration matches canonical ISL standard.",
    "Motion trajectory alignment: 98.5% similarity."
  ],
  "canonical_movement": "Upward join at chest line.",
  "difficulty": "Beginner"
}
```
