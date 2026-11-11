import os
import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import VOCABULARY_LIST, CLASS_NAMES

client = TestClient(app)

def create_dummy_video_bytes(num_frames=15, width=320, height=240, fps=25.0) -> bytes:
    """Helper that generates an in-memory MP4 video file using OpenCV."""
    import tempfile
    temp_p = Path(tempfile.gettempdir()) / f"test_dummy_{os.getpid()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_p), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (width//2, height//2), 20 + i, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    data = temp_p.read_bytes()
    if temp_p.exists():
        temp_p.unlink()
    return data

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "SIGNBRIDGE" in response.text

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.1.0"
    assert data["model_loaded"] is True
    assert data["supported_vocabulary_count"] == 10
    assert data["features"]["video_upload"] is True

def test_get_vocabulary():
    response = client.get("/api/vocabulary")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    vocab = data["vocabulary"]
    assert len(vocab) == 10
    ids = [item["id"] for item in vocab]
    assert "namaste" in ids
    assert "help" in ids
    assert "emergency" in ids

def test_get_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "architecture" in data
    assert data["architecture"]["num_classes"] == 10

def test_get_research_metrics():
    response = client.get("/api/research/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "validation_status" in data
    assert "synthetic_benchmark" in data["validation_status"]
    assert "real_world_validation" in data["validation_status"]

def test_buffering_and_reset():
    for _ in range(5):
        resp = client.post("/api/predict/frame", json={
            "left_hand": [{"x": 0.1, "y": 0.2, "z": 0.0}] * 21,
            "right_hand": [{"x": 0.3, "y": 0.4, "z": 0.0}] * 21,
            "pose": [{"x": 0.0, "y": 0.0, "z": 0.0}] * 25
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "buffering"

    reset_resp = client.post("/api/predict/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "cleared"

def test_prediction_with_full_sequence():
    client.post("/api/predict/reset")
    for i in range(30):
        t = i / 30.0
        resp = client.post("/api/predict/frame", json={
            "left_hand": [{"x": -0.05 * (1 - t), "y": 0.15 * t, "z": 0.1}] * 21,
            "right_hand": [{"x": 0.05 * (1 - t), "y": 0.15 * t, "z": 0.1}] * 21,
            "pose": [
                {"x": 0.0, "y": 0.45, "z": 0.0},
                {"x": -0.3, "y": 0.25, "z": 0.0},
                {"x": 0.3, "y": 0.25, "z": 0.0}
            ] + [{"x": 0.0, "y": 0.0, "z": 0.0}] * 22
        })
        assert resp.status_code == 200

    last_data = resp.json()
    assert "status" in last_data
    assert last_data["status"] in ["recognized", "uncertain", "evaluating"]

def test_practice_evaluation_endpoint():
    resp = client.post("/api/practice/evaluate", json={
        "target_sign": "namaste",
        "left_hand": [{"x": -0.02, "y": 0.15, "z": 0.1}] * 21,
        "right_hand": [{"x": 0.02, "y": 0.15, "z": 0.1}] * 21,
        "pose": [{"x": 0.0, "y": 0.0, "z": 0.0}] * 25
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_sign"] == "namaste"
    assert "accuracy_score" in data

def test_video_upload_endpoint():
    vid_bytes = create_dummy_video_bytes(num_frames=10)
    files = {"file": ("test_gesture.mp4", io.BytesIO(vid_bytes), "video/mp4")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "confidence" in data
    assert "video_metadata" in data
    assert data["video_metadata"]["total_frames"] == 10
    assert data["input_mode"] == "video_upload"

def test_video_upload_invalid_extension():
    files = {"file": ("malicious.exe", io.BytesIO(b"binary data"), "application/octet-stream")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 400
    assert "Unsupported video format" in resp.json()["detail"]
