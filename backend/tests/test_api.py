import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.config import VOCABULARY_LIST, CLASS_NAMES

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    # Root serves either React frontend HTML or API status JSON
    assert "SIGNBRIDGE" in response.text

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["supported_vocabulary_count"] == 10

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
    assert "performance" in data

def test_buffering_and_reset():
    # Push 5 frames
    for _ in range(5):
        resp = client.post("/api/predict/frame", json={
            "left_hand": [{"x": 0.1, "y": 0.2, "z": 0.0}] * 21,
            "right_hand": [{"x": 0.3, "y": 0.4, "z": 0.0}] * 21,
            "pose": [{"x": 0.0, "y": 0.0, "z": 0.0}] * 25
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "buffering"

    # Reset
    reset_resp = client.post("/api/predict/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "cleared"

def test_prediction_with_full_sequence():
    client.post("/api/predict/reset")

    # Generate 30 frames of Namaste-like sequence
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
    assert "confidence" in last_data

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
    assert "feedback" in data
    assert isinstance(data["feedback"], list)
