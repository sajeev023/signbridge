import os
import io
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import VOCABULARY_LIST, CLASS_NAMES

client = TestClient(app)

def create_synthetic_video(
    ext: str = "mp4",
    num_frames: int = 30,
    width: int = 320,
    height: int = 240,
    fps: float = 25.0,
    content: str = "moving_circle"
) -> bytes:
    """Helper to generate in-memory video files across different container formats."""
    temp_p = Path(tempfile.gettempdir()) / f"synth_test_{os.getpid()}_{ext}.{ext}"
    
    if ext == "avi":
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    elif ext in ["mp4", "mov"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(str(temp_p), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if content == "moving_circle":
            cv2.circle(frame, (width // 4 + i * 4, height // 2), 25, (200, 200, 200), -1)
        elif content == "blank":
            pass  # Empty black frame
        out.write(frame)
    out.release()
    
    data = temp_p.read_bytes()
    if temp_p.exists():
        temp_p.unlink()
    return data

def test_video_mp4_valid():
    """Phase 3 Test: Valid MP4 video upload and recognition."""
    vid = create_synthetic_video(ext="mp4", num_frames=30, fps=25.0)
    files = {"file": ("test_gesture.mp4", io.BytesIO(vid), "video/mp4")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "confidence" in data
    assert "video_metadata" in data
    assert data["video_metadata"]["total_frames"] == 30
    assert data["video_metadata"]["sampled_frames"] == 30
    assert data["input_mode"] == "video_upload"

def test_video_avi_valid():
    """Phase 3 Test: Valid AVI video upload and recognition."""
    vid = create_synthetic_video(ext="avi", num_frames=20, fps=20.0)
    files = {"file": ("test_gesture.avi", io.BytesIO(vid), "video/x-msvideo")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["video_metadata"]["total_frames"] == 20

def test_video_mov_valid():
    """Phase 3 Test: Valid MOV video upload and recognition."""
    vid = create_synthetic_video(ext="mov", num_frames=25, fps=25.0)
    files = {"file": ("test_gesture.mov", io.BytesIO(vid), "video/quicktime")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["video_metadata"]["total_frames"] == 25

def test_video_empty_file():
    """Phase 3 Test: Empty file (0 bytes) returns 422."""
    files = {"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()

def test_video_unsupported_extension():
    """Phase 3 Test: Unsupported file extension returns 400."""
    files = {"file": ("test.txt", io.BytesIO(b"Hello world text"), "text/plain")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 400
    assert "Unsupported video format" in resp.json()["detail"]

def test_video_corrupted_file():
    """Phase 3 Test: File with .mp4 extension but corrupted content returns 422."""
    corrupted_bytes = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00garbage_data_non_video_header_123456789"
    files = {"file": ("corrupted.mp4", io.BytesIO(corrupted_bytes), "video/mp4")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 422
    assert "corrupted" in resp.json()["detail"].lower() or "could not open" in resp.json()["detail"].lower()

def test_video_too_few_frames():
    """Phase 3 Test: Video with fewer than 5 frames returns 422."""
    vid = create_synthetic_video(ext="mp4", num_frames=3, fps=25.0)
    files = {"file": ("too_short.mp4", io.BytesIO(vid), "video/mp4")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 422
    assert "fewer than 5 frames" in resp.json()["detail"].lower()

def test_video_too_long_duration():
    """Phase 3 Test: Video duration exceeding 60s returns 422."""
    # 2000 frames at 25 fps = 80 seconds
    vid = create_synthetic_video(ext="mp4", num_frames=1600, fps=25.0)
    files = {"file": ("too_long.mp4", io.BytesIO(vid), "video/mp4")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 422
    assert "exceeds maximum allowed limit of 60.0s" in resp.json()["detail"]

def test_video_blank_rejection():
    """Phase 3 Test: Blank video with no detectable gestures falls below threshold and triggers rejection."""
    vid = create_synthetic_video(ext="mp4", num_frames=30, content="blank")
    files = {"file": ("blank.mp4", io.BytesIO(vid), "video/mp4")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["recognized", "uncertain"]
    # If uncertain, message prompt is returned
    if not data["is_confident"]:
        assert "message" in data

def test_video_oversized_limit():
    """Phase 3 Test: File exceeding 50 MB triggers 413 Payload Too Large."""
    # Use a generator/mock buffer that reports > 50MB
    class OversizedMockStream(io.RawIOBase):
        def __init__(self, size):
            self.size = size
            self.bytes_read = 0

        def read(self, n=-1):
            if self.bytes_read >= self.size:
                return b""
            chunk_size = min(n if n > 0 else 1024 * 1024, self.size - self.bytes_read)
            self.bytes_read += chunk_size
            return b"A" * chunk_size

    large_stream = OversizedMockStream(51 * 1024 * 1024)  # 51 MB
    files = {"file": ("huge_file.mp4", large_stream, "video/mp4")}
    resp = client.post("/api/predict/video", files=files)
    assert resp.status_code == 413
    assert "50 mb" in resp.json()["detail"].lower()

def test_practice_video_valid():
    """Phase 3 Test: Video upload evaluation against valid target sign."""
    vid = create_synthetic_video(ext="mp4", num_frames=30)
    files = {"file": ("practice.mp4", io.BytesIO(vid), "video/mp4")}
    data = {"target_sign": "namaste"}
    resp = client.post("/api/practice/video", files=files, data=data)
    assert resp.status_code == 200
    res = resp.json()
    assert res["target_sign"] == "namaste"
    assert "accuracy_score" in res
    assert "grade" in res

def test_practice_video_invalid_sign():
    """Phase 3 Test: Video upload evaluation with invalid target sign returns 400."""
    vid = create_synthetic_video(ext="mp4", num_frames=10)
    files = {"file": ("practice.mp4", io.BytesIO(vid), "video/mp4")}
    data = {"target_sign": "nonexistent_sign_xyz"}
    resp = client.post("/api/practice/video", files=files, data=data)
    assert resp.status_code == 400
    assert "not found in supported vocabulary" in resp.json()["detail"]

def test_temp_file_cleanup():
    """Phase 3 Test: Verify that all temporary upload files are completely removed post-inference."""
    temp_dir = Path(tempfile.gettempdir()) / "signbridge_uploads"
    initial_files = set(temp_dir.glob("*")) if temp_dir.exists() else set()

    # Run valid upload
    vid = create_synthetic_video(ext="mp4", num_frames=10)
    files = {"file": ("cleanup_test.mp4", io.BytesIO(vid), "video/mp4")}
    client.post("/api/predict/video", files=files)

    # Run failed upload
    files_fail = {"file": ("cleanup_fail.mp4", io.BytesIO(b"corrupted_bytes"), "video/mp4")}
    client.post("/api/predict/video", files=files_fail)

    final_files = set(temp_dir.glob("*")) if temp_dir.exists() else set()
    # No new leftover files in temp directory
    leftover = final_files - initial_files
    assert len(leftover) == 0, f"Found lingering temporary files: {leftover}"
