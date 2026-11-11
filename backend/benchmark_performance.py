import os
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
import cv2

def get_process_memory_mb():
    try:
        pid = os.getpid()
        cmd = f"(Get-Process -Id {pid}).WorkingSet64 / 1MB"
        out = subprocess.check_output(["powershell", "-Command", cmd])
        return float(out.strip())
    except Exception:
        return 0.0

def run_benchmarks():
    print("=" * 70)
    print("SIGNBRIDGE 2.1 — SYSTEM & INFERENCE PERFORMANCE BENCHMARK")
    print("=" * 70)

    # 1. Cold start import latency
    t0 = time.perf_counter()
    import torch
    import mediapipe as mp
    from app.services.inference_engine import InferenceEngine
    from app.services.video_processor import VideoProcessor
    t_imports = (time.perf_counter() - t0) * 1000.0
    print(f"[1] Cold Start Import Latency:        {t_imports:.2f} ms")

    mem_after_import = get_process_memory_mb()
    print(f"    Memory after library imports:      {mem_after_import:.2f} MB")

    # 2. Model loading time
    t0 = time.perf_counter()
    engine = InferenceEngine()
    t_model_load = (time.perf_counter() - t0) * 1000.0
    print(f"[2] Model Loading Latency:            {t_model_load:.2f} ms")

    # 3. MediaPipe Tasks detector initialization
    t0 = time.perf_counter()
    engine.extractor._ensure_models_loaded()
    t_mediapipe_init = (time.perf_counter() - t0) * 1000.0
    print(f"[3] MediaPipe Tasks Init Latency:     {t_mediapipe_init:.2f} ms")

    mem_after_models = get_process_memory_mb()
    print(f"    Memory with Models Loaded:        {mem_after_models:.2f} MB")

    # 4. Pure BiLSTM CPU Inference Latency
    sample_seq = np.random.randn(1, 30, 153).astype(np.float32)
    sample_tensor = torch.tensor(sample_seq)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = engine.model(sample_tensor)

    latencies = []
    iterations = 100
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = engine.model(sample_tensor)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p50_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    throughput = 1000.0 / mean_lat

    print(f"[4] Pure Model Inference (CPU, {iterations} runs):")
    print(f"    - Mean Latency:                   {mean_lat:.2f} ms")
    print(f"    - Median (P50) Latency:           {p50_lat:.2f} ms")
    print(f"    - 95th Percentile (P95):          {p95_lat:.2f} ms")
    print(f"    - 99th Percentile (P99):          {p99_lat:.2f} ms")
    print(f"    - Min / Max:                      {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"    - Throughput:                     {throughput:.1f} inferences/sec")

    # 5. Video Processing & Keyframe Extraction Latency
    video_proc = VideoProcessor(engine)

    import tempfile
    temp_video = Path(tempfile.gettempdir()) / f"bench_{os.getpid()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video), fourcc, 25.0, (320, 240))
    for i in range(50):  # 2.0-second video at 25 fps
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 120), 30, (255, 255, 255), -1)
        out.write(frame)
    out.release()

    t0 = time.perf_counter()
    res = video_proc.process_video_upload(temp_video)
    t_video_pipeline = (time.perf_counter() - t0) * 1000.0

    if temp_video.exists():
        temp_video.unlink()

    print(f"[5] End-to-End Video Processing (2.0s clip, 50 frames):")
    print(f"    - Total Processing Duration:      {t_video_pipeline:.2f} ms")
    print(f"    - Decoded Frames:                 {res['video_metadata']['total_frames']}")
    print(f"    - Uniformly Sampled Frames:       {res['video_metadata']['sampled_frames']}")
    print(f"    - Inferred Status:                {res['status']}")
    print(f"    - Confidence:                     {res['confidence']:.4f}")

    mem_peak = get_process_memory_mb()
    print(f"[6] Peak Working Set Memory:          {mem_peak:.2f} MB")

    print("=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
