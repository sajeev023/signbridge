import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch

from app.config import (
    CLASS_NAMES, VOCABULARY_LIST, SEQUENCE_LENGTH, NUM_FEATURES,
    CONFIDENCE_THRESHOLD, UNCERTAIN_FALLBACK_TEXT
)
from app.services.landmark_extractor import LandmarkExtractor
from app.services.inference_engine import InferenceEngine
from app.services.practice_evaluator import PracticeEvaluator

MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}

class VideoProcessor:
    """
    Asynchronous and batch video processing service for Indian Sign Language recognition.
    
    Accepts uploaded video files (MP4, WebM, MOV, AVI), uniformly samples 30 temporal
    keyframes across the duration, extracts normalized spatial landmarks, and executes
    PyTorch BiLSTM sequence inference.
    """
    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine
        self.extractor = self.engine.extractor
        self.evaluator = PracticeEvaluator()
        self.vocab_lookup = {item["id"]: item for item in VOCABULARY_LIST}

    def validate_video_file(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found at {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        if file_size > MAX_VIDEO_SIZE_BYTES:
            raise ValueError(f"Video file exceeds maximum allowed size of 50 MB (found {file_size / (1024*1024):.1f} MB).")

        ext = file_path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    def extract_sequence_from_video(self, video_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decodes video frames, performs uniform temporal sampling across duration,
        and extracts normalized 153-D landmark vectors for each sampled frame.
        """
        self.validate_video_file(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file. File may be corrupted or use an unsupported codec.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        duration_sec = total_frames / fps if total_frames > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames < 5:
            cap.release()
            raise ValueError("Video contains fewer than 5 frames; insufficient temporal depth for gesture recognition.")

        if duration_sec > 60.0:
            cap.release()
            raise ValueError(f"Video duration ({duration_sec:.1f}s) exceeds maximum allowed limit of 60.0s.")

        # Uniform temporal sampling of exactly SEQUENCE_LENGTH (30) frame indices
        sample_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
        sample_indices_set = set(sample_indices)

        frames_dict = {}
        current_frame_idx = 0

        # Read frames sequentially
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame_idx in sample_indices_set:
                frames_dict[current_frame_idx] = frame.copy()
            current_frame_idx += 1

        cap.release()

        # If video ended prematurely, fill any missing indices with the last available frame
        if len(frames_dict) == 0:
            raise ValueError("Failed to decode frames from the uploaded video.")

        available_keys = sorted(frames_dict.keys())
        last_frame = frames_dict[available_keys[-1]]

        sequence_features = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)

        for step, target_idx in enumerate(sample_indices):
            if target_idx in frames_dict:
                frame = frames_dict[target_idx]
            else:
                # Find nearest frame
                nearest_idx = min(available_keys, key=lambda k: abs(k - target_idx))
                frame = frames_dict[nearest_idx]

            feat = self.extractor.process_image_frame(frame)
            sequence_features[step] = feat

        meta = {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(duration_sec, 2),
            "width": width,
            "height": height,
            "sampled_frames": SEQUENCE_LENGTH
        }

        return sequence_features, meta

    def process_video_upload(self, video_path: Path) -> Dict[str, Any]:
        """
        Processes uploaded video and runs PyTorch BiLSTM gesture recognition.
        """
        t_start = time.perf_counter()
        features, video_meta = self.extract_sequence_from_video(video_path)

        # Reshape to (1, 30, 153)
        input_tensor = torch.tensor(
            features[np.newaxis, ...],
            dtype=torch.float32,
            device=self.engine.device
        )

        with torch.no_grad():
            logits, attn_weights = self.engine.model(input_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            attn = attn_weights.cpu().numpy()[0].tolist()

        processing_ms = (time.perf_counter() - t_start) * 1000.0

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        pred_class = CLASS_NAMES[top_idx]

        all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        meta = self.vocab_lookup.get(pred_class, {})

        is_confident = confidence >= CONFIDENCE_THRESHOLD

        result = {
            "status": "recognized" if is_confident else "uncertain",
            "is_confident": is_confident,
            "sign_id": pred_class,
            "label": meta.get("label", pred_class) if is_confident else UNCERTAIN_FALLBACK_TEXT,
            "category": meta.get("category", "general"),
            "confidence": round(confidence, 4),
            "threshold": CONFIDENCE_THRESHOLD,
            "translations": meta.get("translations", {
                "en": meta.get("label", pred_class),
                "hi": meta.get("label", pred_class),
                "te": meta.get("label", pred_class)
            }),
            "description": meta.get("description", ""),
            "processing_duration_ms": round(processing_ms, 2),
            "video_metadata": video_meta,
            "attention_weights": attn,
            "all_probabilities": all_probs,
            "model_version": "2.0.0",
            "input_mode": "video_upload"
        }

        if not is_confident:
            result["message"] = UNCERTAIN_FALLBACK_TEXT

        return result

    def process_practice_video(self, video_path: Path, target_sign: str) -> Dict[str, Any]:
        """
        Processes uploaded video specifically for Practice Studio form evaluation.
        """
        prediction_result = self.process_video_upload(video_path)

        # Evaluate practice form
        eval_result = self.evaluator.evaluate_practice_attempt(
            target_sign=target_sign,
            current_prediction=prediction_result,
            left_hand=None,  # Video processor computes via sequence
            right_hand=None,
            pose=None
        )

        eval_result["live_prediction"] = prediction_result
        eval_result["video_metadata"] = prediction_result.get("video_metadata", {})
        return eval_result
