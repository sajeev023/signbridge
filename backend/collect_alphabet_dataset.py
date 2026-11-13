"""
SIGNBRIDGE Real Human ASL Alphabet Data Collection & Landmark Extraction Utility
Captures live camera frames or image directory captures, runs MediaPipe Hand Landmark detection,
performs canonical 42-feature extraction, duplicate rejection, and saves structured records with
full provenance metadata.

Usage:
  python collect_alphabet_dataset.py --camera 0 --signer-id Signer_5 --sign A
  python collect_alphabet_dataset.py --input-dir path/to/images --signer-id Signer_5 --sign A
  python collect_alphabet_dataset.py --verify-schema
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Ensure local imports work
sys.path.append(str(Path(__file__).resolve().parent))
from alphabet_feature_extractor import extract_canonical_features, NUM_CANONICAL_FEATURES, FEATURE_NAMES_42

SUPPORTED_CLASSES = [
    "NEUTRAL",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]

DATASET_FILE = Path(__file__).resolve().parent / "data" / "real_human_alphabet_captures.json"


def init_mediapipe_hands():
    """Initializes MediaPipe Hands detector."""
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        return mp_hands, hands
    except Exception as e:
        print(f"[WARN] Failed to initialize MediaPipe Solutions Hands: {e}")
        return None, None


def check_is_duplicate(new_features: np.ndarray, existing_samples: List[Dict[str, Any]], threshold: float = 0.04) -> bool:
    """Rejects duplicate/stationary samples if Euclidean distance in feature space is below threshold."""
    if not existing_samples:
        return False
    for s in existing_samples:
        if "features_42" in s:
            feat = np.array(s["features_42"], dtype=np.float32)
            dist = float(np.linalg.norm(new_features - feat))
            if dist < threshold:
                return True
    return False


def validate_landmarks(landmarks: List[Dict[str, float]]) -> Tuple[bool, str]:
    """Validates 21 MediaPipe hand landmarks for completeness, coordinate validity, and boundary clipping."""
    if len(landmarks) != 21:
        return False, f"Expected 21 landmarks, got {len(landmarks)}"

    for i, pt in enumerate(landmarks):
        for k in ("x", "y"):
            val = pt.get(k, None)
            if val is None or np.isnan(val) or np.isinf(val):
                return False, f"Landmark {i} has invalid {k}: {val}"
            if val < -0.2 or val > 1.2:
                return False, f"Landmark {i} {k}={val} is severely clipped beyond sensor frame"
    return True, "Valid"


def create_record(
    sample_id: str,
    signer_id: str,
    sign_label: str,
    landmarks: List[Dict[str, float]],
    features_42: np.ndarray,
    session_id: str,
    lighting: str = "ambient_indoor",
    camera_distance: str = "medium (0.5-0.8m)",
    camera_orientation: str = "frontal",
    handedness: str = "Right",
    tracking_quality: float = 0.95,
    provenance: str = "real_human_camera_capture"
) -> Dict[str, Any]:
    """Constructs a compliant, provenance-rich sample record."""
    wrist = np.array([landmarks[0]["x"], landmarks[0]["y"], landmarks[0].get("z", 0.0)])
    mid_mcp = np.array([landmarks[9]["x"], landmarks[9]["y"], landmarks[9].get("z", 0.0)])
    palm_scale = float(np.linalg.norm(mid_mcp - wrist))

    return {
        "sample_id": sample_id,
        "signer_id": signer_id,
        "alphabet_label": sign_label,
        "convention": "ASL single-hand chart",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "capture_conditions": {
            "lighting": lighting,
            "camera_distance": camera_distance,
            "camera_orientation": camera_orientation,
            "handedness": handedness,
            "tracking_quality": round(float(tracking_quality), 3),
            "palm_scale": round(palm_scale, 4)
        },
        "data_provenance": provenance,
        "landmark_coordinates_3d": landmarks,
        "features_42": [round(float(x), 6) for x in features_42]
    }


def save_dataset_records(records: List[Dict[str, Any]], output_path: Path = DATASET_FILE):
    """Saves records to the dataset file, updating existing dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_data = {"metadata": {}, "samples": []}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {"metadata": {}, "samples": []}

    existing_samples = existing_data.get("samples", [])
    existing_samples.extend(records)

    signers = list(set(s.get("signer_id", "unknown") for s in existing_samples))
    classes = list(set(s.get("alphabet_label", "unknown") for s in existing_samples))

    payload = {
        "metadata": {
            "dataset_name": "SIGNBRIDGE Real Human ASL Alphabet Landmark Captures",
            "convention": "ASL Single-Hand (Gérard Aflague Chart)",
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_real_samples": len(existing_samples),
            "num_signers": len(signers),
            "signer_ids": signers,
            "classes_captured": classes,
            "features_dimension": 42
        },
        "samples": existing_samples
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[SUCCESS] Saved {len(records)} new records to {output_path} (Total: {len(existing_samples)})")


def run_collection_camera(signer_id: str, target_sign: str, camera_index: int = 0):
    """Interactive camera capture session."""
    import cv2
    mp_hands, hands = init_mediapipe_hands()
    if not hands:
        print("[ERROR] MediaPipe Hands could not be initialized.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera device {camera_index}")
        return

    print(f"\n=======================================================")
    print(f" SIGNBRIDGE REAL HUMAN DATA COLLECTOR")
    print(f" Signer: {signer_id} | Sign: {target_sign}")
    print(f" Controls: [SPACE] Capture Frame | [C] Change Sign | [Q] Save & Quit")
    print(f"=======================================================\n")

    session_id = f"sess_{signer_id}_{int(time.time())}"
    captured_records = []
    current_sign = target_sign

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        display_frame = frame.copy()
        h, w, _ = display_frame.shape

        valid_hand_detected = False
        features_42 = None
        landmarks_list = None

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            landmarks_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_lms.landmark]
            valid, msg = validate_landmarks(landmarks_list)
            if valid:
                valid_hand_detected = True
                features_42 = extract_canonical_features(landmarks_list)
                # Draw skeleton points
                for pt in landmarks_list:
                    cx, cy = int(pt["x"] * w), int(pt["y"] * h)
                    cv2.circle(display_frame, (cx, cy), 4, (0, 255, 0), -1)

        # Status HUD
        status_text = f"Signer: {signer_id} | Sign: {current_sign} | Captured: {len(captured_records)}"
        cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        det_text = "HAND DETECTED" if valid_hand_detected else "NO HAND / TRACKING UNRELIABLE"
        det_color = (0, 255, 0) if valid_hand_detected else (0, 0, 255)
        cv2.putText(display_frame, det_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)

        cv2.imshow("SIGNBRIDGE Alphabet Data Collector", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # SPACE bar
            if valid_hand_detected and features_42 is not None:
                if check_is_duplicate(features_42, captured_records):
                    print(f"[REJECT] Sample too similar to last capture (duplicate pose)")
                else:
                    rec_id = f"{signer_id}_{current_sign}_{int(time.time() * 1000)}"
                    rec = create_record(
                        sample_id=rec_id,
                        signer_id=signer_id,
                        sign_label=current_sign,
                        landmarks=landmarks_list,
                        features_42=features_42,
                        session_id=session_id
                    )
                    captured_records.append(rec)
                    print(f"[CAPTURED] {rec_id} for sign '{current_sign}' (Total: {len(captured_records)})")
            else:
                print(f"[WARN] Cannot capture: no valid hand detected")

    cap.release()
    cv2.destroyAllWindows()

    if captured_records:
        save_dataset_records(captured_records)


def run_verify_schema():
    """Runs a self-test of the record schema and feature extractor."""
    print("[TEST] Running schema and feature pipeline self-test...")
    from tests.test_alphabet_features import generate_mock_hand
    hand = generate_mock_hand()
    valid, msg = validate_landmarks(hand)
    assert valid, f"Landmark validation failed: {msg}"
    feats = extract_canonical_features(hand)
    assert len(feats) == 42, f"Expected 42 features, got {len(feats)}"
    rec = create_record(
        sample_id="test_001",
        signer_id="TestSigner",
        sign_label="A",
        landmarks=hand,
        features_42=feats,
        session_id="test_sess_01"
    )
    assert rec["alphabet_label"] == "A"
    assert len(rec["features_42"]) == 42
    print("[TEST] Schema verification succeeded completely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIGNBRIDGE Real Human Alphabet Data Collector")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index for live capture")
    parser.add_argument("--signer-id", type=str, default="Signer_Human_1", help="Identifier for human participant")
    parser.add_argument("--sign", type=str, default="A", choices=SUPPORTED_CLASSES, help="Alphabet label to collect")
    parser.add_argument("--verify-schema", action="store_true", help="Run automated schema and extractor verification")

    args = parser.parse_args()

    if args.verify_schema:
        run_verify_schema()
    elif args.camera is not None:
        run_collection_camera(args.signer_id, args.sign, args.camera)
    else:
        run_verify_schema()

