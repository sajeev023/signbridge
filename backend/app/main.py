import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    VOCABULARY_LIST, CLASS_NAMES, CONFIDENCE_THRESHOLD,
    MODEL_METADATA_PATH, CORS_ORIGINS
)
from app.services.inference_engine import InferenceEngine
from app.services.practice_evaluator import PracticeEvaluator

app = FastAPI(
    title="SIGNBRIDGE API",
    description="Real-Time AI Indian Sign Language Recognition & Translation Service",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local prototyping & LAN exhibit demos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
inference_engine = InferenceEngine()
practice_evaluator = PracticeEvaluator()

class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float

class FramePayload(BaseModel):
    left_hand: Optional[List[Dict[str, float]]] = None
    right_hand: Optional[List[Dict[str, float]]] = None
    pose: Optional[List[Dict[str, float]]] = None
    timestamp: Optional[float] = None

class PracticePayload(BaseModel):
    target_sign: str
    left_hand: Optional[List[Dict[str, float]]] = None
    right_hand: Optional[List[Dict[str, float]]] = None
    pose: Optional[List[Dict[str, float]]] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": inference_engine.is_loaded,
        "device": str(inference_engine.device),
        "supported_vocabulary_count": len(VOCABULARY_LIST),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "timestamp": time.time()
    }

@app.get("/api/vocabulary")
def get_vocabulary():
    """Returns the comprehensive catalog of supported Indian Sign Language signs."""
    return {
        "count": len(VOCABULARY_LIST),
        "vocabulary": VOCABULARY_LIST
    }

@app.get("/api/model/info")
def get_model_info():
    """Returns metadata about the trained PyTorch BiLSTM neural network."""
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta
    return {
        "error": "Model metadata file not found.",
        "model_loaded": inference_engine.is_loaded
    }

@app.post("/api/predict/frame")
def predict_frame(payload: FramePayload):
    """
    Ingests a single frame of normalized landmark points into the temporal buffer
    and evaluates the gesture sequence.
    """
    result = inference_engine.push_landmarks(
        left_hand=payload.left_hand,
        right_hand=payload.right_hand,
        pose=payload.pose
    )
    return result

@app.post("/api/predict/reset")
def reset_prediction():
    """Clears the temporal inference buffer."""
    inference_engine.reset_buffer()
    return {"status": "cleared", "message": "Inference buffer successfully reset."}

@app.post("/api/practice/evaluate")
def evaluate_practice(payload: PracticePayload):
    """
    Evaluates a user's signing attempt in Learning/Practice mode
    against canonical ISL kinematics.
    """
    pred_result = inference_engine.push_landmarks(
        left_hand=payload.left_hand,
        right_hand=payload.right_hand,
        pose=payload.pose
    )
    eval_result = practice_evaluator.evaluate_practice_attempt(
        target_sign=payload.target_sign,
        current_prediction=pred_result,
        left_hand=payload.left_hand,
        right_hand=payload.right_hand,
        pose=payload.pose
    )
    eval_result["live_prediction"] = pred_result
    return eval_result

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    High-performance real-time WebSocket stream for continuous sign translation
    and live practice mode feedback.
    """
    await websocket.accept()
    session_engine = InferenceEngine()
    session_evaluator = PracticeEvaluator()

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)

            msg_type = data.get("type", "frame")

            if msg_type == "reset":
                session_engine.reset_buffer()
                await websocket.send_json({"type": "reset_ack", "status": "cleared"})
                continue

            left_hand = data.get("left_hand")
            right_hand = data.get("right_hand")
            pose = data.get("pose")
            mode = data.get("mode", "translate")
            target_sign = data.get("target_sign")

            pred_result = session_engine.push_landmarks(left_hand, right_hand, pose)

            response = {
                "type": "prediction",
                "mode": mode,
                "data": pred_result
            }

            if mode == "practice" and target_sign:
                practice_result = session_evaluator.evaluate_practice_attempt(
                    target_sign=target_sign,
                    current_prediction=pred_result,
                    left_hand=left_hand,
                    right_hand=right_hand,
                    pose=pose
                )
                response["practice"] = practice_result

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass

# Mount React Production Frontend if built
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "project": "SIGNBRIDGE",
            "tagline": "AI-Powered Indian Sign Language Recognition, Translation & Voice System",
            "status": "online",
            "docs_url": "/docs"
        }
