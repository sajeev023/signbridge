import os
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent

# Vocabulary definition for Indian Sign Language (ISL) recognition
VOCABULARY_LIST = [
    {
        "id": "namaste",
        "name": "Namaste",
        "label": "Namaste / Welcome",
        "category": "greetings",
        "translations": {
            "en": "Namaste / Welcome",
            "hi": "नमस्ते / स्वागत है",
            "te": "నమస్తే / స్వాగతం"
        },
        "description": "Both hands joined palms together at chest level in traditional Indian reverential greeting with slight forward head tilt.",
        "hand_config": "two_handed",
        "movement": "Upward trajectory to center chest, palms pressed firmly, brief hold.",
        "difficulty": "Beginner"
    },
    {
        "id": "thank_you",
        "name": "Thank You",
        "label": "Thank You",
        "category": "social",
        "translations": {
            "en": "Thank You",
            "hi": "धन्यवाद / शुक्रिया",
            "te": "ధన్యవాదాలు"
        },
        "description": "Dominant flat open hand touching chin/lips, then sweeping smoothly outward and downward towards the conversation partner.",
        "hand_config": "single_handed",
        "movement": "Chin touch to forward downward arc.",
        "difficulty": "Beginner"
    },
    {
        "id": "help",
        "name": "Help",
        "label": "Help / Need Assistance",
        "category": "emergency",
        "translations": {
            "en": "Help! I need assistance",
            "hi": "मदद! मुझे सहायता चाहिए",
            "te": "సహాయం! నాకు సహాయం కావాలి"
        },
        "description": "Dominant hand closed in a fist with thumb up, placed upon open palm of non-dominant hand, both hands lifting upward together.",
        "hand_config": "two_handed",
        "movement": "Base hand supports lifting fist vertically.",
        "difficulty": "Intermediate"
    },
    {
        "id": "yes",
        "name": "Yes",
        "label": "Yes / Agreed",
        "category": "affirmation",
        "translations": {
            "en": "Yes, I agree",
            "hi": "हाँ, मैं सहमत हूँ",
            "te": "అవును, నేను అంగీకరిస్తున్నాను"
        },
        "description": "Dominant hand in a soft fist nodding at wrist up and down repeatedly, mimicking a nodding head.",
        "hand_config": "single_handed",
        "movement": "Vertical wrist flexion nod 2-3 times.",
        "difficulty": "Beginner"
    },
    {
        "id": "no",
        "name": "No",
        "label": "No / Disagree",
        "category": "negation",
        "translations": {
            "en": "No, I disagree",
            "hi": "नहीं, मैं असहमत हूँ",
            "te": "కాదు / వద్దు"
        },
        "description": "Dominant hand held upright with index finger or open palm waving horizontally side to side.",
        "hand_config": "single_handed",
        "movement": "Horizontal lateral sweep side-to-side.",
        "difficulty": "Beginner"
    },
    {
        "id": "water",
        "name": "Water",
        "label": "Water / Thirsty",
        "category": "daily_needs",
        "translations": {
            "en": "I need water",
            "hi": "मुझे पानी चाहिए",
            "te": "నాకు నీళ్లు కావాలి"
        },
        "description": "Curved dominant hand tilted towards mouth mimicking sipping water from a traditional glass or vessel.",
        "hand_config": "single_handed",
        "movement": "Tilting motion toward lips.",
        "difficulty": "Beginner"
    },
    {
        "id": "food",
        "name": "Food / Hungry",
        "label": "Food / Eat",
        "category": "daily_needs",
        "translations": {
            "en": "I want food / I am hungry",
            "hi": "मुझे खाना चाहिए / मुझे भूख लगी है",
            "te": "నాకు ఆహారం కావాలి / ఆకలిగా ఉంది"
        },
        "description": "All fingertips of dominant hand bunched together, moving towards and lightly touching the lips repeatedly.",
        "hand_config": "single_handed",
        "movement": "Repeated touch to lips.",
        "difficulty": "Beginner"
    },
    {
        "id": "emergency",
        "name": "Emergency",
        "label": "Emergency / Urgent",
        "category": "emergency",
        "translations": {
            "en": "Emergency! Immediate attention needed",
            "hi": "आपातकाल! तत्काल सहायता चाहिए",
            "te": "అత్యవసరం! వెంటనే సహాయం కావాలి"
        },
        "description": "Both hands raised to chest/head height with open palms frantically crossing and waving outward in rapid alarm.",
        "hand_config": "two_handed",
        "movement": "Rapid alternating crossed sweeps at upper torso.",
        "difficulty": "Advanced"
    },
    {
        "id": "doctor",
        "name": "Doctor",
        "label": "Doctor / Medical",
        "category": "medical",
        "translations": {
            "en": "Please call a doctor",
            "hi": "कृपया डॉक्टर को बुलाएं",
            "te": "దయచేసి డాక్టర్‌ను పిలవండి"
        },
        "description": "Dominant index and middle fingertips touching the radial artery/pulse of the non-dominant wrist.",
        "hand_config": "two_handed",
        "movement": "Two taps on the inner wrist pulse point.",
        "difficulty": "Intermediate"
    },
    {
        "id": "please",
        "name": "Please",
        "label": "Please / Request",
        "category": "social",
        "translations": {
            "en": "Please",
            "hi": "कृपया",
            "te": "దయచేసి"
        },
        "description": "Dominant open hand placed flat on center of chest making a clockwise circular smoothing gesture.",
        "hand_config": "single_handed",
        "movement": "Clockwise circular chest stroke.",
        "difficulty": "Beginner"
    }
]

CLASS_NAMES = [item["id"] for item in VOCABULARY_LIST]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Model Hyperparameters & Feature Configuration
SEQUENCE_LENGTH = 30  # Number of frames in temporal sequence window (approx 1.0 second at 30 fps)
LEFT_HAND_POINTS = 21
RIGHT_HAND_POINTS = 21
POSE_KEYPOINTS = 9  # Nose, L/R shoulders, L/R elbows, L/R wrists, L/R hips
COORDS_PER_POINT = 3  # (x, y, z)

# 21*3 + 21*3 + 9*3 = 63 + 63 + 27 = 153 features per frame
NUM_FEATURES = (LEFT_HAND_POINTS + RIGHT_HAND_POINTS + POSE_KEYPOINTS) * COORDS_PER_POINT

# Inference and Confidence Settings
CONFIDENCE_THRESHOLD = 0.75  # Rejection threshold below which we fallback
CONSECUTIVE_STABILITY_FRAMES = 3  # Predictions must remain consistent across frames to commit
UNCERTAIN_FALLBACK_TEXT = "Sign not recognised confidently. Please repeat."

# File Paths
DATA_DIR = BASE_DIR / "data"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
MODEL_CHECKPOINT_PATH = SAVED_MODELS_DIR / "signbridge_isl_bilstm.pt"
MODEL_METADATA_PATH = SAVED_MODELS_DIR / "model_meta.json"
DATASET_FILE = DATA_DIR / "isl_sequences.npz"

# Server Settings
HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "8000"))
cors_env = os.getenv("CORS_ORIGINS", "")
if cors_env:
    CORS_ORIGINS = [orig.strip() for orig in cors_env.split(",") if orig.strip()]
else:
    CORS_ORIGINS = ["*"]
