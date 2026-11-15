"""
SIGNBRIDGE Open-Source Integration — Nochnack Temporal Hysteresis Filter
Adapted from Nicholas Renotte's Action Detection sequential smoothing architecture
(https://github.com/nicknochnack/ActionDetectionforSignLanguage).

Engineering Purpose:
Eliminates single-frame classification flicker, tracking jitter, and duplicate letter emissions
in continuous webcam streams by enforcing consecutive prediction agreement and hysteresis debouncing.
"""

from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class NochnackTemporalHysteresisFilter:
    """
    Temporal hysteresis smoothing and prediction debouncer.
    Adapted from Nick Nochnack's Action Detection prediction loop:
      np.unique(predictions[-10:])[0] == np.argmax(res)
    Enhanced with:
    1. Confidence-weighted window averaging.
    2. Dynamic debounce delay to prevent duplicate character spamming.
    3. Neutral/rejection flush mechanism.
    """
    def __init__(
        self,
        window_size: int = 10,
        agreement_threshold: int = 7,
        confidence_threshold: float = 0.72,
        debounce_ms: float = 800.0
    ):
        self.window_size = window_size
        self.agreement_threshold = min(window_size, agreement_threshold)
        self.confidence_threshold = confidence_threshold
        self.debounce_ms = debounce_ms

        self.prediction_history: deque = deque(maxlen=window_size)
        self.confidence_history: deque = deque(maxlen=window_size)
        self.timestamp_history: deque = deque(maxlen=window_size)

        self.last_emitted_sign: Optional[str] = None
        self.last_emitted_time: float = 0.0
        self.consecutive_neutral_count: int = 0
        self.sentence: List[str] = []

    def reset(self):
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.timestamp_history.clear()
        self.last_emitted_sign = None
        self.last_emitted_time = 0.0
        self.consecutive_neutral_count = 0

    def clear_sentence(self):
        self.sentence.clear()

    def process_frame(
        self,
        predicted_sign: str,
        confidence: float,
        timestamp_ms: float
    ) -> Tuple[Optional[str], float, str]:
        """
        Processes a single frame's raw classification and returns:
          (emitted_sign, smoothed_confidence, status_flag)
        Status flags:
          - 'EMITTED': New verified character accepted into sequence.
          - 'HOLDING': Sign confirmed but held steady (debounced from re-emitting).
          - 'UNSTABLE': Fluctuation across window (suppressed).
          - 'LOW_CONFIDENCE': Insufficient confidence (suppressed).
          - 'NEUTRAL': Neutral resting pose or rejection.
        """
        self.prediction_history.append(predicted_sign)
        self.confidence_history.append(confidence)
        self.timestamp_history.append(timestamp_ms)

        # 1. Handle explicit neutral or rejected states
        if predicted_sign in ("NEUTRAL", "REJECTED"):
            self.consecutive_neutral_count += 1
            if self.consecutive_neutral_count >= 2:
                # Sustained neutral releases debounce lock so repeat letters can emit
                self.last_emitted_sign = None
            return None, confidence, "NEUTRAL"

        self.consecutive_neutral_count = 0

        # 2. Window sufficiency check
        if len(self.prediction_history) < self.agreement_threshold:
            return None, confidence, "ACCUMULATING"

        # 3. Check consecutive agreement on last K frames (Nochnack pattern)
        recent_preds = list(self.prediction_history)[-self.agreement_threshold:]
        unique_recent = list(set(recent_preds))

        if len(unique_recent) != 1 or unique_recent[0] != predicted_sign:
            # Predictions are fluctuating across the window
            return None, float(np.mean(self.confidence_history)), "UNSTABLE"

        # 4. Check confidence threshold
        recent_confs = list(self.confidence_history)[-self.agreement_threshold:]
        avg_conf = float(np.mean(recent_confs))
        if avg_conf < self.confidence_threshold:
            return None, avg_conf, "LOW_CONFIDENCE"

        verified_sign = unique_recent[0]

        # 5. Debounce check: Prevent character spamming while user holds pose
        time_since_last = timestamp_ms - self.last_emitted_time
        if verified_sign == self.last_emitted_sign and time_since_last < self.debounce_ms:
            return verified_sign, avg_conf, "HOLDING"

        # 6. Emit verified sign!
        self.last_emitted_sign = verified_sign
        self.last_emitted_time = timestamp_ms
        self.sentence.append(verified_sign)

        return verified_sign, avg_conf, "EMITTED"

    def get_accumulated_sentence(self) -> str:
        return "".join(self.sentence)
