"""
SIGNBRIDGE Phase 4 — Hysteresis Robustness Test Suite
Exhaustively tests the 8 mandatory physical camera interaction scenarios:
1. Continuous static sign hold (suppression of auto-repeat spamming).
2. Deliberate neutral release (intentional repeated letter emission).
3. Repeated identical letters sequence (e.g. triple 'A' spelling).
4. Brief tracking dropout coasting (sensor glitch immunity).
5. Dropout followed by confirmed neutral (proper release sequence).
6. Dropout during intended transition (robustness against tracking drop during motion).
7. Rapid different-letter transitions (immediate emission on consensus without debounce penalty).
8. Long neutral intervals (graceful idle-to-active recovery).
"""

import pytest
from backend.open_source_integration.nochnack_temporal_hysteresis import NochnackTemporalHysteresisFilter


@pytest.fixture
def hysteresis_filter():
    return NochnackTemporalHysteresisFilter(
        window_size=10,
        agreement_threshold=7,
        confidence_threshold=0.72,
        debounce_ms=800.0,
        require_neutral_release=True
    )


def test_scenario_1_continuous_static_sign(hysteresis_filter):
    """Scenario 1: Holding sign steadily for 1500ms emits ONCE, maintaining HOLDING."""
    emissions = []
    for t in range(0, 1500, 30):
        sign, conf, status = hysteresis_filter.process_frame("B", 0.95, timestamp_ms=t)
        if status == "EMITTED":
            emissions.append((sign, t))

    assert len(emissions) == 1
    assert emissions[0] == ("B", 180)
    assert hysteresis_filter.get_accumulated_sentence() == "B"


def test_scenario_2_deliberate_neutral_release(hysteresis_filter):
    """Scenario 2: Intentional resting hand (>=3 frames NEUTRAL) permits repeated letter."""
    # First 'B'
    for t in range(0, 240, 30):
        hysteresis_filter.process_frame("B", 0.95, timestamp_ms=t)
    assert hysteresis_filter.get_accumulated_sentence() == "B"

    # Resting hand for 4 frames (120ms)
    for t in range(240, 360, 30):
        hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=t)

    # Second 'B'
    for t in range(360, 600, 30):
        hysteresis_filter.process_frame("B", 0.95, timestamp_ms=t)

    assert hysteresis_filter.get_accumulated_sentence() == "BB"


def test_scenario_3_repeated_identical_letters(hysteresis_filter):
    """Scenario 3: Sequential triple repetition ('A' -> neutral -> 'A' -> neutral -> 'A')."""
    t = 0
    for _ in range(3):
        # Present 'A'
        for _ in range(8):
            hysteresis_filter.process_frame("A", 0.95, timestamp_ms=t)
            t += 30
        # Present neutral release (4 frames)
        for _ in range(4):
            hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=t)
            t += 30

    assert hysteresis_filter.get_accumulated_sentence() == "AAA"


def test_scenario_4_brief_tracking_dropout_coasting(hysteresis_filter):
    """Scenario 4: 2 frames of REJECTED tracking loss must NOT cause duplicate re-emission."""
    # Present 'B'
    for t in range(0, 240, 30):
        hysteresis_filter.process_frame("B", 0.95, timestamp_ms=t)
    assert hysteresis_filter.get_accumulated_sentence() == "B"

    # 2 dropped frames (60ms) of sensor dropout
    hysteresis_filter.process_frame("REJECTED", 0.0, timestamp_ms=240)
    hysteresis_filter.process_frame("REJECTED", 0.0, timestamp_ms=270)

    # Hand reacquired holding 'B'
    for t in range(300, 540, 30):
        hysteresis_filter.process_frame("B", 0.95, timestamp_ms=t)

    assert hysteresis_filter.get_accumulated_sentence() == "B"


def test_scenario_5_dropout_followed_by_neutral(hysteresis_filter):
    """Scenario 5: Dropout followed by confirmed neutral properly releases debounce lock."""
    # Present 'C'
    for t in range(0, 240, 30):
        hysteresis_filter.process_frame("C", 0.95, timestamp_ms=t)
    assert hysteresis_filter.get_accumulated_sentence() == "C"

    # Tracking dropout for 2 frames
    hysteresis_filter.process_frame("REJECTED", 0.0, timestamp_ms=240)
    hysteresis_filter.process_frame("REJECTED", 0.0, timestamp_ms=270)

    # Confirmed neutral for 4 frames
    for t in range(300, 420, 30):
        hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=t)

    # Present 'C' again
    for t in range(420, 660, 30):
        hysteresis_filter.process_frame("C", 0.95, timestamp_ms=t)

    assert hysteresis_filter.get_accumulated_sentence() == "CC"


def test_scenario_6_dropout_during_intended_transition(hysteresis_filter):
    """Scenario 6: Transient sensor drop during neutral transition does not block subsequent letter."""
    # Present 'D'
    for t in range(0, 240, 30):
        hysteresis_filter.process_frame("D", 0.95, timestamp_ms=t)

    # Neutral transition with 1 dropped frame in between: NEUTRAL -> REJECTED -> NEUTRAL -> NEUTRAL
    hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=240)
    hysteresis_filter.process_frame("REJECTED", 0.0, timestamp_ms=270)
    hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=300)
    hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=330)
    hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=360)

    # Present 'D' again
    for t in range(390, 630, 30):
        hysteresis_filter.process_frame("D", 0.95, timestamp_ms=t)

    assert hysteresis_filter.get_accumulated_sentence() == "DD"


def test_scenario_7_rapid_different_letter_transitions(hysteresis_filter):
    """Scenario 7: Switching between different letters occurs immediately without debounce penalty."""
    t = 0
    letters = ["A", "B", "C", "D", "E"]
    for l in letters:
        for _ in range(7):
            hysteresis_filter.process_frame(l, 0.95, timestamp_ms=t)
            t += 30

    assert hysteresis_filter.get_accumulated_sentence() == "ABCDE"


def test_scenario_8_long_neutral_interval(hysteresis_filter):
    """Scenario 8: Long idle gap (5000ms) between signs maintains clean state."""
    # Present 'X'
    for t in range(0, 240, 30):
        hysteresis_filter.process_frame("X", 0.95, timestamp_ms=t)
    assert hysteresis_filter.get_accumulated_sentence() == "X"

    # 5 seconds of resting idle hand
    hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=1000)
    hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=3000)
    hysteresis_filter.process_frame("NEUTRAL", 0.0, timestamp_ms=5000)

    # Present 'Y' at t=5100ms
    for t in range(5100, 5340, 30):
        hysteresis_filter.process_frame("Y", 0.95, timestamp_ms=t)

    assert hysteresis_filter.get_accumulated_sentence() == "XY"
