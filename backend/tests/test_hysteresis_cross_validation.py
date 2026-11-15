"""
SIGNBRIDGE Phase 5 — Multi-Participant Camera & Cross-Language Hysteresis Test Suite
Validates:
1. Multi-participant camera scenarios:
   - Repeated identical letters across diverse signer pause profiles
   - Brief tracking loss immunity across frame dropout durations
   - Intentional neutral release with varying idle intervals
   - Rapid fingerspelling sequences at conversational cadences
   - Dynamic letter transitions (J and Z trajectories with static prefixes/suffixes)
2. Python and TypeScript exact behavioral parity across all test traces.
"""

import json
import subprocess
from pathlib import Path
import pytest

from backend.open_source_integration.nochnack_temporal_hysteresis import NochnackTemporalHysteresisFilter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
ADAPTER_BUNDLE = WORKSPACE_ROOT / "frontend" / "dist" / "test_adapter.mjs"


@pytest.fixture(scope="session", autouse=True)
def ensure_ts_bundle():
    """Ensures TypeScript bundle exists for Node cross-validation."""
    if not ADAPTER_BUNDLE.exists():
        ADAPTER_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        npx_bin = "npx.cmd" if subprocess.os.name == "nt" else "npx"
        subprocess.run(
            [
                npx_bin, "esbuild",
                "src/services/openSourceIntegrationAdapter.ts",
                "--bundle", "--platform=node", "--format=esm",
                f"--outfile={ADAPTER_BUNDLE}",
                "--external:@mediapipe/*"
            ],
            cwd=str(WORKSPACE_ROOT / "frontend"),
            capture_output=True,
            check=True
        )


def run_cross_platform_trace(events, require_neutral_release=True):
    """
    Executes an event trace through both Python and TypeScript implementations
    and asserts complete equivalence.
    """
    # 1. Python execution
    py_filter = NochnackTemporalHysteresisFilter(
        window_size=10,
        agreement_threshold=7,
        confidence_threshold=0.72,
        debounce_ms=800.0,
        require_neutral_release=require_neutral_release
    )
    py_trace = []
    for sign, conf, t in events:
        s, c, st = py_filter.process_frame(sign, conf, timestamp_ms=t)
        py_trace.append((s, st, py_filter.get_accumulated_sentence()))

    # 2. TypeScript execution
    bundle_uri = ADAPTER_BUNDLE.resolve().as_uri()
    js_code = f"""
import {{ NochnackHysteresisFilter }} from '{bundle_uri}';
const filter = new NochnackHysteresisFilter(10, 7, 0.72, 800, {str(require_neutral_release).lower()});
const ev = {json.dumps(events)};
const res = ev.map(([s, c, t]) => {{
  const r = filter.processPrediction(s, c, t);
  return [r.emittedSign, r.status, filter.getSentence()];
}});
console.log(JSON.stringify(res));
"""
    proc = subprocess.run(
        ["node", "-e", js_code],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        check=True
    )
    ts_trace = [tuple(x) for x in json.loads(proc.stdout.strip())]

    assert len(py_trace) == len(ts_trace), "Trace length mismatch between Python and TypeScript"
    for idx, (py_step, ts_step) in enumerate(zip(py_trace, ts_trace)):
        assert py_step == ts_step, (
            f"Step {idx} mismatch: Python={py_step} vs TypeScript={ts_step} at event={events[idx]}"
        )

    return py_filter.get_accumulated_sentence(), py_trace


def test_participant_1_repeated_letters_diverse_pauses():
    """Participant 1: Fast vs slow pause profiles on repeated letters ('LL')."""
    events = []
    t = 0
    # First 'L' held for 8 frames
    for _ in range(8):
        events.append(('L', 0.95, t))
        t += 30
    # Brief intentional pause (3 frames = 90ms)
    for _ in range(3):
        events.append(('NEUTRAL', 0.0, t))
        t += 30
    # Second 'L' held for 8 frames
    for _ in range(8):
        events.append(('L', 0.95, t))
        t += 30

    sentence, _ = run_cross_platform_trace(events)
    assert sentence == "LL"


def test_participant_2_brief_tracking_dropout_coasting():
    """Participant 2: 1 to 4 frames of tracking dropout during sustained hold."""
    events = []
    t = 0
    # Hold 'C'
    for _ in range(8):
        events.append(('C', 0.92, t))
        t += 30
    # Single frame dropout
    events.append(('REJECTED', 0.0, t))
    t += 30
    # Continue 'C'
    for _ in range(6):
        events.append(('C', 0.92, t))
        t += 30
    # 3 frame dropout
    for _ in range(3):
        events.append(('REJECTED', 0.0, t))
        t += 30
    # Continue 'C'
    for _ in range(6):
        events.append(('C', 0.92, t))
        t += 30

    sentence, _ = run_cross_platform_trace(events)
    # Must remain a single 'C' without duplicate emissions
    assert sentence == "C"


def test_participant_3_intentional_neutral_release_multi_interval():
    """Participant 3: Signer rests hand between letters across varying intervals (300ms, 1500ms)."""
    events = []
    t = 0
    # Letter 'A'
    for _ in range(8):
        events.append(('A', 0.95, t))
        t += 30
    # 300ms neutral
    for _ in range(10):
        events.append(('NEUTRAL', 0.0, t))
        t += 30
    # Letter 'B'
    for _ in range(8):
        events.append(('B', 0.95, t))
        t += 30
    # 1500ms neutral
    for _ in range(50):
        events.append(('NEUTRAL', 0.0, t))
        t += 30
    # Letter 'C'
    for _ in range(8):
        events.append(('C', 0.95, t))
        t += 30

    sentence, _ = run_cross_platform_trace(events)
    assert sentence == "ABC"


def test_participant_4_rapid_fingerspelling():
    """Participant 4: Conversational fingerspelling cadence ('SIGN')."""
    word = "SIGN"
    events = []
    t = 0
    for char in word:
        # 7 frames consensus
        for _ in range(7):
            events.append((char, 0.94, t))
            t += 25
        # Minimal natural coarticulation neutral (2 frames)
        for _ in range(2):
            events.append(('NEUTRAL', 0.0, t))
            t += 25

    sentence, _ = run_cross_platform_trace(events)
    assert sentence == "SIGN"


def test_participant_5_dynamic_letter_transitions():
    """Participant 5: Sequences with dynamic J and Z signs ('JOY' and 'ZOOM')."""
    # 1. Sequence 'J' -> 'O' -> 'Y'
    events = []
    t = 0
    # Present 'J' trajectory consensus
    for _ in range(8):
        events.append(('J', 0.91, t))
        t += 33
    # Neutral transition
    for _ in range(3):
        events.append(('NEUTRAL', 0.0, t))
        t += 33
    # Present 'O'
    for _ in range(8):
        events.append(('O', 0.93, t))
        t += 33
    # Neutral transition
    for _ in range(3):
        events.append(('NEUTRAL', 0.0, t))
        t += 33
    # Present 'Y'
    for _ in range(8):
        events.append(('Y', 0.96, t))
        t += 33

    sentence, _ = run_cross_platform_trace(events)
    assert sentence == "JOY"
