# SIGNBRIDGE 2.0 System Architecture

## Architectural Principles

SIGNBRIDGE 2.0 is built upon five architectural pillars:
1. **Triple-Mode Input Decoupling**: Seamlessly operates via Video Upload (Mode A), Live Camera (Mode B), or Interactive Demonstration (Mode C), guaranteeing full utility even on desktop environments without webcams.
2. **Zero-Latency Invariance**: Skeletal landmarks are transformed relative to shoulder centers and scaled by bi-deltoid Euclidean span, ensuring distance and positioning invariance.
3. **Spatio-Temporal Sequence Modeling**: Temporal motion windows (30 frames $\approx$ 1.0s at 30 FPS) capture velocities, orientations, and arcs via 1D temporal convolution, a 2-layer Bidirectional LSTM, and Temporal Self-Attention.
4. **Calibrated Confidence Thresholding**: A strict rejection floor ($\tau = 0.75$) prevents false-positive hallucinations when signers rest or make unindexed conversational gestures.
5. **Offline Edge Execution**: Pure CPU inference executes in 2.07 ms, eliminating cloud API dependencies and preserving communication privacy.

```
+---------------------------------------------------------------------------------------+
|                                CLIENT TIER (React + TypeScript)                      |
|                                                                                       |
|  [Mode A: Video Upload]   [Mode B: Live Webcam]   [Mode C: Exhibition Demonstration]  |
|  (MP4/WebM/MOV dropzone)   (Hardware-detected)     (Canonical Simulated Kinematics)   |
|            │                       │                              │                   |
|            ▼                       ▼                              ▼                   |
|  [REST Multipart Post]    [MediaPipe Tasks Vision]      [Canvas Trajectory Stream]    |
|            │                       │                              │                   |
|            ▼                       ▼                              ▼                   |
|  [Speech Synthesis Engine] <── [Gated Prediction Result] ───────┘                   |
|  (English / Hindi / Telugu)                                                           |
+────────────────────────────────────┬──────────────────────────────────────────────────+
                                     │
                                     ▼
+---------------------------------------------------------------------------------------+
|                                SERVER TIER (FastAPI 2.0)                             |
|                                                                                       |
|  [Uniform Temporal Sampler] ───► [MediaPipe Tasks Pipeline]                          |
|  (Linear space 30 keyframes)     (hand_landmarker.task & pose_landmarker.task)        |
|                                            │                                          |
|                                            ▼                                          |
|                                  [153-D Feature Normalizer]                           |
|                                            │                                          |
|                                            ▼                                          |
|                                  [1D-Conv Velocity Extractor]                         |
|                                            │                                          |
|                                            ▼                                          |
|                                  [2-Layer Bidirectional LSTM]                         |
|                                            │                                          |
|                                            ▼                                          |
|                                  [Temporal Attention Pooling]                         |
|                                            │                                          |
|                                            ▼                                          |
|                                  [Calibrated Softmax Head]                            |
|                                            │                                          |
|                                            +── Conf >= 0.75 ? Output Sign : Uncertain |
+---------------------------------------------------------------------------------------+
```

## Detailed Data Pipeline

1. **Acquisition**:
   - **Mode A (Video Upload)**: Decodes variable-duration videos ($D \in [1.0, 10.0]$ s) and samples exactly 30 equidistant keyframes via `np.linspace(0, total_frames - 1, 30)`.
   - **Mode B (Live Camera)**: Captures frames at 640x480 resolution via HTML5 `getUserMedia`.
   - **Mode C (Demonstration)**: Streams canonical kinematic trajectories at 30 FPS.
2. **Extraction**:
   - Left Hand: 21 points $\times$ 3 $(x,y,z) = 63$ features
   - Right Hand: 21 points $\times$ 3 $(x,y,z) = 63$ features
   - Upper Pose: 9 points $\times$ 3 $(x,y,z) = 27$ features
   - Total = 153 features per frame.
3. **Normalization**:
   - $C = (P_{L\_shoulder} + P_{R\_shoulder}) / 2$
   - $S = \|P_{L\_shoulder} - P_{R\_shoulder}\|_2$
   - $P_{norm} = (P - C) / S$
4. **Inference**: PyTorch model predicts class distribution in 2.07 ms on CPU.
5. **Delivery**: Results delivered via REST JSON or WebSocket, triggering browser SpeechSynthesis.
