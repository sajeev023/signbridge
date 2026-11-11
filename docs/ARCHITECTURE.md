# SIGNBRIDGE System Architecture

## Architectural Principles

SIGNBRIDGE is built upon four architectural pillars:
1. **Zero-Latency Invariance**: Skeletal landmarks are transformed relative to shoulder centers to ensure scale and position invariance across differing camera distances.
2. **Spatio-Temporal Sequence Modeling**: Instead of single-frame classification, temporal sliding windows (30 frames $\approx$ 1.0s at 30 FPS) capture velocity and trajectory.
3. **Calibrated Confidence Thresholding**: A strict rejection threshold ($\tau = 0.75$) prevents false-positive hallucinations when signers rest or make non-sign movements.
4. **Offline Edge Execution**: Pure CPU inference executes in ~2.07 ms, eliminating expensive cloud dependencies and preserving privacy.

```
+-------------------------------------------------------------------------+
|                           CLIENT TIER (Browser)                        |
|                                                                         |
|  [User Webcam]                                                          |
|        |                                                                |
|        v                                                                |
|  [MediaPipe Holistic / Hands]                                           |
|        |                                                                |
|        v                                                                |
|  [153-D Normalized Coordinates] ---> [WebSocket / REST Client]         |
|                                                  |                      |
|  [Speech Synthesis API] <--- [Gated Prediction] <+                      |
|  (English / Hindi / Telugu)                                             |
+--------------------------------------------------|----------------------+
                                                   |
                                                   v
+-------------------------------------------------------------------------+
|                           SERVER TIER (FastAPI)                         |
|                                                                         |
|  [FIFO Sequence Buffer (30 Frames)]                                     |
|        |                                                                |
|        v                                                                |
|  [1D-Conv Layer (Feature Velocity)]                                     |
|        |                                                                |
|        v                                                                |
|  [2-Layer Bidirectional LSTM (256-D Context)]                           |
|        |                                                                |
|        v                                                                |
|  [Temporal Self-Attention Pooling]                                      |
|        |                                                                |
|        v                                                                |
|  [Softmax Classifier (10 ISL Classes)]                                  |
|        |                                                                |
|        +---> Confidence >= 0.75 ? Return Translation : Return Fallback  |
+-------------------------------------------------------------------------+
```

## Detailed Data Pipeline

1. **Acquisition**: HTML5 video captures 640x480 resolution at 30 FPS.
2. **Extraction**: MediaPipe captures:
   - Left Hand: 21 points $\times$ 3 $(x,y,z) = 63$ features
   - Right Hand: 21 points $\times$ 3 $(x,y,z) = 63$ features
   - Upper Pose: 9 points $\times$ 3 $(x,y,z) = 27$ features
   - Total = 153 features per frame.
3. **Normalization**:
   - $C = (P_{L\_shoulder} + P_{R\_shoulder}) / 2$
   - $S = \|P_{L\_shoulder} - P_{R\_shoulder}\|_2$
   - $P_{norm} = (P - C) / S$
4. **Sequence Queue**: 30 normalized frames are maintained in a sliding FIFO deque.
5. **Inference**: PyTorch model predicts class distribution in 2.07 ms on CPU.
6. **Delivery**: Results streamed back via WebSocket or REST JSON.
