# SIGNBRIDGE — DEVELOPER INTEGRATION NOTES: KINIVI MEDIAPIPE PATTERNS

**Scope:** Internal engineering guide for maintaining, extending, and debugging the Kinivi-inspired point-history temporal trajectory engine and static regularization modules.

---

## 1. Architectural Overview

The SIGNBRIDGE recognition pipeline integrates two specialized neural inference engines:

```
[Camera Video Frame]
         │
         ▼
[MediaPipe 21 Landmarks Extraction]
         │
         ▼
[PersistentHandTracker: One-Euro Speed-Adaptive Filtering]
         │
         ▼
[TrackingQualityEngine: Gating & Reliability Assessment]
         │
    Reliable?
    ├── NO  ──► Emit 'NEUTRAL' (Suppress false positives)
    └── YES ──► Route by Active Hand Mode
                     │
        ┌────────────┴────────────┐
        │                         │
[1 Hand (Single-Hand Chart)]   [2 Hands (ISLRTC Bimanual)]
        │                         │
        ▼                         ▼
Dynamic Trajectory Check       Dual-Hand 44 Invariant Features
(Engine B: Point History)                 │
  ├── Dynamic J/Z Triggered?              ▼
  │    YES ──► Emit 'J'/'Z'        Dual-Hand MLP
  │    NO                                 │
  ▼                                       ▼
Static 42 Biomechanical Features   ISLRTC Output Class
  │
  ▼
Static MLP (Engine A)
  │
  ▼
Static Alphabet Output Class (A-I, K-Y, NEUTRAL)
```

---

## 2. Point-History Trajectory Buffer Mechanics

### Circular Buffer Specification
The point-history buffer is implemented client-side in `frontend/src/services/dynamicTrajectoryEngine.ts` and mirrored in Python in `backend/trajectory_point_history_engine.py`:

- **Capacity:** Fixed rolling window of 32 frames (minimum 8 frames required for evaluation).
- **Recorded Variables Per Frame:**
  ```typescript
  interface BufferEntry {
    x: number;          // Screen X coordinate in [0, 1]
    y: number;          // Screen Y coordinate in [0, 1]
    z: number;          // Depth Z coordinate from MediaPipe
    palmScale: number;  // ||MCP_9 - Wrist|| Euclidean distance
    timestamp: number;  // Performance.now() millisecond timestamp
    handId: number;     // Persistent track identifier
  }
  ```

### Continuity Guarantee
Optical occlusion or rapid hand movement may cause tracking dropout or ID swaps. The buffer enforces strict temporal continuity before computing features:
```typescript
public isContinuous(maxGapMs: number = 300): boolean {
  if (this.buffer.length < 8) return false;
  const firstId = this.buffer[0].handId;
  for (let i = 1; i < this.buffer.length; i++) {
    if (this.buffer[i].handId !== firstId) return false;
    if (this.buffer[i].timestamp - this.buffer[i - 1].timestamp > maxGapMs) {
      return false;
    }
  }
  return true;
}
```
If a temporal dropout (> 300ms) or hand switch occurs, the buffer is invalidated, preventing erratic trajectory classifications.

---

## 3. Palm-Scale Normalization & Distance Invariance

### The Kinivi Limitation
In Kinivi's original repository, points were normalized as:
$$x_{\text{norm}} = \frac{x - x_0}{\text{image\_width}}, \quad y_{\text{norm}} = \frac{y - y_0}{\text{image\_height}}$$
Consequently, a hand close to the camera produced a stroke 3× to 5× larger in feature space than a hand far from the camera, severely distorting classifier representations.

### The SIGNBRIDGE Solution
SIGNBRIDGE computes the average Euclidean palm scale over the stroke:
$$\bar{s} = \frac{1}{N} \sum_{i=0}^{N-1} \max\left(0.035, \|\text{MCP}_9 - \text{wrist}\|_i\right)$$
Each point is expressed in scale-normalized displacement units relative to the stroke origin:
$$\Delta x_i = \frac{x_i - x_0}{\bar{s}}, \quad \Delta y_i = \frac{y_i - y_0}{\bar{s}}$$

**Empirical Result:** Trajectory features remain invariant across camera distances ranging from 0.35m to 2.50m.

---

## 4. Variable-Speed Arc-Length Resampling

Different signers execute 'J' and 'Z' at different physical tempos. A fast stroke may take 10 frames, while an expressive slow stroke may take 28 frames.

The function `resampleTrajectory2D` resamples arbitrary trajectory lengths into exactly 16 uniform arc-length intervals:
1. Compute cumulative Euclidean distance along the stroke:
   $$d_i = \sum_{j=1}^i \sqrt{(\Delta x_j - \Delta x_{j-1})^2 + (\Delta y_j - \Delta y_{j-1})^2}, \quad d_0 = 0$$
2. Total stroke path length: $L = d_{N-1}$.
3. Target distance for step $k \in \{0, \dots, 15\}$:
   $$t_k = \frac{k}{15} \cdot L$$
4. Locate enclosing segment $[j, j+1]$ such that $d_j \le t_k \le d_{j+1}$ and linearly interpolate:
   $$\vec{p}(t_k) = \vec{p}_j + \frac{t_k - d_j}{d_{j+1} - d_j} (\vec{p}_{j+1} - \vec{p}_j)$$

Flattening the 16 2D coordinates yields a fixed 32-element vector:
$$[x_0, y_0, x_1, y_1, \dots, x_{15}, y_{15}]$$

---

## 5. Anatomical Handshape Context Gating

To prevent arbitrary arm or body gestures from triggering alphabet characters, the trajectory engine enforces strict biological hand posture prerequisites:

### Context Rule for 'J' (`isJHandshapeContext`)
- **Pinky Finger:** Fully extended ($\Delta_{\text{ext}} > 0.40$).
- **Index & Middle Fingers:** Fully curled into palm ($\text{curl} < 0.65$).
- **Tracked Landmark:** Landmark 20 (Pinky Fingertip).

### Context Rule for 'Z' (`isZHandshapeContext`)
- **Index Finger:** Fully extended ($\Delta_{\text{ext}} > 0.45$).
- **Middle & Pinky Fingers:** Fully curled into palm ($\text{curl} < 0.65$).
- **Tracked Landmark:** Landmark 8 (Index Fingertip).

### Static Threshold Suppression
If the total path length $L < 0.70 \times \bar{s}$, the hand is identified as stationary (e.g. resting or holding a static letter). The dynamic engine yields control to Engine A, allowing static 'I' or 'D' to be recognized without false dynamic interference.

---

## 6. Cooldown State Machine & Trigger Protocol

Once a dynamic sign is recognized with high confidence ($\ge 0.85$):
1. The dynamic sign is returned to `alphabetInferenceEngine.ts` with `isDynamic: true`.
2. A cooldown timestamp is set: `cooldownUntil = timestampMs + 750`.
3. Both fingertip buffers (`pinkyBuffer` and `indexBuffer`) are cleared immediately.
4. During the 750ms cooldown window, further dynamic evaluations are suppressed to prevent the hand retraction motion from re-triggering.

---

## 7. Adding New Dynamic Signs or Modifying Buffers

To extend the trajectory engine for additional dynamic signs (e.g., words, numbers, or custom gestures):
1. **Record Episodes:** Collect sample episodes using `backend/collect_trajectory_dataset.py`.
2. **Update Context Guard:** Add an anatomical filter in `dynamicTrajectoryEngine.ts` specifying required extended/curled finger states.
3. **Train Model:** Run `backend/train_and_evaluate_trajectories.py` with the updated class list.
4. **Deploy Weights:** The training script automatically exports the updated weights to both `backend/saved_models/dynamic_alphabet_model.json` and `frontend/src/services/dynamic_alphabet_model.json`.
5. **Run Integration Tests:** Execute `node tests/test_frontend_inference.mjs` and `python -m pytest backend/tests/test_dynamic_trajectory_suite.py` to confirm zero regression.
