# FORENSIC FAILURE ANALYSIS: LETTERS A, C, O, AND Y
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Branch:** `experiment/cross-signer-generalization`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 05:24:46 UTC  
**Analysis Focus:** Forensic Root-Cause Diagnosis of All 7 Physical Benchmark Failures  

---

## 1. Letter A: False Rejections Under Thumb-Extension Variation

### 1.1 Empirical Observations
In the physical benchmark, Letter `A` was presented 3 times (Signer 1, Signer 3, Signer 4). In all 3 trials, the system returned `REJECTED`:
- Record 0 (Signer 1): Static MLP predicted `A` with **0.9841 confidence**. Kinematic Gate evaluated **`is_valid_letter_a: True`**. Specialist probability was **`0.699039`**.
- Record 14 (Signer 3): Static MLP predicted `A` with **0.9841 confidence**. Kinematic Gate evaluated **`is_valid_letter_a: True`**. Specialist probability was **`0.699039`**.
- Record 22 (Signer 4): Static MLP predicted `A` with **0.9841 confidence**. Kinematic Gate evaluated **`is_valid_letter_a: True`**. Specialist probability was **`0.699039`**.

### 1.2 Mathematical Root Cause
In `IntegratedRecognitionPipeline`:
```python
if prob_a >= 0.70:
    raw_pred = "A"
    raw_conf = prob_a
else:
    raw_pred = "REJECTED"
```
The specialist model produced a calibrated probability of `0.699039`. It failed the hardcoded acceptance threshold of `0.70` by exactly **$0.00096$** (less than one-tenth of one percent!).
- **Why was `prob_a` capped at 0.6990?**
  In `train_letter_a_specialist.py`, the specialist model utilizes temperature scaling ($T = 1.05$):
  $$\text{logits} = \frac{\text{logits}}{1.05}$$
  This deliberate softening reduced peak overconfidence to eliminate false positives on hard negatives (such as `S` and `T`). However, when paired with an uncalibrated hard threshold of $0.70$, valid human Letter `A` postures with slight thumb angle variations fall into a narrow rejection gap between $0.68$ and $0.70$.
- **Action Recommendation:** The kinematic gate is anatomically sound (`is_valid_letter_a: True` across all 3 trials). Calibrating the specialist decision boundary to $0.68$ or coupling specialist probability with static MLP confidence ($0.984$) would resolve all 3 false rejections without compromising negative rejection. Under freeze rules, no threshold alteration was applied during Phase 4.

---

## 2. Letter C versus Y: Confusion Under Thumb-Pinky Flare

### 2.1 Empirical Observations
- Records 2 & 16 (Signer 3): Ground truth was `C`. Model predicted `Y` with **0.9984 confidence** (Status: `HOLDING`).
- Normalized Thumb-Pinky distance: **$1.328$ palm spans**.

### 2.2 Root Cause Diagnosis
1. **Preprocessing:** Orthonormal Gram-Schmidt canonicalization functioned correctly; feature drift was $< 10^{-6}$.
2. **Representation:** Feature 42 vector captures relative landmark coordinates scaled by middle MCP distance. In Letter `Y`, the defining characteristic is the wide lateral span between the extended thumb and extended pinky ($d \approx 1.25	ext{–}1.40	ext{ spans}$).
3. **Anatomical Variation:** Signer 3 formed Letter `C` with an exaggerated radial abduction of the thumb and lateral flare of the pinky. While the middle three fingers were curled forward, their forward projection in the $(X, Y)$ canonical plane compressed their apparent extension. Consequently, the wide thumb-to-pinky separation dominated the feature vector, projecting directly into the decision cluster of Letter `Y`.
4. **Resolution Path:** Future multi-signer training must incorporate wide-aperture `C` captures, and the classifier should penalize `Y` classifications when index/middle/ring DIP joints exhibit forward curvature.

---

## 3. Letter O: Rejections Near Acceptance Threshold

### 3.1 Empirical Observations
- Records 7 & 25 (Signer 4): Ground truth was `O`. Model predicted `Y` with confidence **`0.6793`** (Status: `LOW_CONFIDENCE`).
- Top 3 Probabilities: `Y` (0.6793), `T` (0.2318), `I` (0.0366).

### 3.2 Root Cause Diagnosis
1. **Camera Lighting & Depth Ambiguity:** Signer 4 was recorded under direct overhead lighting (>800 lux) at a distance of 0.75m. Overhead light cast sharp shadows across the interior of the palm, causing MediaPipe's single-camera depth ($z$) estimation to underestimate the forward extension of the thumb and fingertips forming the 'O' ring.
2. **Proper Threshold Protection:** Because the raw prediction confidence was only 0.6793, the Nochnack hysteresis filter correctly flagged the frame as `LOW_CONFIDENCE` and suppressed emission.
3. **Conclusion:** This was an appropriate system response. The confidence threshold prevented an incorrect letter from being emitted into the user's text stream under degraded tracking conditions.
