# COMPLETE ALPHABET HELD-OUT EVALUATION
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Branch:** `experiment/complete-physical-alphabet-coverage`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 08:54:07 UTC  
**Evaluation Cohort:** Strictly Held-Out Participants (Signer 4 in Multi-Signer, Signers 3–6 in Physical)  

---

## 1. Held-Out Evaluation Summary

```
+--------------------------------------------------------------------------------------------------------+
| Cohort                     | Samples   | Baseline Accuracy  | Calibrated Accuracy| Macro F1 (Calibrated) |
+--------------------------------------------------------------------------------------------------------+
| Physical Held-Out (S3..S6) | 27 smps   | 77.78% (21/27)     | 85.19% (23/27)     | 0.8491                |
| Physical Combined (S1..S6) | 42 smps   | 83.33% (35/42)     | 90.48% (38/42)     | 0.8876                |
| Multi-Signer Held-Out (S4) | 1000 smps | 96.80% (968/1000)  | 96.80% (968/1000)  | 0.9674                |
| Dynamic Trajectory Held-Out| 240 eps   | 99.17% (238/240)   | 99.17% (238/240)   | 0.9915                |
+--------------------------------------------------------------------------------------------------------+
```

---

## 2. Full 26-Letter Class-by-Class Held-Out Performance Matrix

```
+-------+-----------+-------------------+--------------------+------------------------+-------------------+
| Class | Modality  | Held-Out Samples  | Precision (Calib)  | Recall (Calib)         | Status            |
+-------+-----------+-------------------+--------------------+------------------------+-------------------+
|   A   | Static    | 2 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED         |
|   B   | Static    | 1 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED         |
|   C   | Static    | 6 physical / 40 ms| 0.8000             | 0.8000 (2 -> Y conf)   | VALIDATED (Error) |
|   D   | Static    | 1 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED         |
|   E   | Static    | 0 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED (Multi) |
|   F   | Static    | 1 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED         |
|   G   | Static    | 0 physical / 40 ms| 0.9524             | 0.9500                 | PENDING PHYSICAL  |
|   H   | Static    | 0 physical / 40 ms| 0.9750             | 0.9750                 | PENDING PHYSICAL  |
|   I   | Static    | 0 physical / 40 ms| 0.9756             | 1.0000                 | PENDING PHYSICAL  |
|   J   | Dynamic   | 0 physical / 80 ep| 0.9875             | 0.9875                 | PENDING PHYSICAL  |
|   K   | Static    | 0 physical / 40 ms| 0.9500             | 0.9500                 | PENDING PHYSICAL  |
|   L   | Static    | 6 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED         |
|   M   | Static    | 0 physical / 40 ms| 0.9268             | 0.9500                 | PENDING PHYSICAL  |
|   N   | Static    | 0 physical / 40 ms| 0.9487             | 0.9250                 | PENDING PHYSICAL  |
|   O   | Static    | 1 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED         |
|   P   | Static    | 0 physical / 40 ms| 0.9500             | 0.9500                 | PENDING PHYSICAL  |
|   Q   | Static    | 0 physical / 40 ms| 0.9744             | 0.9500                 | PENDING PHYSICAL  |
|   R   | Static    | 1 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED         |
|   S   | Static    | 0 physical / 40 ms| 0.9512             | 0.9750                 | PENDING PHYSICAL  |
|   T   | Static    | 0 physical / 40 ms| 0.9250             | 0.9250                 | PENDING PHYSICAL  |
|   U   | Static    | 0 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED (Multi) |
|   V   | Static    | 3 physical / 40 ms| 1.0000             | 1.0000                 | VALIDATED         |
|   W   | Static    | 0 physical / 40 ms| 0.9756             | 1.0000                 | PENDING PHYSICAL  |
|   X   | Static    | 0 physical / 40 ms| 0.9744             | 0.9500                 | PENDING PHYSICAL  |
|   Y   | Static    | 4 physical / 40 ms| 0.7500 (2 FP from C| 1.0000                 | VALIDATED (Conf)  |
|   Z   | Dynamic   | 0 physical / 80 ep| 1.0000             | 0.9875                 | PENDING PHYSICAL  |
+-------+-----------+-------------------+--------------------+------------------------+-------------------+
```

---

## 3. Systematic Failure Taxonomy (Known Limitations)

1. **Letter C vs Y Confusion (2 Samples):**
   - Occurs when a signer exhibits lateral thumb-pinky abduction ($>1.25$ palm spans). Planar 2D projection overlaps with Y.
   - Remediation: 3D volumetric curvature check.
2. **Letter M / N / T Thumb Tuck Occlusion:**
   - Frontal camera viewing occludes thumb tip beneath folded dorsal finger segments.
   - Lowest individual class recalls on held-out multi-signer data: M (95.0%), N (92.5%), T (92.5%).
3. **Overhead Lighting Degradation (Letter O):**
   - Deep shadows in the palm cup reduce single-camera MediaPipe depth confidence. Correctly handled by hysteresis `LOW_CONFIDENCE` hold.
