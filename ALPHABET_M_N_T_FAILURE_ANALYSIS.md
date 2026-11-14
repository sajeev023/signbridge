# SIGNBRIDGE — OPERATION HUMAN DATA
## Fist-Tuck Confusion Forensic Analysis: Letters M, N, and T

**Evaluation Baseline:** Synthetic Multi-Signer Test Split (Signer_4, N=1,000)  
**True M Confusions:** 2 samples misclassified as N (Precision: 88.37%, Recall: 95.00%)  
**True N Confusions:** 5 samples misclassified as M (Precision: 85.37%, Recall: 87.50%)  
**True T Confusions:** 5 samples misclassified as N (Precision: 100.0%, Recall: 87.50%)  

---

## 1. Anatomical Distinction Baseline

In the ASL manual alphabet, letters M, N, and T form the fist-tuck family. All three handshapes feature a closed fist with curled fingers, distinguished strictly by which fingers fold over the thumb:

| Letter | Thumb Tip Placement | Number of Overlapping Digits | Visual Knuckle "Humps" |
| :---: | :--- | :---: | :---: |
| **M** | Tucked under index, middle, and ring; emerges between ring (16) and pinky (20) | 3 Digits | 3 Humps |
| **N** | Tucked under index and middle; emerges between middle (12) and ring (16) | 2 Digits | 2 Humps |
| **T** | Tucked under index only; emerges between index (8) and middle (12) | 1 Digit | 1 Hump |

---

## 2. Quantitative Fisher Separation Analysis

Fisher separation scores ($d = |\mu_1 - \mu_2| / \sqrt{\sigma_1^2 + \sigma_2^2}$) between the triplet:

```
===========================================================================
FISHER SEPARATION RANKINGS:
===========================================================================
Pair 'M' vs 'N':
1. ext_thumb:                  d = 7.66  (M: 0.278 ± 0.009 vs N: 0.364 ± 0.007)
2. canonical_thumb_y:          d = 6.80  (M: 0.702 ± 0.013 vs N: 0.817 ± 0.011)
3. dist_tip_thumb_pinky:       d = 6.17  (M: 0.489 ± 0.010 vs N: 0.580 ± 0.010)
4. canonical_thumb_x:          d = 6.08  (M: -0.016 ± 0.008 vs N: 0.053 ± 0.008)
5. dist_tip_thumb_mid:         d = 6.05  (M: 0.105 ± 0.015 vs N: 0.222 ± 0.012)
---------------------------------------------------------------------------
Pair 'N' vs 'T':
1. ext_thumb:                  d = 8.21  (N: 0.364 ± 0.007 vs T: 0.448 ± 0.008)
2. canonical_thumb_y:          d = 7.92  (N: 0.817 ± 0.011 vs T: 0.942 ± 0.012)
3. y_span_ratio:               d = 6.64  (N: 0.580 ± 0.010 vs T: 0.672 ± 0.011)
---------------------------------------------------------------------------
Pair 'M' vs 'T':
1. ext_thumb:                  d = 16.54 (M: 0.278 ± 0.009 vs T: 0.448 ± 0.008)
2. canonical_thumb_y:          d = 14.82 (M: 0.702 ± 0.013 vs T: 0.942 ± 0.012)
3. dist_tip_thumb_mid:         d = 14.78 (M: 0.105 ± 0.015 vs T: 0.320 ± 0.011)
===========================================================================
```

### Forensic Takeaway
The separation between M and N ($d pprox 6.1$–$7.6$) and between N and T ($d pprox 6.6$–$8.2$) is an order of magnitude smaller than well-separated pairs (e.g., A vs E: $d=92.6$, L vs D: $d=92.1$, R vs U: $d=1000.0$). This mathematical compression directly explains the observed off-diagonal misclassifications.

---

## 3. Information Loss Mechanisms

1. **Occlusion of Landmark 4 (Thumb Tip):**  
   In all three signs, the distal segment of the thumb is occluded from direct camera view by the overlapping flexed fingers. MediaPipe estimates landmark 4 based on hand silhouette and anatomical prior. Slight noise in landmark 4 shifts the calculated distance metrics across the narrow boundary between M, N, and T.
2. **Finger Curl Invariance:**  
   In all three signs, all four fingers are fully curled into the palm (`curl_index`, `curl_middle`, `curl_ring`, `curl_pinky` all $pprox 0.15$). Thus, 20 of the 42 features provide zero discriminative signal.
3. **Absence of Interdigital Knuckle Cleft Metrics:**  
   The current 42-feature extractor measures thumb-to-tip distances (`dist_tip_thumb_idx`, `dist_tip_thumb_mid`, `dist_tip_thumb_pinky`). However, because the fingertips are all curled against the palm, their relative tip positions are clustered. The true anatomical discriminator is the lateral projection of the thumb tip along the **metacarpophalangeal (MCP) knuckle arch** (landmarks 5, 9, 13, 17):
   - In T: Thumb emerges between MCP 5 and MCP 9.
   - In N: Thumb emerges between MCP 9 and MCP 13.
   - In M: Thumb emerges between MCP 13 and MCP 17.

---

## 4. Remediation and Candidate Feature Protocol

1. **Candidate Metric Formulation (Development Testing):**
   - Relative knuckle projection:
     $$\Delta X_{	ext{MCP}} = rac{(x_4 - x_5) \cdot (x_{17} - x_5)}{\|x_{17} - x_5\|^2}$$
     This normalized scalar measures where the thumb tip projects along the lateral MCP arch from index MCP (0.0) to pinky MCP (1.0).
2. **Regression Constraint:**
   - Production features cannot be modified without proving zero degradation across the remaining 23 alphabet classes on the 123-test regression suite.
