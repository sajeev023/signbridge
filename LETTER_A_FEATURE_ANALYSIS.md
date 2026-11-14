# ASL Manual Alphabet: Letter A Feature Separation & Biomechanical Analysis

**Document Version:** 1.0.0-phase1-isolated-a  
**Feature Space:** SIGNBRIDGE Canonical 42-Feature Continuous Biomechanical Representation  
**Evaluation Script:** `backend/compute_letter_a_separation.py`  
**Output Data:** `backend/letter_a_feature_separation.json`  

---

## 1. Mathematical Feature Representation

The 42 canonical features represent scale-, translation-, and 3D rotation-invariant properties extracted directly from 21 MediaPipe hand landmarks.

### Coordinate Frame Transformation
The local orthonormal coordinate frame is constructed at the wrist $\vec{L}_0$:
1. **Longitudinal Axis ($\vec{u}_y$):** Wrist to middle MCP:
   $$\vec{u}_y = \frac{\vec{L}_9 - \vec{L}_0}{\|\vec{L}_9 - \vec{L}_0\|}, \quad \text{scale} = \max(0.035, \|\vec{L}_9 - \vec{L}_0\|)$$
2. **Normal Axis ($\vec{u}_z$):** Cross product of index MCP and pinky MCP vectors:
   $$\vec{n} = (\vec{L}_5 - \vec{L}_0) \times (\vec{L}_{17} - \vec{L}_0), \quad \vec{u}_z = \frac{\vec{n}}{\|\vec{n}\|}$$
3. **Lateral Axis ($\vec{u}_x$):** Cross product of longitudinal and normal axes:
   $$\vec{u}_x = \vec{u}_y \times \vec{u}_z$$
4. **Canonical Coordinates:** Every landmark $\vec{L}_i$ is projected into local space:
   $$\hat{\vec{L}}_i = \left[ \frac{(\vec{L}_i - \vec{L}_0) \cdot \vec{u}_x}{\text{scale}}, \frac{(\vec{L}_i - \vec{L}_0) \cdot \vec{u}_y}{\text{scale}}, \frac{(\vec{L}_i - \vec{L}_0) \cdot \vec{u}_z}{\text{scale}} \right]^T$$

---

## 2. Quantitative Separation: Letter A vs Hard Negatives

Separation between Class A ($n=160$) and confusable classes ($n=160$ each) was evaluated using:
- **Cohen's $d$ Effect Size:** $d = \frac{|\mu_A - \mu_T|}{\sqrt{(\sigma_A^2 + \sigma_T^2)/2}}$
- **Fisher Discriminant Ratio:** $F = \frac{(\mu_A - \mu_T)^2}{\sigma_A^2 + \sigma_T^2}$

### 2.1 A vs E (Thumb Folding vs Elevation)
In Letter **E**, all four fingertips curl downward with the thumb tucked horizontally across the palm underneath them. In Letter **A**, the thumb rests upright along the radial side.

| Rank | Feature Index | Feature Name | Mean (A) | Std (A) | Mean (E) | Std (E) | Cohen's $d$ | Fisher Ratio |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **25** | `canonical_thumb_y` | **1.119** | 0.048 | **0.357** | 0.031 | **18.86** | **177.90** |
| 2 | **15** | `cos_spread_thumb_idx` | **-0.387** | 0.071 | **+0.626** | 0.059 | **15.52** | **120.46** |
| 3 | **0** | `ext_thumb` | **+0.569** | 0.048 | **-0.046** | 0.034 | **14.78** | **109.26** |
| 4 | **21** | `dist_tip_thumb_pinky` | **0.985** | 0.055 | **0.449** | 0.030 | **12.09** | **73.08** |
| 5 | **39** | `y_span_ratio` | **0.985** | 0.055 | **0.449** | 0.030 | **12.09** | **73.08** |
| 6 | **24** | `canonical_thumb_x` | **+0.391** | 0.037 | **-0.074** | 0.040 | **11.96** | **71.58** |

*Analysis:* `canonical_thumb_y` ($F=177.9$) and `cos_spread_thumb_idx` ($F=120.5$) provide massive separability. Thumb elevation in E is compressed to $0.357$, while A maintains $\ge 0.55$.

### 2.2 A vs S (Thumb Crossing vs Lateral Margin)
In Letter **S**, the thumb wraps across the anterior face of the curled fingers. In Letter **A**, the thumb remains strictly along the radial margin.

| Rank | Feature Index | Feature Name | Mean (A) | Std (A) | Mean (S) | Std (S) | Cohen's $d$ | Fisher Ratio |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **15** | `cos_spread_thumb_idx` | **-0.387** | 0.071 | **+0.654** | 0.049 | **17.14** | **146.85** |
| 2 | **25** | `canonical_thumb_y` | **1.119** | 0.048 | **0.510** | 0.031 | **15.16** | **114.88** |
| 3 | **39** | `y_span_ratio` | **0.985** | 0.055 | **0.440** | 0.033 | **12.02** | **72.27** |
| 4 | **21** | `dist_tip_thumb_pinky` | **0.985** | 0.055 | **0.440** | 0.033 | **12.02** | **72.27** |
| 5 | **24** | `canonical_thumb_x` | **+0.391** | 0.037 | **-0.046** | 0.045 | **10.56** | **55.75** |
| 6 | **0** | `ext_thumb` | **+0.569** | 0.048 | **+0.120** | 0.037 | **10.45** | **54.59** |

*Analysis:* `cos_spread_thumb_idx` ($F=146.85$) is the defining metric. In S, the thumb vector points across index ($+0.654$), whereas in A it points radially outward ($-0.387$).

### 2.3 A vs M, N, T (Tucked Thumb Configurations)
In letters **M**, **N**, and **T**, the thumb tucks underneath the curled fingers:

| Feature Index | Feature Name | Mean (A) | Mean (M) | Mean (N) | Mean (T) | Separator Role |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **15** | `cos_spread_thumb_idx` | **-0.387** | +0.492 | +0.352 | +0.206 | High spread indicates tucked/adducted thumb in M/N/T |
| **20** | `dist_tip_thumb_mid` | **0.718** | **0.118** | **0.151** | **0.351** | Tip touches/tucks near middle finger in M/N/T; distant in A |
| **24** | `canonical_thumb_x` | **+0.391** | +0.021 | +0.058 | +0.141 | Lateral displacement is large in A, tucked inside in M/N/T |
| **26** | `canonical_thumb_z` | **-0.216** | -0.445 | -0.553 | **-0.508** | Protrusion between knuckles in T produces negative depth |

### 2.4 A vs NEUTRAL (Closed Fist vs Relaxed Hand)
In **NEUTRAL**, fingers are relaxed or extended:

| Feature Index | Feature Name | Mean (A) | Mean (NEUTRAL) | Cohen's $d$ | Fisher Ratio | Physical Effect |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **15** | `cos_spread_thumb_idx` | **-0.387** | **+0.779** | 7.55 | 28.48 | Thumb angle vs index |
| **8** | `dist_tip_ring_palm` | **0.491** | **0.910** | 5.24 | 13.71 | Ring finger curled into palm in A, open in NEUTRAL |
| **35** | `curl_angle_idx_pip` | **+0.143** | **-0.687** | 5.18 | 13.44 | Index PIP flexed in A, straight in NEUTRAL |
| **3** | `ext_ring` | **-0.128** | **+0.529** | 5.15 | 13.24 | Ring extension near zero/negative in A |
| **7** | `dist_tip_mid_palm` | **0.515** | **1.028** | 5.12 | 13.11 | Middle fingertip distance to palm |

---

## 3. Biomechanical Rule Gate Definition

Based on the empirical distributions, the calibrated geometric gate enforces:
$$\text{Gate}(x) = (E_{\text{idx}} < 0.35) \land (E_{\text{mid}} < 0.35) \land (E_{\text{ring}} < 0.35) \land (E_{\text{pky}} < 0.35) \land (\hat{L}_{4, y} \ge 0.55) \land (\hat{L}_{4, x} \ge 0.12) \land (\cos_{\text{spread}} < 0.15) \land (\text{dist}_{4, 12} \ge 0.38)$$

This gate yields **100% acceptance** on real and synthetic ASL A, while achieving **0% acceptance** on E, S, M, and N.
