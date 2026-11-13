# SIGNBRIDGE TITAN: ASL MANUAL ALPHABET CLASS DEFINITIONS
## Mathematical & Biomechanical Specification of the 26 ASL Manual Letters

**Document ID:** `SB-TITAN-ASL-CLASS-01`  
**System Class:** Biomechanical Landmark Specification  
**Active Research Branch:** `experiment/signbridge-titan`  
**Reference Ground Truth:** Gérard Aflague Collection Poster  
**Date:** November 13, 2026  
**Status:** FORMALIZED ACROSS 26 MANUAL CLASSES + NEUTRAL  

---

## 1. 27-CLASS CANONICAL VOCABULARY

The SIGNBRIDGE TITAN ASL recognition vocabulary consists of **27 discrete indices**:
- **Static Alphabet Classes (24):** A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
- **Dynamic Trajectory Classes (2):** J, Z
- **Background / Idle State (1):** NEUTRAL

```
Index Mapping:
 0: A   1: B   2: C   3: D   4: E   5: F   6: G   7: H   8: I
 9: J  10: K  11: L  12: M  13: N  14: NEUTRAL  15: O  16: P
17: Q  18: R  19: S  20: T  21: U  22: V  23: W  24: X  25: Y  26: Z
```

---

## 2. MATHEMATICAL BIOMECHANICAL EQUATIONS

All raw optical landmark coordinates $L_i = (x_i, y_i, z_i) \in \mathbb{R}^3$ for $i \in \{0, \dots, 20\}$ detected by MediaPipe HandLandmarker undergo rigorous spatial transformation before feature classification.

### 2.1 Coordinate Frame Normalization
1. **Wrist Translation Invariance:**
   $$\tilde{L}_i = L_i - L_0, \quad \forall i \in \{0, \dots, 20\}$$
2. **Palm Scale Invariance:**
   Palm baseline distance $s_{\text{palm}}$ is measured between wrist ($L_0$) and middle MCP ($L_9$):
   $$s_{\text{palm}} = \|\tilde{L}_9\|_2 + \epsilon, \quad \epsilon = 10^{-6}$$
   Scale-normalized coordinate tensor:
   $$\hat{L}_i = \frac{\tilde{L}_i}{s_{\text{palm}}}$$

### 2.2 Kinematic Parent Displacements
Each node encodes its spatial offset from its anatomical predecessor in the kinematic skeleton:
$$\Delta \hat{L}_i = \hat{L}_i - \hat{L}_{\text{parent}(i)}$$
where $\text{parent}(i)$ is defined by the kinematic tree root at wrist Landmark 0.

### 2.3 Phalanx Joint Flexion (Curl Metric)
Finger curl $C_f$ for finger $f \in \{\text{index}, \text{middle}, \text{ring}, \text{pinky}\}$ evaluates the Euclidean distance between fingertip and MCP normalized by the total anatomical bone length:
$$C_f = \frac{\|\hat{L}_{\text{tip}} - \hat{L}_{\text{MCP}}\|_2}{\|\hat{L}_{\text{PIP}} - \hat{L}_{\text{MCP}}\|_2 + \|\hat{L}_{\text{DIP}} - \hat{L}_{\text{PIP}}\|_2 + \|\hat{L}_{\text{tip}} - \hat{L}_{\text{DIP}}\|_2}$$
- **Full Extension:** $C_f \ge 0.82$
- **Curved Arch (C/O):** $0.45 \le C_f < 0.82$
- **Clenched Fist:** $C_f < 0.45$

### 2.4 Inter-Digital Abduction & Spread
Separation between adjacent fingertips $f_1, f_2$:
$$D_{f_1, f_2} = \|\hat{L}_{\text{tip}, f_1} - \hat{L}_{\text{tip}, f_2}\|_2$$
- **U-Sign (Adducted):** $D_{8, 12} < 0.25$
- **V-Sign (Abducted):** $D_{8, 12} \ge 0.40$
- **W-Sign (Spread):** $D_{8, 12} \ge 0.35 \land D_{12, 16} \ge 0.35$

### 2.5 Thumb Lateral Position & Crossing (A vs S vs E vs M/N/T)
1. **Thumb Lateral Margin:**
   $$X_{\text{thumb}} = \hat{L}_{4, x} - \hat{L}_{5, x}$$
   - **A-Sign:** $X_{\text{thumb}} < -0.15$ (thumb rests strictly outside index MCP).
   - **S-Sign:** $X_{\text{thumb}} \ge 0.05$ (thumb crosses anterior fist over index and middle).
2. **Thumb Vertical Crease Elevation:**
   $$Y_{\text{thumb}} = \hat{L}_{4, y} - \hat{L}_{5, y}$$
   - **A-Sign:** $Y_{\text{thumb}} \le -0.10$ (thumb tip raised near index PIP).
   - **E-Sign:** $Y_{\text{thumb}} \ge 0.15$ (thumb tucked below bent fingertips).
3. **Finger-Tuck Emergence Ratio:**
   Thumb tip proximity to ring MCP ($L_{13}$) and pinky MCP ($L_{17}$) separates the thumb-under triplet:
   - **M-Sign:** Protrudes between ring and pinky ($L_4 \to L_{17}$).
   - **N-Sign:** Protrudes between middle and ring ($L_4 \to L_{13}$).
   - **T-Sign:** Protrudes between index and middle ($L_4 \to L_9$).

### 2.6 Inter-Finger Crossing Metric (R-Sign)
Evaluates middle finger crossing over the dorsal plane of the index finger:
$$X_{\text{cross}} = \hat{L}_{12, x} - \hat{L}_{8, x}$$
For a right hand viewed from the camera, middle finger crossing index yields $X_{\text{cross}} > 0.10$ with depth disparity $\hat{L}_{12, z} < \hat{L}_{8, z}$.

### 2.7 Inversion Pitch (P and Q Signs)
Evaluates pitch angle of the primary pointing finger vector relative to the vertical axis:
$$\Pi = \hat{L}_{\text{tip}, y} - \hat{L}_{\text{MCP}, y}$$
- **Upright (K, G):** $\Pi \le -0.30$ (pointing upwards or horizontal).
- **Downward Inverted (P, Q):** $\Pi \ge 0.40$ (pointing toward floor).

---

## 3. DYNAMIC TRAJECTORY SPECIFICATIONS: J AND Z

### 3.1 Sign J (Pinky Curved Hook)
- **Static Pre-Condition:** Hand initiates from an `I`-handshape: pinky extended ($C_{\text{pinky}} \ge 0.80$), index/middle/ring clenched ($C_f \le 0.40$).
- **Temporal Trajectory:** Pinky fingertip $L_{20}(t)$ over $t \in [0, T]$ ($T \ge 250\ \text{ms}$, 16 frames) satisfies:
  1. Initial downward displacement:
     $$\Delta y_{\text{disp}} = y(t_{\text{mid}}) - y(0) > 0.05$$
  2. Terminal inward hook:
     $$\Delta x_{\text{hook}} = x(T) - x(t_{\text{mid}}) < -0.04$$
  3. Non-zero path curvature:
     $$\kappa = \frac{\sum_{t=1}^{T-1} \|\vec{v}_t \times \vec{v}_{t+1}\|}{\sum_{t=1}^{T} \|\vec{v}_t\|} > 0.25$$

### 3.2 Sign Z (Index 3-Stroke Zigzag)
- **Static Pre-Condition:** Hand initiates from index-extended stylus handshape: index extended ($C_{\text{index}} \ge 0.85$), thumb folded over curled middle finger.
- **Temporal Trajectory:** Index fingertip $L_8(t)$ over $t \in [0, T]$ ($T \ge 300\ \text{ms}$, 16 frames) satisfies a 3-phase piecewise motion:
  1. **Stroke 1 (Horizontal Right):** $v_x > 0 \land |v_y| \approx 0$
  2. **Stroke 2 (Diagonal Down-Left):** $v_x < 0 \land v_y > 0$
  3. **Stroke 3 (Horizontal Right):** $v_x > 0 \land |v_y| \approx 0$
  4. **Path Complexity:** Total path length $S = \sum \|\Delta L_8(t)\|_2$ exceeds Euclidean net displacement $D_{\text{net}} = \|L_8(T) - L_8(0)\|_2$ by ratio:
     $$\frac{S}{D_{\text{net}}} \ge 1.70$$
