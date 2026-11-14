# SIGNBRIDGE — OPERATION ALPHABET MASTERY
## Full 26-Letter Confusion Matrix and Forensic Pair Analysis

**Evaluation Standard:** Independent confusion analysis across held-out synthetic test set (Signer_4, N=1,000) and held-out real-human benchmark (Signer_3-6, N=27).

---

## 1. Synthetic 25-Class Static Confusion Matrix (Signer_4, N=1,000)

Evaluated on 40 samples per class across all 25 static classes:

```
True \ Pred   NEU  A   B   C   D   E   F   G   H   I   K   L   M   N   O   P   Q   R   S   T   U   V   W   X   Y
NEUTRAL        39  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
A               0 40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
B               0  0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
C               0  0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
D               0  0   0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
E               0  0   0   0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
F               0  0   0   0   0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
G               0  0   0   0   0   0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
H               0  0   0   0   0   0   0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
I               0  0   0   0   0   0   0   0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
K               0  0   0   0   0   0   0   0   0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0   0
L               0  0   0   0   0   0   0   0   0   0   0  40   0   0   0   0   0   0   0   0   0   0   0   0   0
M               0  0   0   0   0   0   0   0   0   0   0   0  38   2   0   0   0   0   0   0   0   0   0   0   0
N               0  0   0   0   0   0   0   0   0   0   0   0   5  35   0   0   0   0   0   0   0   0   0   0   0
O               0  0   0   0   0   0   0   0   0   0   0   0   0   0  40   0   0   0   0   0   0   0   0   0   0
P               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0  40   0   0   0   0   0   0   0   0   0
Q               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40   0   0   0   0   0   0   0   0
R               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40   0   0   0   0   0   0   0
S               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40   0   0   0   0   0   0
T               0  0   0   0   0   0   0   0   0   0   0   0   0   5   0   0   0   0   0  35   0   0   0   0   0
U               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40   0   0   0   0
V               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40   0   0   0
W               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40   0   0
X               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40   0
Y               0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40
```

### Non-Zero Off-Diagonal Errors (Synthetic Test):
1. **True M -> Pred N:** 2 samples (thumb tuck depth ambiguity)
2. **True N -> Pred M:** 5 samples (thumb tuck depth ambiguity)
3. **True T -> Pred N:** 5 samples (thumb placement under single vs dual digits)
4. **True NEUTRAL -> Pred C:** 1 sample (resting claw pose proximity)

---

## 2. Real-Human Held-Out Benchmark Confusion Analysis (N=27)

On physical held-out recordings across Signers 3, 4, 5, and 6:
```
True \ Pred     A    B    C    D    L    O    R    V    Y
A                2    0    0    0    0    0    0    0    0
B                0    1    0    0    0    0    0    0    0
C                0    0    8    0    0    0    0    0    2   <-- 2 misclassified as Y
D                0    0    0    1    0    0    0    0    0
L                0    0    0    0    2    0    0    0    0
O                0    0    0    0    0    0    0    0    2   <-- 2 misclassified as Y
R                0    0    0    0    0    0    1    0    0
V                0    0    0    0    0    0    0    2    0
Y                0    0    0    0    0    0    0    0    6
```

### Forensic Root Cause of Real-Human C/O -> Y Confusion
- In genuine webcam video, participants signing C and O often present hands at a 30°–45° oblique angle to the optical axis.
- When viewed obliquely, the curved open C-arc and circular O-loop suffer perspective compression: the thumb and pinky tips appear maximally separated laterally along the image plane, mimicking the wide lateral span of letter Y (`y_span_ratio` and `dist_tip_thumb_pinky`).
- This failure is strictly documented as an unmitigated real-human vulnerability.

---

## 3. Fisher Separation Ranking for Confusing Pairs

| Pair | Primary Distinguishing Feature | Feature Description | Separation Ratio ($d$) |
| :--- | :--- | :--- | :---: |
| **R vs U** | `cross_product_sign_idx_mid` | Cross product sign (crossing vs parallel) | 1,000.0 |
| **U vs V** | `cross_product_sign_idx_mid` | Cross product sign (parallel vs spread) | 1,000.0 |
| **G vs H** | `cross_product_sign_idx_mid` | Pronation angle of lateral fingers | 1,000.0 |
| **A vs E** | `canonical_thumb_y` | Thumb vertical elevation (upright vs tucked) | 92.6 |
| **L vs D** | `canonical_thumb_x` | Lateral thumb abduction (90° vs loop) | 92.1 |
| **D vs F** | `ext_middle` | Middle finger extension (curled vs upright) | 94.2 |
| **A vs S** | `canonical_thumb_y` | Thumb position (lateral upright vs crossing) | 53.9 |
| **M vs T** | `ext_thumb` | Thumb tuck depth under fingers | 16.5 |
| **M vs N** | `ext_thumb` | Thumb tuck depth under 3 vs 2 fingers | 7.7 |
| **N vs T** | `ext_thumb` | Thumb tuck depth under 2 vs 1 finger | 8.2 |
