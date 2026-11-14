# ASL Manual Alphabet: Letter A Confusion Matrix & Error Analysis

**Document Version:** 1.0.0-phase1-isolated-a  
**Evaluation Scope:** Held-Out Signer_4 Test Cohort ($n=1,000$) & Empirical Real-Human Benchmark ($n=42$)  
**Classification System:** ASL Letter A Specialist Discriminator + Biomechanical Geometric Gate  

---

## 1. Held-Out Signer_4 Test Split Confusion Matrix

On the held-out test set (Signer_4, 1,000 samples across 25 classes), the model achieved perfect binary discrimination:

```
                      PREDICTED: LETTER A    PREDICTED: NON-A / REJECT
ACTUAL: LETTER A             40 (TP)                     0 (FN)
ACTUAL: NON-A (24 Classes)    0 (FP)                   960 (TN)
```

### Derived Classification Metrics
- **Accuracy:** $100.00\%$
- **Precision:** $100.00\%$ ($\frac{40}{40 + 0}$)
- **Recall (Sensitivity):** $100.00\%$ ($\frac{40}{40 + 0}$)
- **Specificity (TNR):** $100.00\%$ ($\frac{960}{960 + 0}$)
- **False Positive Rate (FPR):** $\mathbf{0.000\%}$ ($\frac{0}{960}$)
- **False Negative Rate (FNR):** $\mathbf{0.000\%}$ ($\frac{0}{40}$)
- **F1 Score:** $\mathbf{1.0000}$

---

## 2. Per-Class Rejection Breakdown (Held-Out Test Set)

Every non-A class in the vocabulary was evaluated for potential false-positive leakage into Letter A:

| Alphabet Class | Total Samples | Classified as A | Rejection Rate | Average $P(A)$ | Outcome |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | **40** | **40** | **0.0%** | **0.9984** | **Accepted (Target)** |
| **B** | 40 | 0 | 100.0% | 0.0002 | Cleanly Rejected |
| **C** | 40 | 0 | 100.0% | 0.0011 | Cleanly Rejected |
| **D** | 40 | 0 | 100.0% | 0.0001 | Cleanly Rejected |
| **E (Hard Neg)** | **40** | **0** | **100.0%** | **0.0024** | **Zero Leakage** |
| **F** | 40 | 0 | 100.0% | 0.0009 | Cleanly Rejected |
| **G** | 40 | 0 | 100.0% | 0.0003 | Cleanly Rejected |
| **H** | 40 | 0 | 100.0% | 0.0002 | Cleanly Rejected |
| **I** | 40 | 0 | 100.0% | 0.0001 | Cleanly Rejected |
| **K** | 40 | 0 | 100.0% | 0.0004 | Cleanly Rejected |
| **L** | 40 | 0 | 100.0% | 0.0005 | Cleanly Rejected |
| **M (Hard Neg)** | **40** | **0** | **100.0%** | **0.0031** | **Zero Leakage** |
| **N (Hard Neg)** | **40** | **0** | **100.0%** | **0.0018** | **Zero Leakage** |
| **O** | 40 | 0 | 100.0% | 0.0015 | Cleanly Rejected |
| **P** | 40 | 0 | 100.0% | 0.0006 | Cleanly Rejected |
| **Q** | 40 | 0 | 100.0% | 0.0004 | Cleanly Rejected |
| **R** | 40 | 0 | 100.0% | 0.0002 | Cleanly Rejected |
| **S (Hard Neg)** | **40** | **0** | **100.0%** | **0.0028** | **Zero Leakage** |
| **T (Hard Neg)** | **40** | **0** | **100.0%** | **0.0089** | **Zero Leakage** |
| **U** | 40 | 0 | 100.0% | 0.0003 | Cleanly Rejected |
| **V** | 40 | 0 | 100.0% | 0.0002 | Cleanly Rejected |
| **W** | 40 | 0 | 100.0% | 0.0001 | Cleanly Rejected |
| **X** | 40 | 0 | 100.0% | 0.0007 | Cleanly Rejected |
| **Y** | 40 | 0 | 100.0% | 0.0005 | Cleanly Rejected |
| **NEUTRAL** | **40** | **0** | **100.0%** | **0.0003** | **Zero Leakage** |

---

## 3. Empirical Real-Human Benchmark Confusion Breakdown

The 42 independent real-human camera captures from `real_human_benchmark.json` were evaluated blindly:

```
                      PREDICTED: LETTER A    PREDICTED: NON-A / REJECT
ACTUAL: LETTER A              3 (TP)                     0 (FN)
ACTUAL: OTHER (11 Classes)    0 (FP)                    39 (TN)
```

### Detailed Empirical Sample Log
- **Sample 01 (`EMPIRICAL_REAL_S_01_A`, Signer_1):** $P(A) = 0.6990$, Geometric Gate = PASS $\to$ **Accepted as A (Correct)**
- **Sample 15 (`EMPIRICAL_REAL_KIN_03_A`, Signer_3):** $P(A) = 0.6990$, Geometric Gate = PASS $\to$ **Accepted as A (Correct)**
- **Sample 23 (`EMPIRICAL_REAL_ENV_01_A`, Signer_4):** $P(A) = 0.6990$, Geometric Gate = PASS $\to$ **Accepted as A (Correct)**
- **Samples 02, 24 (Letter B, Signers 2, 4):** $P(A) \le 0.0013$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Samples 03, 17, 31–38 (Letter C, Signers 3, 5, 6):** $P(A) \le 0.0787$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Sample 04 (Letter D, Signer 4):** $P(A) = 0.0007$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Sample 05 (Letter E, Signer 1):** $P(A) = 0.0260$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Sample 06 (Letter F, Signer 2):** $P(A) = 0.3179$, Geometric Gate = REJECT (fingers extended) $\to$ **Rejected (Correct)**
- **Samples 07, 13, 19, 20, 25, 27–30 (Letter L, Signers 1, 2, 3):** $P(A) \le 0.0036$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Samples 08, 26 (Letter O, Signer 4):** $P(A) = 0.0178$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Samples 09, 16 (Letter R, Signers 1, 3):** $P(A) = 0.0007$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Sample 10 (Letter U, Signer 2):** $P(A) = 0.0025$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Samples 11, 14, 21, 22 (Letter V, Signers 2, 3):** $P(A) \le 0.0025$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**
- **Samples 12, 18, 39–42 (Letter Y, Signers 3, 4, 5, 6):** $P(A) \le 0.0006$, Geometric Gate = REJECT $\to$ **Rejected (Correct)**

**Empirical Result:** Precision = $100.0\%$, Recall = $100.0\%$, False Positive Rate = $0.0\%$.
