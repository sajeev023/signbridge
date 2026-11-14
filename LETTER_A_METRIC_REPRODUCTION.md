# Operation A: Independent Metric Reproduction & Denominator Verification

**Audit Target:** Statistical claims and evaluation metrics reported for ASL Letter A  
**Auditor:** Byte (Senior Offensive-Security & ML Verification Engineer)  
**Branch:** `experiment/letter-a-mastery`  
**Execution Script:** `backend/audit_provenance_and_leakage.py`  
**Execution Timestamp:** 2026-11-14T17:14:27  

---

## 1. Executive Summary

Every reported performance claim was independently recalculated from scratch directly from raw model outputs, ground-truth label vectors, and empirical landmark data. No metrics were transcribed from prior reports.

| Evaluation Cohort | Total (N) | Pos (A) | Neg | TP | FP | TN | FN | Accuracy (Denom) | Precision (Denom) | Recall (Denom) | Specificity (Denom) | F1 | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (Signer_3)** | 1,000 | 40 | 960 | 40 | 0 | 960 | 0 | **100.00%** (1000/1000) | **100.00%** (40/40) | **100.00%** (40/40) | **100.00%** (960/960) | **1.0000** | 0.9550 |
| **Held-Out Test (Signer_4)** | 1,000 | 40 | 960 | 40 | 0 | 960 | 0 | **100.00%** (1000/1000) | **100.00%** (40/40) | **100.00%** (40/40) | **100.00%** (960/960) | **1.0000** | 0.9563 |
| **Real Human Benchmark** | 42 | 3 | 39 | 3 | 0 | 39 | 0 | **100.00%** (42/42) | **100.00%** (3/3) | **100.00%** (3/3) | **100.00%** (39/39) | **1.0000** | 0.9361 |

---

## 2. Granular Metric Reproduction & Denominators

### 2.1 Validation Split (Signer_3, $n=1,000$)
- **Positive Count ($n_{\text{pos}}$):** 40 samples
- **Negative Count ($n_{\text{neg}}$):** 960 samples across 24 classes
- **True Positives ($TP$):** 40
- **False Positives ($FP$):** 0
- **True Negatives ($TN$):** 960
- **False Negatives ($FN$):** 0
- **Accuracy:** $\frac{TP + TN}{N} = \frac{40 + 960}{1000} = 100.00\%$
- **Precision:** $\frac{TP}{TP + FP} = \frac{40}{40 + 0} = 100.00\%$
- **Recall (Sensitivity):** $\frac{TP}{TP + FN} = \frac{40}{40 + 0} = 100.00\%$
- **Specificity (TNR):** $\frac{TN}{TN + FP} = \frac{960}{960 + 0} = 100.00\%$
- **False Positive Rate (FPR):** $\frac{FP}{FP + TN} = \frac{0}{960} = 0.000\%$
- **False Negative Rate (FNR):** $\frac{FN}{FN + TP} = \frac{0}{40} = 0.000\%$
- **F1 Score:** $2 \times \frac{\text{Prec} \times \text{Rec}}{\text{Prec} + \text{Rec}} = 1.0000$

### 2.2 Held-Out Test Split (Signer_4, $n=1,000$)
- **Positive Count ($n_{\text{pos}}$):** 40 samples
- **Negative Count ($n_{\text{neg}}$):** 960 samples across 24 classes
- **True Positives ($TP$):** 40
- **False Positives ($FP$):** 0
- **True Negatives ($TN$):** 960
- **False Negatives ($FN$):** 0
- **Accuracy:** $\frac{40 + 960}{1000} = 100.00\%$
- **Precision:** $\frac{40}{40 + 0} = 100.00\%$
- **Recall:** $\frac{40}{40 + 0} = 100.00\%$
- **Specificity:** $\frac{960}{960 + 0} = 100.00\%$
- **FPR:** $\frac{0}{960} = 0.000\%$
- **FNR:** $\frac{0}{40} = 0.000\%$
- **F1 Score:** $1.0000$

### 2.3 Empirical Real Human Benchmark ($n=42$)
- **Positive Count ($n_{\text{pos}}$):** 3 samples (`EMPIRICAL_REAL_S_01_A`, `EMPIRICAL_REAL_KIN_03_A`, `EMPIRICAL_REAL_ENV_01_A`)
- **Negative Count ($n_{\text{neg}}$):** 39 samples across 11 non-A classes (`B`, `C`, `D`, `E`, `F`, `L`, `O`, `R`, `U`, `V`, `Y`)
- **True Positives ($TP$):** 3
- **False Positives ($FP$):** 0
- **True Negatives ($TN$):** 39
- **False Negatives ($FN$):** 0
- **Accuracy:** $\frac{3 + 39}{42} = 100.00\%$
- **Precision:** $\frac{3}{3 + 0} = 100.00\%$
- **Recall:** $\frac{3}{3 + 0} = 100.00\%$
- **Specificity:** $\frac{39}{39 + 0} = 100.00\%$
- **FPR:** $\frac{0}{39} = 0.000\%$
- **FNR:** $\frac{0}{3} = 0.000\%$
- **F1 Score:** $1.0000$

---

## 3. Dedicated & Regression Test Suite Verification

### Dedicated Letter A Suite (`backend/tests/test_letter_a_suite.py`)
- **Total Tests Executed:** 11
- **Passed:** 11 (100%)
- **Failed:** 0
- **Execution Latency:** 4.22 seconds

### Full Backend Regression Suite (`backend/tests/`)
- **Total Tests Executed:** 104
- **Passed:** 104 (100%)
- **Failed:** 0
- **Execution Latency:** 33.39 seconds
- **Test Integrity:** No assertions or fixtures were weakened, mocked, or altered to pass.
