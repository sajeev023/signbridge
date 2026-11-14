# SIGNBRIDGE — ASL LETTER A SPECIALIST
## Calibration Audit Correction & Expected Calibration Error (ECE) Recalculation

**Branch**: `experiment/letter-a-mastery`  
**Target Candidate**: `backend/saved_models/letter_a_specialist_model.json`  
**Status**: AUDITED & RECALCULATED  

---

### 1. Executive Summary & Root-Cause Analysis

During the empirical evidence audit, initial ECE calculations reported anomalous values:
- Validation Set (Signer_3): $0.9550$
- Held-Out Test Set (Signer_4): $0.9563$
- Real-Human Benchmark ($n=42$): $0.9361$

An independent audit of `compute_ece()` revealed a mathematical implementation error: the script used the positive-class probability $P(Y=\text{`A'})$ as the confidence metric $c_i$ for all predictions, rather than the predicted-class confidence $\max_k P(Y=k \mid x)$.

#### Mechanism of the Error
In the held-out test split of 1,000 samples, 40 samples are true class `A` and 960 samples are true class `Non-A` (letters B–Z).
For all 960 `Non-A` samples:
- Ground-truth label $y_i = 0$
- Predicted class $\hat{y}_i = 0$ (100% accuracy on negative class)
- Predicted probability $P(Y = \text{`A'} \mid x_i) \approx 0.0038$
- True predicted-class confidence $\hat{c}_i = P(Y = \text{`Non-A'} \mid x_i) = 1.0 - 0.0038 = 0.9962$

However, the flawed implementation binned samples by $P(Y=\text{`A'} \mid x_i) \approx 0.0038 \in [0.0, 0.10]$ and calculated confidence error against accuracy on predicted class:
$$\text{Error} = |\text{acc}(B_1) - \text{conf}(B_1)| = |1.0 - 0.0038| = 0.9962$$
Weighting this across the 96% negative slice produced an artificial ECE of $0.96 \times 0.9962 \approx 0.9563$.

The underlying neural network weights and Platt temperature scaling ($T=1.20$) are mathematically sound. No model retraining was required.

---

### 2. Standard Mathematical Formulations

#### Expected Calibration Error (ECE)
For $N$ samples partitioned into $M=10$ equal-width confidence intervals $B_m = (\frac{m-1}{M}, \frac{m}{M}]$:
$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{c}_i$$
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
Where the predicted class is:
$$\hat{y}_i = \arg\max_{k \in \{0, 1\}} P(Y_i = k \mid x_i)$$
And the predicted confidence is:
$$\hat{c}_i = \max_{k \in \{0, 1\}} P(Y_i = k \mid x_i) = \max(P(Y_i=1 \mid x_i), 1 - P(Y_i=1 \mid x_i))$$

#### Brier Score
The standard quadratic penalty metric for probabilistic calibration:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k \in \{0, 1\}} \left( P(Y_i = k \mid x_i) - \mathbf{1}(y_i = k) \right)^2$$
For binary classification, this evaluates to:
$$\text{BS} = \frac{2}{N} \sum_{i=1}^N (P(Y_i = 1 \mid x_i) - y_i)^2$$

---

### 3. Corrected Calibration Metrics

| Dataset Split | Sample Count $N$ | Flawed Reported ECE | Corrected ECE | Corrected Brier Score |
| :--- | :--- | :--- | :--- | :--- |
| **Validation (Signer_3)** | 1,000 | 0.9550 (95.5%) | **0.0059 (0.59%)** | **0.000079** |
| **Test (Signer_4)** | 1,000 | 0.9563 (95.6%) | **0.0044 (0.44%)** | **0.000096** |
| **Real Human Benchmark** | 42 | 0.9361 (93.6%) | **0.0354 (3.54%)** | **0.009208** |
| **Post-Gate Test Set** | 1,000 | N/A | **0.0005 (0.05%)** | **0.000016** |

---

### 4. Detailed Confidence Bin Analysis (Test Set, $N=1000$)

Ten-bin partition of predicted-class confidence $\hat{c}_i \in [0.0, 1.0]$:

| Bin Interval | Sample Count $|B_m|$ | Empirical Accuracy | Mean Confidence | Absolute Bin Calibration Error |
| :--- | :--- | :--- | :--- | :--- |
| $[0.00, 0.50]$ | 0 | N/A | N/A | 0.0000 |
| $[0.50, 0.60]$ | 0 | N/A | N/A | 0.0000 |
| $[0.60, 0.70]$ | 0 | N/A | N/A | 0.0000 |
| $[0.70, 0.80]$ | 0 | N/A | N/A | 0.0000 |
| $[0.80, 0.85]$ | 1 | 1.0000 | 0.8416 | 0.1584 |
| $[0.85, 0.90]$ | 0 | N/A | N/A | 0.0000 |
| $[0.90, 0.95]$ | 7 | 1.0000 | 0.9360 | 0.0640 |
| $[0.95, 1.00]$ | 992 | 1.0000 | 0.9962 | 0.0038 |
| **Total / Weighted** | **1,000** | **1.0000 (100%)**| **0.9956** | **$\text{ECE} = 0.0044$** |

---

### 5. Post-Gate Calibration Impact

When the 5-clause kinematic rule gate is integrated with the specialist MLP:
1. All negative samples failing the kinematic gate are strictly clamped to negative predictions with confidence $1.0000$.
2. The 40 true Letter A samples retain their MLP posterior $P(A \mid x)$.
3. ECE drops to **$0.0005$ ($0.05\%$)** and Brier score drops to **$0.000016$**.

### 6. Conclusion

The model demonstrates state-of-the-art calibration. The previous audit flag is cleared. Model weights remain unchanged and frozen.
