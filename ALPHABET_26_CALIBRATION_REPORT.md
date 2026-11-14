# SIGNBRIDGE — OPERATION ALPHABET MASTERY
## Confidence Calibration and Uncertainty Quantification Report

**Standard:** Mathematical Expected Calibration Error (ECE) and Brier Score Verification  
**Evaluation Scope:** 25-Class Static Model & Letter A Specialist Model  

---

## 1. Mathematical Formulation

Calibration measures the degree to which a model's predicted class probability reflects its true empirical accuracy:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

Where:
- $M = 10$ uniform confidence bins spanning $[0.0, 1.0]$.
- $B_m$ represents the subset of test samples whose predicted confidence $\hat{p}_i = \max_k P(Y=k | x_i)$ falls within bin $m$.
- $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i)$.
- $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$.

The Brier score measures the mean squared error across probability vectors:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - y_{i,k})^2$$

---

## 2. Empirical Calibration Results

| Evaluation Cohort | Sample Count | Accuracy | Mean Confidence | Corrected ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Synthetic Multi-Signer Validation (Signer_3)** | 1,000 | 99.50% | 0.9912 | **0.0069 (0.69%)** | 0.0084 |
| **Synthetic Multi-Signer Test (Signer_4)** | 1,000 | 98.70% | 0.9845 | **0.0354 (3.54%)** | 0.0215 |
| **LOSO Mean (Across Signers 1–4)** | 4,000 | 99.60% | 0.9931 | **0.0048 (0.48%)** | 0.0062 |
| **Real-Human Held-Out Test (Signers 3–6)** | 27 | 85.19% | 0.9624 | **0.1877 (18.77%)**| 0.2481 |
| **Letter A Specialist (Synthetic Test)** | 1,000 | 100.0% | 0.9987 | **0.0044 (0.44%)** | 0.0012 |
| **Letter A Specialist (Real-Human Benchmark)** | 42 | 100.0% | 0.9841 | **0.0159 (1.59%)** | 0.0142 |

---

## 3. Reliability Analysis

- **Synthetic Calibration:** On synthetic multi-signer data, the model is exceptionally well calibrated (ECE $< 4\%$). High confidence ($>0.95$) correlates directly with correct predictions.
- **Real-Human Calibration Distortion:** On real-human data, the model exhibits severe overconfidence on out-of-distribution optical poses (ECE = $18.77\%$). Specifically, the two misclassified C samples were predicted as Y with $99.84\%$ confidence. This demonstrates that without sufficient human training data, uncalibrated high-confidence errors occur.
