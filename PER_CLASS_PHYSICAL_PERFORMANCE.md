# PER-CLASS PHYSICAL PERFORMANCE REPORT
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Dataset:** Curated Physical Human Benchmark (42 Samples, 6 Signers)  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** {NOW_UTC}  

---

## 1. Physical Human Performance Scorecard (12 Validated Classes)

Performance computed across genuine physical human captures without threshold tuning:

| Letter | Total Human Captures | True Positives | False Positives | False Negatives | Precision | Recall | F1 Score | Accuracy | Dominant Error Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | 3 | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 100% False Rejection (Kinematic gate) |
| **B** | 2 | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.0% | Perfect recognition |
| **C** | 10 | 8 | 0 | 2 | 1.0000 | 0.8000 | 0.8889 | 80.0% | Confusion with 'Y' (thumb-pinky flare) |
| **D** | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.0% | Perfect recognition |
| **E** | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.0% | Perfect recognition |
| **F** | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.0% | Perfect recognition |
| **L** | 9 | 9 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.0% | Perfect recognition |
| **O** | 2 | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0% | False Rejection (Conf < 0.70) |
| **R** | 2 | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.0% | Perfect recognition |
| **U** | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.0% | Perfect recognition |
| **V** | 4 | 4 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.0% | Perfect recognition |
| **Y** | 6 | 6 | 2 | 0 | 0.7500 | 1.0000 | 0.8571 | 100.0% | 2 False Positives from 'C' |
| **TOTAL**| **42** | **35** | **2** | **7** | **0.8125**| **0.8167**| **0.8122**| **83.33%**| Macro F1: 0.8122 |

---

## 2. Missing Physical Classes (14 Classes Pending Collection)

| Letter | Modality | Target Active Model | Expected Hard Negatives | Physical Status |
| :--- | :--- | :--- | :--- | :--- |
| **G** | Static | Multi-Signer Static MLP | `H`, `Q`, `P` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **H** | Static | Multi-Signer Static MLP | `G`, `U`, `N` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **I** | Static | Multi-Signer Static MLP | `J`, `Y`, `D` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **J** | **Dynamic** | Kinivi Point History Trajectory | `I`, `Z` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **K** | Static | Multi-Signer Static MLP | `V`, `P`, `D` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **M** | Static | Multi-Signer Static MLP | `N`, `T`, `S`, `A` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **N** | Static | Multi-Signer Static MLP | `M`, `T`, `S`, `A` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **P** | Static | Multi-Signer Static MLP | `K`, `Q`, `D` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **Q** | Static | Multi-Signer Static MLP | `G`, `P`, `O` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **S** | Static | Multi-Signer Static MLP | `A`, `E`, `T`, `M`, `N` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **T** | Static | Multi-Signer Static MLP | `A`, `S`, `N`, `M` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **W** | Static | Multi-Signer Static MLP | `B`, `V`, `F` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **X** | Static | Multi-Signer Static MLP | `D`, `S`, `E` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
| **Z** | **Dynamic** | Kinivi Point History Trajectory | `D`, `J` | **ZERO SAMPLES (PENDING PHYSICAL COLLECTION)** |
