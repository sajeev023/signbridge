# LETTER A CALIBRATION INDEPENDENT REVIEW
**System:** SIGNBRIDGE ASL Recognition Pipeline  
**Branch:** `experiment/complete-physical-alphabet-coverage`  
**Engineer:** Senior QA & Computer Vision Validation Engineer  
**Date:** 2026-11-15 08:54:07 UTC  
**Focus:** Mathematical Verification of Temperature Scaling & Boundary Sensitivity  

---

## 1. Problem Statement: Specialist False Rejections

In physical-human testing, all 3 genuine Letter A captures failed the specialist model gate ($P(A) \ge 0.70$) despite 98.41% static MLP confidence and passing the binary kinematic curl gate (`is_valid_letter_a = True`). The specialist model output was $0.698974$ to $0.698976$, falling short of $0.70$ by $0.10\%$.

---

## 2. Mathematical Derivation of Temperature Scaling

The specialist model computes softmax probabilities from logits $\mathbf{z} = [z_0, z_1]$ using temperature parameter $T$:

$$P(y = 1 \mid \mathbf{x}) = rac{\exp(z_1 / T)}{\exp(z_0 / T) + \exp(z_1 / T)} = rac{1}{1 + \exp\left(-rac{\Delta z}{T}ight)}$$

where $\Delta z = z_1 - z_0$ is the logit difference.

### Partial Derivative with Respect to Temperature
$$rac{\partial P}{\partial T} = rac{\exp(-\Delta z / T)}{\left(1 + \exp(-\Delta z / T)ight)^2} \cdot \left(rac{\Delta z}{T^2}ight)$$

This derivative reveals two critical mathematical regimes:
1. **For True Positives ($\Delta z > 0$):**
   $rac{\partial P}{\partial T} > 0$. Lowering $T$ from $1.0621$ to $1.0000$ **increases** $P(A)$, moving true human captures from $0.69897$ up to $0.70986 > 0.70$.
2. **For Negatives ($\Delta z < 0$):**
   $rac{\partial P}{\partial T} < 0$. Lowering $T$ from $1.0621$ to $1.0000$ **decreases** $P(A)$, pushing negative samples further away from the acceptance threshold!

---

## 3. Empirical Verification Across All 3,840 Negative Samples

To test the hypothesis that lowering $T$ might increase false positives in deployment, all 3,840 negative samples in `alphabet_dataset_multisigner.json` were evaluated:

```
+-------------------+----------------+--------------------+--------------------+-----------------------+
| Partition         | Negative Count | Max Neg P(T=1.0621)| Max Neg P(T=1.0000)| Worst-Case Class      |
+-------------------+----------------+--------------------+--------------------+-----------------------+
| Training Split    | 1920 samples   | 0.125942           | 0.113285           | Class T (fist/thumb)  |
| Validation Split  | 960 samples    | 0.082156           | 0.071541           | Class T (fist/thumb)  |
| Held-Out Test     | 960 samples    | 0.158383           | 0.145044           | Class T (fist/thumb)  |
+-------------------+----------------+--------------------+--------------------+-----------------------+
```

### Key Findings
1. Lowering $T$ to $1.0000$ reduces the maximum negative probability from $0.1584$ down to $0.1450$.
2. The nearest morphological competitor is **Class T** (fist with thumb tucked between index and middle fingers).
3. The safety margin against the $0.70$ threshold is **55.5%** ($0.7000 - 0.1450 = 0.5550$).

---

## 4. Production Deployment & Frozen Weights Policy

As required by project charter:
- `backend/saved_models/letter_a_specialist_model.json` remains **strictly frozen** with SHA-256 `dc7cd03a...`.
- Zero false positives on 960 validation samples does not guarantee zero in production; hence, the two-stage defense (Binary Kinematic Gate + Specialist Model) remains mandatory.
- Recommending adoption of $T=1.0000$ upon formal model unfreeze.
