# Operation A: Statistical Uncertainty & Sample Size Power Analysis

**Audit Target:** Sample size limitations, binomial confidence intervals, and power analysis for ASL Letter A  
**Auditor:** Byte (Senior Offensive-Security & ML Verification Engineer)  
**Branch:** `experiment/letter-a-mastery`  
**Execution Timestamp:** 2026-11-14T17:14:27  

---

## 1. Executive Summary & Core Statistical Principle

> [!CAUTION]
> **100% observed accuracy on a small sample is NOT proof of 100% population accuracy.**  
> In empirical machine learning, evaluating a small benchmark ($n=3$) yields wide confidence intervals. Reporting a point estimate of $100\%$ without its statistical bounds misleads downstream deployment decisions.

This audit computes exact **Clopper-Pearson** and **Wilson Score** 95% confidence intervals across all evaluation splits and establishes a sample-size power analysis for production-grade validation.

---

## 2. Exact Binomial Confidence Intervals (95% CI)

| Metric | Cohort | Observed Successes ($k$) | Sample Size ($n$) | Point Estimate | Wilson Score 95% CI | Clopper-Pearson Exact 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Letter A Recall** | **Real Human Benchmark** | **3** | **3** | **100.0%** | **[43.85%, 100.00%]** | **[29.24%, 100.00%]** |
| **Non-A Specificity** | **Real Human Benchmark** | **39** | **39** | **100.0%** | **[91.03%, 100.00%]** | **[90.97%, 100.00%]** |
| **Letter A Recall** | Held-Out Test (Signer_4) | 40 | 40 | 100.0% | [91.19%, 100.00%] | [91.19%, 100.00%] |
| **Non-A Specificity** | Held-Out Test (Signer_4) | 960 | 960 | 100.0% | [99.62%, 100.00%] | [99.62%, 100.00%] |

---

## 3. Dissecting the $n=3$ Real-Human Limitation

While all 3 empirical Letter A captures in `real_human_benchmark.json` were recognized successfully:
- The **Clopper-Pearson 95% lower bound is 29.24%**.
- This means that an underlying recognizer with a true population recall of only **30%** has a non-negligible probability ($0.30^3 \approx 2.7\%$, and at $50\%$ true recall, $0.50^3 = 12.5\%$) of passing all 3 trials purely by chance.
- Conversely, for **Non-A Specificity** ($n=39$), observing 39/39 rejections guarantees with 95% confidence that the true population specificity is at least **90.97%**.

---

## 4. Power Analysis: Required Sample Sizes for Production Sign-Off

To mathematically guarantee high population reliability before deployment to production users, we calculate the minimum sample size $n$ required when observing zero errors ($k=n$).

For an exact Clopper-Pearson 95% lower bound $p_{\text{lower}}$ at significance level $\alpha = 0.05$:
$$p_{\text{lower}} = \alpha^{1/n} \implies \ln(p_{\text{lower}}) = \frac{\ln(\alpha)}{n} \implies n \ge \frac{\ln(0.05)}{\ln(p_{\text{lower}})}$$

| Target Production Standard | Desired 95% CI Lower Bound | Required Real Human Captures ($n$) | Current Benchmark ($n=3$) Deficit |
| :--- | :---: | :---: | :---: |
| **Preliminary Signer Generalization** | $\ge 80.0\%$ | **$n \ge 14$ samples** | Deficit of 11 samples |
| **Production Grade Validation** | $\ge 90.0\%$ | **$n \ge 29$ samples** | Deficit of 26 samples |
| **High-Assurance Production Standard** | $\ge 95.0\%$ | **$n \ge 59$ samples** | Deficit of 56 samples |
| **Ultra-High Reliability Standard** | $\ge 98.0\%$ | **$n \ge 149$ samples** | Deficit of 146 samples |

### Key Takeaway:
To claim with 95% statistical confidence that ASL Letter A has a real-world human recall $\ge 90.0\%$, the project must evaluate at least **29 independent, non-synthetic human captures** with zero false negatives.

---

## 5. Audit Conclusion

The current $n=3$ empirical benchmark provides **preliminary proof of concept**, not statistically conclusive validation. The project must conduct a formal human validation trial ($n \ge 30$) before promoting Letter A to `main`.
