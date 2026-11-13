"""
SIGNBRIDGE Confusing Pairs Forensic Diagnostic Script
Compares the canonical 42-feature vectors between commonly confused ASL alphabet pairs:
1. C vs Y (Thumb/Pinky spread and optical foreshortening)
2. A vs E (Fist vs Clenched with thumb tucked under fingertips)
3. L vs D (Index extended: Lateral thumb vs opposed thumb)
4. R vs U vs V (Crossed vs Parallel vs V-spread)
5. M vs N (3 fingers over thumb vs 2 fingers over thumb)
6. O vs T (Circular loop closure vs Thumb tucked under index)
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from alphabet_feature_extractor import extract_canonical_features, FEATURE_NAMES_42
from build_alphabet_dataset import generate_anatomical_hand

def analyze_pair(sign1: str, sign2: str, n_samples: int = 50):
    feats1 = []
    feats2 = []
    for _ in range(n_samples):
        # Scale 1.0, standard ratio
        pts1 = generate_anatomical_hand(sign1, 1.0, 1.0, motor_noise_std=0.02)
        lm1 = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts1]
        feats1.append(extract_canonical_features(lm1))

        pts2 = generate_anatomical_hand(sign2, 1.0, 1.0, motor_noise_std=0.02)
        lm2 = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts2]
        feats2.append(extract_canonical_features(lm2))

    feats1 = np.array(feats1)
    feats2 = np.array(feats2)

    mean1 = np.mean(feats1, axis=0)
    std1 = np.std(feats1, axis=0)
    mean2 = np.mean(feats2, axis=0)
    std2 = np.std(feats2, axis=0)

    # Compute Fisher's separation ratio (Cohen's d-like criterion)
    # separation = |mean1 - mean2| / sqrt(std1^2 + std2^2 + 1e-6)
    sep = np.abs(mean1 - mean2) / np.sqrt(std1**2 + std2**2 + 1e-6)

    # Top distinguishing features
    top_indices = np.argsort(sep)[::-1][:6]

    print(f"\n{'='*75}")
    print(f"CONFUSION PAIR ANALYSIS: '{sign1}' vs '{sign2}'")
    print(f"{'='*75}")
    print(f"{'Idx':<4} | {'Feature Name':<28} | {sign1+' Mean (Std)':<18} | {sign2+' Mean (Std)':<18} | {'Separation'}")
    print(f"{'-'*75}")
    for idx in top_indices:
        f_name = FEATURE_NAMES_42[idx]
        val1 = f"{mean1[idx]:.3f} ({std1[idx]:.3f})"
        val2 = f"{mean2[idx]:.3f} ({std2[idx]:.3f})"
        print(f"{idx:<4} | {f_name:<28} | {val1:<18} | {val2:<18} | {sep[idx]:.2f}")

    return {
        "pair": f"{sign1}_vs_{sign2}",
        "top_features": [{
            "index": int(idx),
            "name": FEATURE_NAMES_42[idx],
            "sign1_mean": float(mean1[idx]),
            "sign1_std": float(std1[idx]),
            "sign2_mean": float(mean2[idx]),
            "sign2_std": float(std2[idx]),
            "separation": float(sep[idx])
        } for idx in top_indices]
    }

if __name__ == "__main__":
    pairs = [
        ("C", "Y"),
        ("A", "E"),
        ("L", "D"),
        ("R", "U"),
        ("R", "V"),
        ("U", "V"),
        ("M", "N"),
        ("O", "T")
    ]
    all_reports = []
    for s1, s2 in pairs:
        all_reports.append(analyze_pair(s1, s2))

    out_file = Path("backend/saved_models/confusion_pairs_analysis.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nSaved confusion pairs analysis to: {out_file}")
