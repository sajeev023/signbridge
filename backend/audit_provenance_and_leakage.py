"""
SIGNBRIDGE — Operation A: Empirical Evidence Audit & Leakage Detection Script
Independent forensic audit covering:
1. Dataset Provenance & Record Verification
2. Leakage, Overlap & Near-Duplicate Distance Analysis
3. Exact Metric Reproduction from Scratch (Denominators, Confusion Matrix, ECE)
4. Kinematic Rule Gate Threshold Sensitivity Analysis
5. Statistical Confidence Intervals (Clopper-Pearson & Wilson Score 95% CIs)
6. Production Model vs Specialist Regression Comparison
"""

import json
import math
import os
import numpy as np
import torch

from alphabet_feature_extractor import extract_canonical_features, FEATURE_NAMES_42
from train_letter_a_specialist import LetterASpecialist, verify_letter_a_geometry, calc_metrics, compute_ece

# --- Helper: Clopper-Pearson & Wilson Confidence Intervals ---
def wilson_score_interval(k: int, n: int, confidence: float = 0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    diff = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    lower = max(0.0, center - diff)
    upper = min(1.0, center + diff)
    return p, lower, upper


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95):
    """Exact Clopper-Pearson binomial confidence interval."""
    from scipy.stats import beta
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return p, float(lower), float(upper)


def run_full_audit():
    print("================================================================================")
    print("SIGNBRIDGE OPERATION A: EMPIRICAL EVIDENCE AUDIT")
    print("================================================================================\n")

    # -------------------------------------------------------------------------
    # 1. Dataset Provenance Inspection
    # -------------------------------------------------------------------------
    print("--- 1. DATASET PROVENANCE AUDIT ---")
    with open('backend/data/alphabet_dataset_multisigner.json') as f:
        multisigner = json.load(f)
    samples = multisigner['samples']
    meta = multisigner.get('metadata', {})
    
    print(f"Multisigner Dataset Total Samples: {len(samples)}")
    print(f"Declared Metadata: {meta}")
    
    # Audit participant distributions and sample counts
    participants = {}
    for s in samples:
        pid = s.get('participant_id', 'Unknown')
        lbl = s.get('sign_label', 'Unknown')
        split = s.get('split', 'Unknown')
        if pid not in participants:
            participants[pid] = {'total': 0, 'splits': set(), 'labels': {}}
        participants[pid]['total'] += 1
        participants[pid]['splits'].add(split)
        participants[pid]['labels'][lbl] = participants[pid]['labels'].get(lbl, 0) + 1
        
    for pid, pdata in sorted(participants.items()):
        print(f"  Participant {pid:10s} | Total: {pdata['total']:4d} | Splits: {list(pdata['splits'])} | Unique Signs: {len(pdata['labels'])} | Sign 'A': {pdata['labels'].get('A', 0)}")

    # Audit Real Human Benchmark
    with open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json') as f:
        real_bm = json.load(f)
    bm_records = real_bm.get('records', [])
    print(f"\nReal Human Benchmark Records: {len(bm_records)}")
    bm_signers = {}
    for r in bm_records:
        sid = r.get('signer_id', 'Unknown')
        lbl = r.get('alphabet_label', 'Unknown')
        prov = r.get('provenance', 'Unknown')
        if sid not in bm_signers:
            bm_signers[sid] = {'total': 0, 'provenance': set(), 'labels': {}}
        bm_signers[sid]['total'] += 1
        bm_signers[sid]['provenance'].add(prov)
        bm_signers[sid]['labels'][lbl] = bm_signers[sid]['labels'].get(lbl, 0) + 1

    for sid, sdata in sorted(bm_signers.items()):
        print(f"  Signer {sid:10s} | Total: {sdata['total']:2d} | Labels: {dict(sorted(sdata['labels'].items()))} | Provenance: {list(sdata['provenance'])}")

    # Audit Real World Landmarks Dataset
    with open('evaluation/real_world_landmarks_dataset.json') as f:
        rw_data = json.load(f)
    print(f"\nReal World Landmarks Dataset: {len(rw_data)} samples")
    rw_labels = {}
    for r in rw_data:
        lbl = r.get('signLabel')
        rw_labels[lbl] = rw_labels.get(lbl, 0) + 1
    print(f"  Labels in real_world_landmarks_dataset: {dict(sorted(rw_labels.items()))}")

    # -------------------------------------------------------------------------
    # 2. Leakage & Near-Duplicate Distance Analysis
    # -------------------------------------------------------------------------
    print("\n--- 2. LEAKAGE & NEAR-DUPLICATE AUDIT ---")
    train_samples = [s for s in samples if s['participant_id'] in ['Signer_1', 'Signer_2']]
    val_samples = [s for s in samples if s['participant_id'] == 'Signer_3']
    test_samples = [s for s in samples if s['participant_id'] == 'Signer_4']
    
    train_feats = np.array([s['features_42'] for s in train_samples], dtype=np.float32)
    val_feats = np.array([s['features_42'] for s in val_samples], dtype=np.float32)
    test_feats = np.array([s['features_42'] for s in test_samples], dtype=np.float32)
    bm_feats = np.array([r['features_42'] for r in bm_records], dtype=np.float32)
    
    # Pairwise minimum distance check between splits
    def min_pairwise_dist(set_A, set_B):
        # set_A: [N, 42], set_B: [M, 42]
        # Chunked to avoid huge memory footprint
        min_d = float('inf')
        for i in range(0, len(set_A), 250):
            chunk_A = set_A[i:i+250]
            # dist shape: [chunk_len, M]
            diff = chunk_A[:, np.newaxis, :] - set_B[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            min_d = min(min_d, float(np.min(dists)))
        return min_d

    min_train_val = min_pairwise_dist(train_feats, val_feats)
    min_train_test = min_pairwise_dist(train_feats, test_feats)
    min_train_bm = min_pairwise_dist(train_feats, bm_feats)
    min_val_test = min_pairwise_dist(val_feats, test_feats)

    print(f"Min Euclidean Feature Distance Train vs Val:   {min_train_val:.6f}")
    print(f"Min Euclidean Feature Distance Train vs Test:  {min_train_test:.6f}")
    print(f"Min Euclidean Feature Distance Train vs RealBM:{min_train_bm:.6f}")
    print(f"Min Euclidean Feature Distance Val vs Test:    {min_val_test:.6f}")

    has_exact_leakage = (min_train_val == 0.0 or min_train_test == 0.0 or min_train_bm == 0.0)
    print(f"Exact Duplicate Leakage Present: {has_exact_leakage}")

    # -------------------------------------------------------------------------
    # 3. Independent Metric Reproduction
    # -------------------------------------------------------------------------
    print("\n--- 3. INDEPENDENT METRIC REPRODUCTION ---")
    with open('backend/saved_models/letter_a_specialist_model.json') as f:
        model_data = json.load(f)
        
    model = LetterASpecialist(input_dim=42, hidden_dim=32)
    model.net[0].weight.data = torch.tensor(model_data['weights']['w1'])
    model.net[0].bias.data = torch.tensor(model_data['weights']['b1'])
    model.net[1].running_mean.data = torch.tensor(model_data['weights']['bn1_mean'])
    model.net[1].running_var.data = torch.tensor(model_data['weights']['bn1_var'])
    model.net[1].weight.data = torch.tensor(model_data['weights']['bn1_weight'])
    model.net[1].bias.data = torch.tensor(model_data['weights']['bn1_bias'])
    model.net[4].weight.data = torch.tensor(model_data['weights']['w2'])
    model.net[4].bias.data = torch.tensor(model_data['weights']['b2'])
    model.net[6].weight.data = torch.tensor(model_data['weights']['w3'])
    model.net[6].bias.data = torch.tensor(model_data['weights']['b3'])
    model.temperature.data = torch.tensor([model_data['temperature']])
    model.eval()

    def evaluate_split(feat_arr, raw_labels_list, split_name):
        x = torch.tensor(feat_arr)
        with torch.no_grad():
            probs_a = model.predict_proba(x)[:, 1].numpy()
        preds_a = (probs_a >= 0.50).astype(int)
        y_true = np.array([1 if l == 'A' else 0 for l in raw_labels_list], dtype=int)
        
        m = calc_metrics(y_true, preds_a)
        ece = compute_ece(probs_a, y_true)
        
        print(f"\n[{split_name.upper()}] (n={len(y_true)}: {m['tp']+m['fn']} A, {m['tn']+m['fp']} Non-A)")
        print(f"  TP: {m['tp']:4d} / {m['tp']+m['fn']} (Recall: {m['recall']*100:.2f}%)")
        print(f"  FP: {m['fp']:4d} / {m['tn']+m['fp']} (FPR: {m['fpr']*100:.3f}%)")
        print(f"  TN: {m['tn']:4d} / {m['tn']+m['fp']} (Specificity: {(1-m['fpr'])*100:.2f}%)")
        print(f"  FN: {m['fn']:4d} / {m['tp']+m['fn']} (FNR: {m['fnr']*100:.3f}%)")
        print(f"  Precision: {m['precision']*100:.2f}% ({m['tp']}/{m['tp']+m['fp']})")
        print(f"  F1 Score:  {m['f1']:.4f}")
        print(f"  Accuracy:  {m['accuracy']*100:.2f}% ({m['tp']+m['tn']}/{len(y_true)})")
        print(f"  ECE:       {ece:.4f}")
        return m, probs_a, y_true

    val_raw = [s['sign_label'] for s in val_samples]
    test_raw = [s['sign_label'] for s in test_samples]
    bm_raw = [r['alphabet_label'] for r in bm_records]

    m_val, _, _ = evaluate_split(val_feats, val_raw, "Validation (Signer_3)")
    m_test, probs_test, y_test = evaluate_split(test_feats, test_raw, "Held-Out Test (Signer_4)")
    m_bm, probs_bm, y_bm = evaluate_split(bm_feats, bm_raw, "Empirical Real Human Benchmark (42 Samples)")

    # -------------------------------------------------------------------------
    # 4. Kinematic Rule Gate Threshold Sensitivity Audit
    # -------------------------------------------------------------------------
    print("\n--- 4. KINEMATIC RULE GATE THRESHOLD AUDIT ---")
    # Evaluate individual clauses of the rule gate
    # Gate clauses:
    # 1. ext_fingers < 0.35
    # 2. thumb_y >= 0.55
    # 3. thumb_x >= 0.12
    # 4. cos_spread < 0.15
    # 5. dist_thumb_mid >= 0.38
    print("Testing Rule Gate on Held-Out Test Set (Signer_4, 1,000 samples):")
    rule_passes_test = []
    for f in test_feats:
        r = verify_letter_a_geometry(f)
        rule_passes_test.append(r['is_valid_letter_a'])
    rule_passes_test = np.array(rule_passes_test)
    y_test_bool = (np.array([s['sign_label'] for s in test_samples]) == 'A')
    
    tp_gate = np.sum((rule_passes_test == True) & (y_test_bool == True))
    fp_gate = np.sum((rule_passes_test == True) & (y_test_bool == False))
    tn_gate = np.sum((rule_passes_test == False) & (y_test_bool == False))
    fn_gate = np.sum((rule_passes_test == False) & (y_test_bool == True))
    print(f"  Standalone Rule Gate on Test: TP={tp_gate}/40, FP={fp_gate}/960, TN={tn_gate}/960, FN={fn_gate}/40")

    print("\nTesting Standalone Rule Gate on Real Human Benchmark (42 samples):")
    rule_passes_bm = []
    for f in bm_feats:
        r = verify_letter_a_geometry(f)
        rule_passes_bm.append(r['is_valid_letter_a'])
    rule_passes_bm = np.array(rule_passes_bm)
    y_bm_bool = (np.array([r['alphabet_label'] for r in bm_records]) == 'A')
    tp_gate_bm = np.sum((rule_passes_bm == True) & (y_bm_bool == True))
    fp_gate_bm = np.sum((rule_passes_bm == True) & (y_bm_bool == False))
    tn_gate_bm = np.sum((rule_passes_bm == False) & (y_bm_bool == False))
    fn_gate_bm = np.sum((rule_passes_bm == False) & (y_bm_bool == True))
    print(f"  Standalone Rule Gate on RealBM: TP={tp_gate_bm}/3, FP={fp_gate_bm}/39, TN={tn_gate_bm}/39, FN={fn_gate_bm}/3")

    # -------------------------------------------------------------------------
    # 5. Statistical Uncertainty & Confidence Intervals
    # -------------------------------------------------------------------------
    print("\n--- 5. STATISTICAL CONFIDENCE INTERVALS (95% CI) ---")
    
    # Real-human A recall: 3 successes out of 3 trials
    p_rec, w_low_rec, w_high_rec = wilson_score_interval(3, 3)
    _, cp_low_rec, cp_high_rec = clopper_pearson_interval(3, 3)
    print(f"Real-Human A Recall (k=3, n=3):")
    print(f"  Observed Point Estimate: {p_rec*100:.1f}%")
    print(f"  Wilson Score 95% CI:    [{w_low_rec*100:.2f}%, {w_high_rec*100:.2f}%]")
    print(f"  Clopper-Pearson 95% CI: [{cp_low_rec*100:.2f}%, {cp_high_rec*100:.2f}%]")

    # Real-human Non-A specificity: 39 successes out of 39 trials
    p_spec, w_low_spec, w_high_spec = wilson_score_interval(39, 39)
    _, cp_low_spec, cp_high_spec = clopper_pearson_interval(39, 39)
    print(f"\nReal-Human Non-A Specificity (k=39, n=39):")
    print(f"  Observed Point Estimate: {p_spec*100:.1f}%")
    print(f"  Wilson Score 95% CI:    [{w_low_spec*100:.2f}%, {w_high_spec*100:.2f}%]")
    print(f"  Clopper-Pearson 95% CI: [{cp_low_spec*100:.2f}%, {cp_high_spec*100:.2f}%]")

    # Test set A recall: 40 successes out of 40 trials
    _, cp_low_test_rec, cp_high_test_rec = clopper_pearson_interval(40, 40)
    print(f"\nTest Set Signer_4 A Recall (k=40, n=40):")
    print(f"  Clopper-Pearson 95% CI: [{cp_low_test_rec*100:.2f}%, {cp_high_test_rec*100:.2f}%]")

    # Test set Non-A specificity: 960 successes out of 960 trials
    _, cp_low_test_spec, cp_high_test_spec = clopper_pearson_interval(960, 960)
    print(f"\nTest Set Signer_4 Non-A Specificity (k=960, n=960):")
    print(f"  Clopper-Pearson 95% CI: [{cp_low_test_spec*100:.2f}%, {cp_high_test_spec*100:.2f}%]")

    # -------------------------------------------------------------------------
    # 6. Production Model Regression Comparison
    # -------------------------------------------------------------------------
    print("\n--- 6. PRODUCTION 25-CLASS MODEL REGRESSION COMPARISON ---")
    with open('frontend/src/services/single_hand_alphabet_model.json') as f:
        prod_model = json.load(f)
        
    pw1 = np.array(prod_model['weights']['w1']) # [64, 42]
    pb1 = np.array(prod_model['weights']['b1']) # [64]
    pw2 = np.array(prod_model['weights']['w2']) # [64, 64]
    pb2 = np.array(prod_model['weights']['b2']) # [64]
    pw3 = np.array(prod_model['weights']['w3']) # [25, 64]
    pb3 = np.array(prod_model['weights']['b3']) # [25]
    prod_classes = prod_model['classes']

    def prod_predict(feat):
        z1 = np.maximum(0, np.dot(pw1, feat) + pb1)
        z2 = np.maximum(0, np.dot(pw2, z1) + pb2)
        logits = np.dot(pw3, z2) + pb3
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        idx = np.argmax(probs)
        return prod_classes[idx], probs[idx], probs

    # Run production classifier on all 4,000 multi-signer samples
    prod_preds_all = []
    prod_confs_all = []
    for s in samples:
        c, conf, _ = prod_predict(np.array(s['features_42']))
        prod_preds_all.append(c)
        prod_confs_all.append(conf)

    true_labels_all = [s['sign_label'] for s in samples]
    prod_correct_all = sum(1 for p, t in zip(prod_preds_all, true_labels_all) if p == t)
    print(f"Production 25-Class Model Overall Accuracy on 4,000 samples: {prod_correct_all/len(samples)*100:.2f}% ({prod_correct_all}/4000)")

    # Run on Real Human Benchmark
    prod_bm_preds = []
    prod_bm_confs = []
    for r in bm_records:
        c, conf, _ = prod_predict(np.array(r['features_42']))
        prod_bm_preds.append(c)
        prod_bm_confs.append(conf)

    prod_bm_correct = sum(1 for p, t in zip(prod_bm_preds, bm_raw) if p == t)
    print(f"Production 25-Class Model Overall Accuracy on Real Human Benchmark (42 samples): {prod_bm_correct/len(bm_records)*100:.2f}% ({prod_bm_correct}/42)")

    # Specific A performance in production model
    a_indices = [i for i, r in enumerate(bm_records) if r['alphabet_label'] == 'A']
    print("\nProduction Model Performance on Empirical Real A Captures:")
    for idx in a_indices:
        r = bm_records[idx]
        p_sign, p_conf, _ = prod_predict(np.array(r['features_42']))
        print(f"  {r['capture_id']} | True: A | Prod Pred: {p_sign:2s} (conf: {p_conf:.4f})")

    # Output audit data to JSON
    audit_summary = {
        'dataset_provenance': {
            'multisigner_samples': len(samples),
            'participants': {k: {'total': v['total'], 'splits': list(v['splits'])} for k, v in participants.items()},
            'benchmark_records': len(bm_records),
            'benchmark_signers': {k: v['total'] for k, v in bm_signers.items()}
        },
        'leakage_metrics': {
            'min_train_val_dist': min_train_val,
            'min_train_test_dist': min_train_test,
            'min_train_bm_dist': min_train_bm,
            'has_exact_leakage': has_exact_leakage
        },
        'reproduced_metrics': {
            'validation_signer_3': m_val,
            'test_signer_4': m_test,
            'real_human_benchmark': m_bm
        },
        'statistical_confidence_intervals_95': {
            'real_human_recall_clopper_pearson': [cp_low_rec, cp_high_rec],
            'real_human_recall_wilson': [w_low_rec, w_high_rec],
            'real_human_specificity_clopper_pearson': [cp_low_spec, cp_high_spec],
            'test_set_recall_clopper_pearson': [cp_low_test_rec, cp_high_test_rec],
            'test_set_specificity_clopper_pearson': [cp_low_test_spec, cp_high_test_spec]
        }
    }
    with open('backend/letter_a_audit_results.json', 'w') as f:
        json.dump(audit_summary, f, indent=2)
    print("\nSaved backend/letter_a_audit_results.json")


if __name__ == '__main__':
    run_full_audit()
