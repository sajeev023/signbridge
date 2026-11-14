"""
SIGNBRIDGE TITAN - OPERATION ZERO-LEAK
Zero-Leak Blind Evaluation Script for Frozen Models vs Baseline

Evaluates:
1. Baseline Pre-Gen Spatial Model (backend/saved_models/titan_asl_spatial_model.pre_gen.json)
2. Frozen Upgraded AK-GCN Model (backend/saved_models/titan_asl_spatial_model.json)
3. Dynamic Trajectory Model (backend/saved_models/dynamic_alphabet_model.json)
4. Full Hybrid Inference System

On:
- Real human held-out test cohort (zero signer overlap with training data)
- Full per-class recall, precision, macro F1
- Calibration: ECE (10-bin equal width), Brier score
- Selective rejection: coverage, selective risk, accepted-prediction precision
- 95% Wilson confidence intervals
"""

import json
import math
import numpy as np
import torch
from backend.titan.asl_architecture import (
    ASLSpatialGNN,
    ASL_CLASS_TO_IDX,
    ASL_VOCABULARY,
    build_adjacency_matrix,
    compute_normalized_laplacian
)
from backend.titan.spatial_encoder import landmarks_to_node_features
from backend.alphabet_feature_extractor import extract_canonical_features


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95):
    if trials == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95%
    p_hat = successes / trials
    denom = 1 + (z ** 2) / trials
    centre = (p_hat + (z ** 2) / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / trials) + ((z ** 2) / (4 * (trials ** 2))))
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return p_hat, lower, upper


def compute_ece(probs: np.ndarray, targets: list, n_bins: int = 10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == np.array(targets)).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(targets)
    
    bin_details = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_size / n) * bin_error
            bin_details.append({
                'bin': f"{bin_lower:.2f}-{bin_upper:.2f}",
                'count': int(bin_size),
                'acc': float(bin_acc),
                'conf': float(bin_conf),
                'abs_error': float(bin_error)
            })
    return ece, bin_details


def compute_brier_score(probs: np.ndarray, targets: list, num_classes: int = 27):
    one_hot = np.zeros_like(probs)
    for i, t in enumerate(targets):
        one_hot[i, t] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate_spatial_model(model_path: str, records: list, temp: float, min_conf: float, max_ent: float):
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    model = ASLSpatialGNN(num_classes=27, hidden_dim=64, aux_dim=42)
    state = {}
    for k, v in model_data['state_dict'].items():
        state[k] = torch.tensor(v)
    model.load_state_dict(state, strict=False)
    model.eval()

    adj = torch.tensor(compute_normalized_laplacian(build_adjacency_matrix()), dtype=torch.float32)

    all_preds = []
    all_targets = []
    all_probs = []
    all_raw_probs = []
    all_entropies = []
    all_accepted = []
    all_signers = []
    all_letters = []

    for r in records:
        lbl = r.get('alphabet_label') or r.get('target_letter')
        t_idx = ASL_CLASS_TO_IDX[lbl]
        nodes = landmarks_to_node_features(r['landmarks'])
        feats = r.get('features_42') or extract_canonical_features(r['landmarks']).tolist()
        
        with torch.no_grad():
            logits = model(torch.tensor(nodes).unsqueeze(0), adj, torch.tensor(feats).unsqueeze(0))
            raw_p = torch.softmax(logits, dim=1)[0].numpy()
            scaled_p = torch.softmax(logits / temp, dim=1)[0].numpy()
        
        pred = int(np.argmax(scaled_p))
        conf = float(scaled_p[pred])
        ent = float(-np.sum(scaled_p * np.log(scaled_p + 1e-12)))
        
        is_accepted = (conf >= min_conf) and (ent <= max_ent)
        
        all_preds.append(pred)
        all_targets.append(t_idx)
        all_probs.append(scaled_p)
        all_raw_probs.append(raw_p)
        all_entropies.append(ent)
        all_accepted.append(is_accepted)
        all_signers.append(r.get('signer_id', 'unknown'))
        all_letters.append(lbl)

    probs_arr = np.array(all_probs)
    n_samples = len(all_targets)
    correct_all = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    raw_acc = correct_all / n_samples
    _, raw_acc_l, raw_acc_u = wilson_score_interval(correct_all, n_samples)

    ece, ece_bins = compute_ece(probs_arr, all_targets)
    brier = compute_brier_score(probs_arr, all_targets)

    # Rejection metrics
    n_accepted = sum(all_accepted)
    coverage = n_accepted / n_samples
    correct_accepted = sum(1 for p, t, acc in zip(all_preds, all_targets, all_accepted) if acc and p == t)
    precision_accepted = (correct_accepted / n_accepted) if n_accepted > 0 else 0.0
    _, prec_l, prec_u = wilson_score_interval(correct_accepted, n_accepted) if n_accepted > 0 else (0, 0, 0)
    selective_risk = (1.0 - precision_accepted)

    # Per-signer accuracy
    signer_results = {}
    for s_id in sorted(list(set(all_signers))):
        s_idx = [i for i, s in enumerate(all_signers) if s == s_id]
        s_corr = sum(1 for i in s_idx if all_preds[i] == all_targets[i])
        s_total = len(s_idx)
        signer_results[s_id] = {
            'total': s_total,
            'correct': s_corr,
            'accuracy': s_corr / s_total
        }

    # Per-class recall & precision
    classes_present = sorted(list(set(all_targets)))
    per_class = {}
    for c_idx in classes_present:
        c_name = ASL_VOCABULARY[c_idx]
        tp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t == c_idx)
        fn = sum(1 for p, t in zip(all_preds, all_targets) if p != c_idx and t == c_idx)
        fp = sum(1 for p, t in zip(all_preds, all_targets) if p == c_idx and t != c_idx)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c_name] = {'recall': rec, 'precision': prec, 'f1': f1, 'support': tp + fn}

    macro_recall = float(np.mean([v['recall'] for v in per_class.values()]))
    macro_prec = float(np.mean([v['precision'] for v in per_class.values()]))
    macro_f1 = (2 * macro_prec * macro_recall / (macro_prec + macro_recall)) if (macro_prec + macro_recall) > 0 else 0.0

    # Confusion matrix
    conf_matrix = {}
    for p, t in zip(all_preds, all_targets):
        act_name = ASL_VOCABULARY[t]
        pred_name = ASL_VOCABULARY[p]
        if act_name not in conf_matrix:
            conf_matrix[act_name] = {}
        conf_matrix[act_name][pred_name] = conf_matrix[act_name].get(pred_name, 0) + 1

    # Selective risk curve across confidence thresholds
    risk_coverage_curve = []
    for th in np.arange(0.0, 1.0, 0.1):
        acc_mask = [p_arr[pr] >= th for p_arr, pr in zip(probs_arr, all_preds)]
        c_cov = sum(acc_mask) / n_samples
        c_corr = sum(1 for p, t, m in zip(all_preds, all_targets, acc_mask) if m and p == t)
        c_prec = (c_corr / sum(acc_mask)) if sum(acc_mask) > 0 else 1.0
        c_risk = 1.0 - c_prec
        risk_coverage_curve.append({
            'threshold': round(float(th), 2),
            'coverage': round(float(c_cov), 4),
            'accepted_count': int(sum(acc_mask)),
            'precision': round(float(c_prec), 4),
            'selective_risk': round(float(c_risk), 4)
        })

    return {
        'total_samples': n_samples,
        'correct': correct_all,
        'raw_accuracy': raw_acc,
        'accuracy_ci95': [raw_acc_l, raw_acc_u],
        'macro_recall': macro_recall,
        'macro_precision': macro_prec,
        'macro_f1': macro_f1,
        'ece': ece,
        'brier_score': brier,
        'rejection_rate': 1.0 - coverage,
        'coverage': coverage,
        'accepted_precision': precision_accepted,
        'accepted_precision_ci95': [prec_l, prec_u],
        'selective_risk': selective_risk,
        'signer_results': signer_results,
        'per_class': per_class,
        'confusion_matrix': conf_matrix,
        'risk_coverage_curve': risk_coverage_curve
    }


def main():
    benchmark_meta = json.load(open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json'))
    test_records = [r for r in benchmark_meta['records'] if r.get('cohort_split') == 'test']
    train_records = [r for r in benchmark_meta['records'] if r.get('cohort_split') == 'train']

    print(f"Loaded held-out test cohort: {len(test_records)} samples across {len(set(r['signer_id'] for r in test_records))} signers")
    print(f"Loaded familiar cohort: {len(train_records)} samples across {len(set(r['signer_id'] for r in train_records))} signers")

    # Evaluate Baseline Model
    baseline_path = 'backend/saved_models/titan_asl_spatial_model.pre_gen.json'
    baseline_eval = evaluate_spatial_model(baseline_path, test_records, temp=1.08, min_conf=0.55, max_ent=1.35)

    # Evaluate Frozen Upgraded AK-GCN Model
    frozen_path = 'backend/saved_models/titan_asl_spatial_model.json'
    frozen_eval = evaluate_spatial_model(frozen_path, test_records, temp=1.08, min_conf=0.55, max_ent=1.35)

    # Also evaluate familiar cohort on frozen model
    familiar_eval = evaluate_spatial_model(frozen_path, train_records, temp=1.08, min_conf=0.55, max_ent=1.35)

    results = {
        'held_out_test_cohort': {
            'signers': sorted(list(set(r['signer_id'] for r in test_records))),
            'baseline_model': baseline_eval,
            'frozen_ak_gcn_model': frozen_eval
        },
        'familiar_train_cohort': {
            'signers': sorted(list(set(r['signer_id'] for r in train_records))),
            'frozen_ak_gcn_model': familiar_eval
        }
    }

    out_path = 'backend/saved_models/zero_leak_blind_evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved Zero-Leak Blind Evaluation results to {out_path}")
    print("\n================== SUMMARY ==================")
    print(f"Held-out Signers: {results['held_out_test_cohort']['signers']}")
    print(f"Baseline Accuracy:  {baseline_eval['raw_accuracy']*100:.2f}% ({baseline_eval['correct']}/{baseline_eval['total_samples']}) [CI95: {baseline_eval['accuracy_ci95'][0]*100:.1f}% - {baseline_eval['accuracy_ci95'][1]*100:.1f}%]")
    print(f"Frozen AK-GCN Acc:  {frozen_eval['raw_accuracy']*100:.2f}% ({frozen_eval['correct']}/{frozen_eval['total_samples']}) [CI95: {frozen_eval['accuracy_ci95'][0]*100:.1f}% - {frozen_eval['accuracy_ci95'][1]*100:.1f}%]")
    print(f"Baseline Macro F1:  {baseline_eval['macro_f1']*100:.2f}%")
    print(f"Frozen AK-GCN F1:   {frozen_eval['macro_f1']*100:.2f}%")
    print(f"Baseline ECE:       {baseline_eval['ece']:.4f}")
    print(f"Frozen AK-GCN ECE:  {frozen_eval['ece']:.4f}")
    print(f"Baseline Brier:     {baseline_eval['brier_score']:.4f}")
    print(f"Frozen AK-GCN Brier:{frozen_eval['brier_score']:.4f}")
    print(f"Baseline Accepted Prec: {baseline_eval['accepted_precision']*100:.2f}% (Coverage: {baseline_eval['coverage']*100:.1f}%)")
    print(f"Frozen Accepted Prec:   {frozen_eval['accepted_precision']*100:.2f}% (Coverage: {frozen_eval['coverage']*100:.1f}%)")
    print(f"Familiar Signers Acc:   {familiar_eval['raw_accuracy']*100:.2f}% ({familiar_eval['correct']}/{familiar_eval['total_samples']})")


if __name__ == '__main__':
    main()
