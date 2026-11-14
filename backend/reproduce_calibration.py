import json
import numpy as np
import torch
from train_letter_a_specialist import LetterASpecialist, compute_ece, verify_letter_a_geometry

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

# Helper for correct ECE:
def calculate_ece_and_brier(all_probs, y_true, n_bins=10):
    # all_probs: [N, 2] (column 0: Non-A, column 1: A)
    # y_true: [N] (0 or 1)
    probs_a = all_probs[:, 1]
    pred_classes = np.argmax(all_probs, axis=-1)
    confidences = np.max(all_probs, axis=-1)
    
    # Standard multi-class / binary ECE: bins partitioned over [1/K, 1.0] -> [0.5, 1.0]
    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= b_low) & (confidences <= b_high if i == n_bins - 1 else confidences < b_high)
        count = int(np.sum(in_bin))
        if count > 0:
            acc = float(np.mean(pred_classes[in_bin] == y_true[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            weight = float(count / len(y_true))
            diff = abs(avg_conf - acc)
            ece += diff * weight
            bin_details.append({
                'bin': f"[{b_low:.2f}, {b_high:.2f}]",
                'count': count,
                'accuracy': round(acc, 4),
                'confidence': round(avg_conf, 4),
                'abs_error': round(diff, 4)
            })
            
    brier = float(np.mean((probs_a - y_true)**2))
    return float(ece), brier, bin_details


def run_calibration_audit():
    with open('backend/data/alphabet_dataset_multisigner.json') as f:
        multisigner = json.load(f)

    # 1. Validation Split (Signer_3)
    val_samples = [s for s in multisigner['samples'] if s['participant_id'] == 'Signer_3']
    val_feats = np.array([s['features_42'] for s in val_samples], dtype=np.float32)
    val_y = np.array([1 if s['sign_label'] == 'A' else 0 for s in val_samples], dtype=int)

    with torch.no_grad():
        val_probs = model.predict_proba(torch.tensor(val_feats)).numpy()
    val_buggy_ece = compute_ece(val_probs[:, 1], val_y)
    val_true_ece, val_brier, val_bins = calculate_ece_and_brier(val_probs, val_y)

    # 2. Test Split (Signer_4)
    test_samples = [s for s in multisigner['samples'] if s['participant_id'] == 'Signer_4']
    test_feats = np.array([s['features_42'] for s in test_samples], dtype=np.float32)
    test_y = np.array([1 if s['sign_label'] == 'A' else 0 for s in test_samples], dtype=int)

    with torch.no_grad():
        test_probs = model.predict_proba(torch.tensor(test_feats)).numpy()
    test_buggy_ece = compute_ece(test_probs[:, 1], test_y)
    test_true_ece, test_brier, test_bins = calculate_ece_and_brier(test_probs, test_y)

    # 3. Real Human Benchmark (42 samples)
    with open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json') as f:
        bm = json.load(f)
    bm_feats = np.array([r['features_42'] for r in bm['records']], dtype=np.float32)
    bm_y = np.array([1 if r['alphabet_label'] == 'A' else 0 for r in bm['records']], dtype=int)

    with torch.no_grad():
        bm_probs = model.predict_proba(torch.tensor(bm_feats)).numpy()
    bm_buggy_ece = compute_ece(bm_probs[:, 1], bm_y)
    bm_true_ece, bm_brier, bm_bins = calculate_ece_and_brier(bm_probs, bm_y)

    # 4. Gated Confidence Analysis:
    # What happens when we evaluate confidence AFTER the kinematic rule gate?
    # For any sample where the gate FAILS, the output is forced to Class 0 (Non-A / Reject) with confidence 1.0.
    test_probs_gated = np.copy(test_probs)
    for i, feat in enumerate(test_feats):
        gate_res = verify_letter_a_geometry(feat)
        if not gate_res['is_valid_letter_a']:
            # Gate rejects A; probability of A forced to 0.0, Non-A to 1.0
            test_probs_gated[i, 0] = 1.0
            test_probs_gated[i, 1] = 0.0
            
    gated_ece, gated_brier, _ = calculate_ece_and_brier(test_probs_gated, test_y)

    print("================================================================================")
    print("CALIBRATION CORRECTION AUDIT RESULTS")
    print("================================================================================")
    print(f"Validation Set (Signer_3, n=1,000):")
    print(f"  Flawed Reported ECE: {val_buggy_ece:.4f}")
    print(f"  Corrected True ECE:  {val_true_ece:.4f}")
    print(f"  Brier Score:         {val_brier:.6f}")
    print(f"  Confidence Bins:     {val_bins}\n")

    print(f"Held-Out Test Set (Signer_4, n=1,000):")
    print(f"  Flawed Reported ECE: {test_buggy_ece:.4f}")
    print(f"  Corrected True ECE:  {test_true_ece:.4f}")
    print(f"  Brier Score:         {test_brier:.6f}")
    print(f"  Confidence Bins:     {test_bins}\n")

    print(f"Real Human Benchmark (n=42):")
    print(f"  Flawed Reported ECE: {bm_buggy_ece:.4f}")
    print(f"  Corrected True ECE:  {bm_true_ece:.4f}")
    print(f"  Brier Score:         {bm_brier:.6f}")
    print(f"  Confidence Bins:     {bm_bins}\n")

    print(f"Held-Out Test Set Post-Rule-Gate:")
    print(f"  Gated Corrected ECE: {gated_ece:.4f}")
    print(f"  Gated Brier Score:   {gated_brier:.6f}\n")

    # Persist corrected calibration results
    calib_out = {
        'flawed_reported_metrics': {
            'val_ece': val_buggy_ece,
            'test_ece': test_buggy_ece,
            'real_bm_ece': bm_buggy_ece
        },
        'corrected_metrics': {
            'val': {'ece': val_true_ece, 'brier': val_brier, 'bins': val_bins},
            'test': {'ece': test_true_ece, 'brier': test_brier, 'bins': test_bins},
            'real_bm': {'ece': bm_true_ece, 'brier': bm_brier, 'bins': bm_bins},
            'test_gated': {'ece': gated_ece, 'brier': gated_brier}
        }
    }
    with open('backend/letter_a_calibration_corrected.json', 'w') as f:
        json.dump(calib_out, f, indent=2)
    print("Saved backend/letter_a_calibration_corrected.json")

if __name__ == '__main__':
    run_calibration_audit()
