import json
import torch
import numpy as np
from backend.titan.asl_architecture import (
    ASLSpatialGNN,
    ASL_CLASS_TO_IDX,
    ASL_VOCABULARY,
    build_adjacency_matrix,
    compute_normalized_laplacian
)
from backend.titan.spatial_encoder import landmarks_to_node_features
from backend.alphabet_feature_extractor import extract_canonical_features

with open('backend/saved_models/titan_asl_spatial_model.json') as f:
    m = json.load(f)

model = ASLSpatialGNN(node_in_dim=6, hidden_dim=64, aux_dim=42, num_classes=27, dropout=0.0)
model.load_state_dict({k: torch.tensor(v) for k, v in m['state_dict'].items()}, strict=False)
model.eval()

adj = torch.tensor(compute_normalized_laplacian(build_adjacency_matrix()), dtype=torch.float32)
temp = m.get('temperature', 1.08)

records = json.load(open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json'))['records']
test_records = [r for r in records if r.get('cohort_split') == 'test']

print(f"Total held-out test samples: {len(test_records)}")

results = []
for r in test_records:
    lbl = r.get('alphabet_label') or r.get('target_letter')
    t_idx = ASL_CLASS_TO_IDX[lbl]
    sid = r.get('signer_id')
    cid = r.get('capture_id')
    cam = r.get('camera_conditions', {})
    
    feats = r.get('features_42')
    if feats is None:
        feats = extract_canonical_features(r['landmarks']).tolist()
    nodes = landmarks_to_node_features(r['landmarks'])
    
    with torch.no_grad():
        logits = model(torch.tensor(nodes).unsqueeze(0), adj, torch.tensor(feats).unsqueeze(0))
        calib_logits = logits / temp
        probs = torch.softmax(calib_logits, dim=1)[0].numpy()
        pred_idx = int(np.argmax(probs))
        p_conf = float(probs[pred_idx])
        t_conf = float(probs[t_idx])
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    
    pred_lbl = ASL_VOCABULARY[pred_idx]
    is_correct = (pred_idx == t_idx)
    results.append({
        'capture_id': cid,
        'signer_id': sid,
        'true': lbl,
        'pred': pred_lbl,
        'correct': is_correct,
        'conf': p_conf,
        't_conf': t_conf,
        'entropy': entropy,
        'lighting': cam.get('lighting'),
        'distance': cam.get('distance_m'),
        'orientation': cam.get('hand_orientation'),
        'palm_scale': cam.get('palm_scale')
    })

correct_count = sum(1 for r in results if r['correct'])
print(f"Accuracy: {correct_count}/{len(results)} ({correct_count/len(results)*100:.2f}%)")

print("\n=== ALL INCORRECT SAMPLES (HELD-OUT TEST) ===")
for r in results:
    if not r['correct']:
        print(f"[{r['signer_id']}] {r['capture_id']}: True={r['true']} -> Pred={r['pred']} | pred_conf={r['conf']:.3f}, true_conf={r['t_conf']:.3f}, H={r['entropy']:.2f} | Dist={r['distance']}m, Light='{r['lighting']}', Orient='{r['orientation']}'")

print("\n=== ALL CORRECT SAMPLES (HELD-OUT TEST) ===")
for r in results:
    if r['correct']:
        print(f"[{r['signer_id']}] {r['capture_id']}: True={r['true']} | conf={r['conf']:.3f}, H={r['entropy']:.2f} | Dist={r['distance']}m, Light='{r['lighting']}', Orient='{r['orientation']}'")
