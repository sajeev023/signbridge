import json
import numpy as np
from backend.alphabet_feature_extractor import FEATURE_NAMES_42

with open('backend/data/alphabet_dataset_multisigner.json') as f:
    d = json.load(f)

samples = d['samples']

def get_class_features(label):
    feats = [s['features_42'] for s in samples if s['sign_label'] == label]
    return np.array(feats, dtype=np.float32)

feats_A = get_class_features('A')
print(f"Loaded {len(feats_A)} samples for A")

classes_to_compare = ['E', 'S', 'M', 'N', 'T', 'NEUTRAL']
comparison_results = {}

for target in classes_to_compare:
    feats_target = get_class_features(target)
    mean_A = np.mean(feats_A, axis=0)
    std_A = np.std(feats_A, axis=0) + 1e-6
    mean_T = np.mean(feats_target, axis=0)
    std_T = np.std(feats_target, axis=0) + 1e-6
    
    # Cohen's d: |mean_A - mean_T| / sqrt((std_A^2 + std_T^2)/2)
    pooled_std = np.sqrt(0.5 * (std_A**2 + std_T**2))
    cohens_d = np.abs(mean_A - mean_T) / pooled_std
    
    # Fisher discriminant ratio: (mean_A - mean_T)^2 / (var_A + var_T)
    fisher_ratio = ((mean_A - mean_T)**2) / (std_A**2 + std_T**2)
    
    ranked_indices = np.argsort(fisher_ratio)[::-1]
    
    top_features = []
    for rank in range(6):
        idx = int(ranked_indices[rank])
        top_features.append({
            'rank': rank + 1,
            'index': idx,
            'feature_name': FEATURE_NAMES_42[idx],
            'mean_A': float(mean_A[idx]),
            'std_A': float(std_A[idx]),
            f'mean_{target}': float(mean_T[idx]),
            f'std_{target}': float(std_T[idx]),
            'cohens_d': float(cohens_d[idx]),
            'fisher_ratio': float(fisher_ratio[idx])
        })
    comparison_results[f"A_vs_{target}"] = top_features

print(json.dumps(comparison_results, indent=2))

with open('backend/letter_a_feature_separation.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)
print("Saved backend/letter_a_feature_separation.json")
