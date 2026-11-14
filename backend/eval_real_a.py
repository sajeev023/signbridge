import json
import numpy as np
from backend.alphabet_feature_extractor import extract_canonical_features

with open('frontend/src/services/single_hand_alphabet_model.json') as f:
    model = json.load(f)
w1 = np.array(model['weights']['w1'])
b1 = np.array(model['weights']['b1'])
w2 = np.array(model['weights']['w2'])
b2 = np.array(model['weights']['b2'])
w3 = np.array(model['weights']['w3'])
b3 = np.array(model['weights']['b3'])
classes = model['classes']

def predict(feat):
    z1 = np.maximum(0, np.dot(w1, feat) + b1)
    z2 = np.maximum(0, np.dot(w2, z1) + b2)
    logits = np.dot(w3, z2) + b3
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    idx = np.argmax(probs)
    return classes[idx], probs[idx], probs

print('=== REAL_WORLD_LANDMARKS_DATASET.JSON ===')
with open('evaluation/real_world_landmarks_dataset.json') as f:
    d = json.load(f)

for s in d:
    if s.get('signLabel') == 'A':
        raw = s['rawMediaPipeDetection']
        lms = raw if isinstance(raw, list) else s.get('landmarks')
        clean_lms = [{'x': p['x'], 'y': p['y'], 'z': p.get('z', 0.0)} for p in lms]
        feat = extract_canonical_features(clean_lms)
        pred, conf, probs = predict(feat)
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(classes[i], round(float(probs[i]), 4)) for i in top3_idx]
        print(f"{s['sampleId']}: signer={s['participantId']} orient='{s['handOrientation']}' => pred={pred} conf={conf:.4f} top3={top3}")

print('\n=== REAL_HUMAN_BENCHMARK.JSON ===')
with open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json') as f:
    bm = json.load(f)

for s in bm.get('records', []):
    if s.get('alphabet_label') == 'A':
        feat = np.array(s['features_42'])
        pred, conf, probs = predict(feat)
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(classes[i], round(float(probs[i]), 4)) for i in top3_idx]
        print(f"{s['capture_id']}: signer={s['signer_id']} => pred={pred} conf={conf:.4f} top3={top3}")
