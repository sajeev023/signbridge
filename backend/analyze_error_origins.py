import json
import numpy as np

records = json.load(open('backend/titan/datasets/asl_alphabet/metadata/real_human_benchmark.json'))['records']

# 1. Analyze C captures
c_records = [r for r in records if (r.get('alphabet_label') or r.get('target_letter')) == 'C']
print(f"Total C records: {len(c_records)}")

for r in c_records:
    lms = r['landmarks']
    wrist = np.array([lms[0]['x'], lms[0]['y'], lms[0].get('z', 0)])
    thumb_tip = np.array([lms[4]['x'], lms[4]['y'], lms[4].get('z', 0)])
    index_tip = np.array([lms[8]['x'], lms[8]['y'], lms[8].get('z', 0)])
    mcp_mid = np.array([lms[9]['x'], lms[9]['y'], lms[9].get('z', 0)])
    palm_scale = np.linalg.norm(mcp_mid - wrist) or 1.0
    
    a2d = np.linalg.norm(thumb_tip[:2] - index_tip[:2]) / palm_scale
    a3d = np.linalg.norm(thumb_tip - index_tip) / palm_scale
    depth_diff = abs(thumb_tip[2] - index_tip[2]) / palm_scale
    orient = r.get('camera_conditions', {}).get('hand_orientation', 'unknown')
    sid = r.get('signer_id')
    cid = r.get('capture_id')
    cohort = r.get('cohort_split')
    print(f"C: {cid} [{cohort}, {sid}, orient={orient}]: a2D={a2d:.3f}, a3D={a3d:.3f}, depth_diff={depth_diff:.3f}, scale={palm_scale:.3f}")

# 2. Analyze R vs U captures
r_records = [r for r in records if (r.get('alphabet_label') or r.get('target_letter')) in ['R', 'U']]
print(f"\nTotal R/U records: {len(r_records)}")
for r in r_records:
    lms = r['landmarks']
    lbl = r.get('alphabet_label') or r.get('target_letter')
    wrist = np.array([lms[0]['x'], lms[0]['y'], lms[0].get('z', 0)])
    idx_tip = np.array([lms[8]['x'], lms[8]['y'], lms[8].get('z', 0)])
    mid_tip = np.array([lms[12]['x'], lms[12]['y'], lms[12].get('z', 0)])
    mcp_mid = np.array([lms[9]['x'], lms[9]['y'], lms[9].get('z', 0)])
    palm_scale = np.linalg.norm(mcp_mid - wrist) or 1.0
    
    # Crossing metric: middle tip x relative to index tip x
    crossing_dx = (mid_tip[0] - idx_tip[0]) / palm_scale
    dist_tip = np.linalg.norm(mid_tip[:2] - idx_tip[:2]) / palm_scale
    cid = r.get('capture_id')
    sid = r.get('signer_id')
    print(f"{lbl}: {cid} [{sid}]: crossing_dx={crossing_dx:.3f}, inter_tip_dist={dist_tip:.3f}")

# 3. Analyze O captures
o_records = [r for r in records if (r.get('alphabet_label') or r.get('target_letter')) == 'O']
print(f"\nTotal O records: {len(o_records)}")
for r in o_records:
    lms = r['landmarks']
    wrist = np.array([lms[0]['x'], lms[0]['y'], lms[0].get('z', 0)])
    thumb_tip = np.array([lms[4]['x'], lms[4]['y'], lms[4].get('z', 0)])
    idx_tip = np.array([lms[8]['x'], lms[8]['y'], lms[8].get('z', 0)])
    mid_tip = np.array([lms[12]['x'], lms[12]['y'], lms[12].get('z', 0)])
    mcp_mid = np.array([lms[9]['x'], lms[9]['y'], lms[9].get('z', 0)])
    palm_scale = np.linalg.norm(mcp_mid - wrist) or 1.0
    
    t_to_idx = np.linalg.norm(thumb_tip - idx_tip) / palm_scale
    t_to_mid = np.linalg.norm(thumb_tip - mid_tip) / palm_scale
    cid = r.get('capture_id')
    sid = r.get('signer_id')
    print(f"O: {cid} [{sid}]: t_to_idx={t_to_idx:.3f}, t_to_mid={t_to_mid:.3f}")
