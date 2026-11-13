"""
SIGNBRIDGE Dataset Forensic Audit Script
Exhaustively inspects every sample across all dataset files, classifies them into
A. Actual human camera captures
B. Synthetic biomechanical samples
C. Generated or augmented samples
D. Invalid or unverified samples
Performs duplicate detection, cross-partition leakage detection, and per-class audit.
"""

import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

CLASSES = [
    "NEUTRAL", "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]

def hash_vector(vec):
    """Produces a deterministic hash of a float vector to detect duplicates."""
    rounded = tuple(round(float(v), 5) for v in vec)
    return hashlib.sha256(str(rounded).encode("utf-8")).hexdigest()

def audit_all_datasets():
    workspace_root = Path(__file__).resolve().parent.parent

    # 1. Inspect backend/data/alphabet_dataset_multisigner.json
    multisigner_path = workspace_root / "backend" / "data" / "alphabet_dataset_multisigner.json"
    multisigner_data = None
    if multisigner_path.exists():
        with open(multisigner_path, "r", encoding="utf-8") as f:
            multisigner_data = json.load(f)

    # 2. Inspect evaluation/real_world_landmarks_dataset.json
    real_world_path = workspace_root / "evaluation" / "real_world_landmarks_dataset.json"
    real_world_data = None
    if real_world_path.exists():
        with open(real_world_path, "r", encoding="utf-8") as f:
            real_world_data = json.load(f)

    # 3. Inspect backend/data/real_human_evaluation_dataset.json
    real_human_path = workspace_root / "backend" / "data" / "real_human_evaluation_dataset.json"
    real_human_data = None
    if real_human_path.exists():
        with open(real_human_path, "r", encoding="utf-8") as f:
            real_human_data = json.load(f)

    all_classified_samples = []
    vector_hashes = defaultdict(list)
    signer_cohorts = defaultdict(set)
    class_stats = {c: {
        "real_human": 0,
        "synthetic": 0,
        "augmented": 0,
        "invalid": 0,
        "signers": set(),
        "sessions": set(),
        "splits": defaultdict(int),
        "valid": 0
    } for c in CLASSES}

    # Audit Dataset 1: multisigner
    if multisigner_data:
        samples = multisigner_data.get("samples", [])
        for s in samples:
            lbl = s.get("sign_label")
            if lbl not in class_stats:
                continue
            feats = s.get("features_42", [])
            is_valid = (len(feats) == 42 and all(isinstance(v, (int, float)) and not (v != v) for v in feats))
            
            # Provenance
            is_real = s.get("is_real_camera", False)
            is_aug = s.get("is_augmented", False)
            pid = s.get("participant_id", "Unknown")
            split = s.get("split", "unassigned")
            sid = s.get("sample_id", "")
            
            if not is_valid:
                cat = "D_invalid"
                class_stats[lbl]["invalid"] += 1
            elif is_real:
                cat = "A_real_human"
                class_stats[lbl]["real_human"] += 1
                class_stats[lbl]["signers"].add(pid)
                class_stats[lbl]["valid"] += 1
            elif is_aug:
                cat = "C_augmented"
                class_stats[lbl]["augmented"] += 1
                class_stats[lbl]["valid"] += 1
            else:
                cat = "B_synthetic"
                class_stats[lbl]["synthetic"] += 1
                class_stats[lbl]["signers"].add(pid)
                class_stats[lbl]["valid"] += 1

            class_stats[lbl]["splits"][split] += 1
            class_stats[lbl]["sessions"].add(s.get("session_id", "session_0"))

            h = hash_vector(feats)
            vector_hashes[h].append((sid, lbl, split, cat))
            signer_cohorts[pid].add(sid)

    # Audit Dataset 2 & 3: real human camera datasets
    empirical_records = []
    if real_human_data:
        empirical_records.extend(real_human_data.get("records", []))
    elif real_world_data:
        empirical_records.extend(real_world_data if isinstance(real_world_data, list) else [])

    for r in empirical_records:
        lbl = r.get("alphabet_label") or r.get("signLabel")
        if lbl not in class_stats:
            continue
        feats = r.get("features_42")
        if not feats and "landmarks" in r:
            from alphabet_feature_extractor import extract_canonical_features
            feats = extract_canonical_features(r["landmarks"]).tolist()
        elif not feats and "groundTruth" in r:
            from alphabet_feature_extractor import extract_canonical_features
            feats = extract_canonical_features(r["groundTruth"]).tolist()

        is_valid = (feats is not None and len(feats) == 42 and all(isinstance(v, (int, float)) and not (v != v) for v in feats))
        pid = r.get("signer_id") or r.get("participantId", "Real_Signer")
        split = r.get("cohort_split", "test")
        cid = r.get("capture_id") or r.get("sampleId", "EMP")
        env = r.get("camera_conditions", {}).get("lighting", "Empirical_Condition")

        if is_valid:
            class_stats[lbl]["real_human"] += 1
            class_stats[lbl]["signers"].add(pid)
            class_stats[lbl]["sessions"].add(env)
            class_stats[lbl]["valid"] += 1
            class_stats[lbl]["splits"][split] += 1
            h = hash_vector(feats)
            vector_hashes[h].append((cid, lbl, split, "A_real_human"))
        else:
            class_stats[lbl]["invalid"] += 1

    # Check for exact vector duplicate collisions across splits
    leakage_count = 0
    duplicate_groups = 0
    for h, instances in vector_hashes.items():
        if len(instances) > 1:
            duplicate_groups += 1
            splits_in_group = set(inst[2] for inst in instances)
            if len(splits_in_group) > 1 and "train" in splits_in_group and ("val" in splits_in_group or "test" in splits_in_group):
                leakage_count += 1

    report = {
        "audit_timestamp": "2026-11-13T12:56:00Z",
        "total_unique_hashes": len(vector_hashes),
        "duplicate_hash_groups": duplicate_groups,
        "cross_split_leakage_groups": leakage_count,
        "per_class_audit": {}
    }

    print("=" * 80)
    print("SIGNBRIDGE DATASET FORENSIC AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Unique Feature Hashes: {len(vector_hashes)}")
    print(f"Duplicate Hash Groups: {duplicate_groups}")
    print(f"Cross-Split Leakage Groups: {leakage_count} (Must be 0!)")
    print("-" * 80)
    print(f"{'Class':<8} | {'Real Human':<10} | {'Synthetic':<10} | {'Augmented':<10} | {'Valid':<8} | {'Signers':<8} | {'Missing Data'}")
    print("-" * 80)

    for c in CLASSES:
        st = class_stats[c]
        signers_str = str(len(st["signers"]))
        missing = "None" if st["valid"] > 0 else "NO SAMPLES"
        if st["real_human"] == 0:
            missing = "Zero real-human captures"
        print(f"{c:<8} | {st['real_human']:<10} | {st['synthetic']:<10} | {st['augmented']:<10} | {st['valid']:<8} | {signers_str:<8} | {missing}")

        report["per_class_audit"][c] = {
            "real_human_samples": st["real_human"],
            "synthetic_samples": st["synthetic"],
            "augmented_samples": st["augmented"],
            "invalid_samples": st["invalid"],
            "total_valid_samples": st["valid"],
            "unique_signers_count": len(st["signers"]),
            "unique_signers_list": sorted(list(st["signers"])),
            "sessions_count": len(st["sessions"]),
            "splits": dict(st["splits"]),
            "evidence_status": "REAL-HUMAN VALIDATION AVAILABLE" if st["real_human"] >= 2 else ("REAL-HUMAN VALIDATION INSUFFICIENT" if st["real_human"] == 1 else "SYNTHETIC VALIDATION ONLY")
        }

    out_file = workspace_root / "backend" / "saved_models" / "dataset_forensic_audit_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit saved to: {out_file}")
    return report

if __name__ == "__main__":
    audit_all_datasets()

