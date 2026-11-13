import json

with open("backend/saved_models/titan_ablation_results.json") as f:
    res = json.load(f)

print(f"{'Architecture':<35} | {'Acc':<7} | {'Macro F1':<8} | {'ECE':<7} | {'Brier':<7} | {'Latency':<10} | {'Params':<8}")
print("-" * 95)
for k, v in res.items():
    print(f"{k:<35} | {v['accuracy']*100:6.2f}% | {v['macro_f1']*100:6.2f}% | {v['ece_calibrated']:7.4f} | {v['brier_score']:7.4f} | {v['avg_latency_ms']:8.4f}ms | {v['parameter_count']:8d}")
