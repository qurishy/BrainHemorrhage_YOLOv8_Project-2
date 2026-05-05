"""
Notebook 3: Evaluation & Ablation Studies
"""

# Cell 1: Confidence Sweep
from src.utils.config_loader import load_config
from src.models.evaluate import evaluate_model
import pandas as pd
import matplotlib.pyplot as plt

config = load_config("configs/config.yaml")

conf_results = []
for conf_val in config.inference.conf_sweep:
    metrics = evaluate_model(
        model_path=config.paths.baseline_model,
        conf=conf_val,
        iou=config.inference.default_iou
    )
    conf_results.append(metrics)
    print(f"Conf={conf_val:.3f}: mAP50={metrics['mAP50']:.4f}, F1={metrics['f1']:.4f}")

# Cell 2: NMS IoU Comparison
nms_results = []
for iou_val in config.inference.nms_iou_values:
    metrics = evaluate_model(
        model_path=config.paths.baseline_model,
        conf=0.001,
        iou=iou_val
    )
    nms_results.append(metrics)
    print(f"IoU={iou_val:.2f}: mAP50={metrics['mAP50']:.4f}, F1={metrics['f1']:.4f}")

# Cell 3: Save Results
import json
from pathlib import Path

output_dir = Path(config.paths.outputs_ablation)
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "conf_sweep.json", "w") as f:
    json.dump(conf_results, f, indent=2)

with open(output_dir / "nms_comparison.json", "w") as f:
    json.dump(nms_results, f, indent=2)

print("Ablation results saved!")
