"""
Notebook 2: Model Training
Run baseline and ablation training.
"""

# Cell 1: Baseline Training
from src.models.train_baseline import main as train_baseline
train_baseline()

# Cell 2: Ablation A Training
from src.models.train_ablation_a import main as train_ablation
train_ablation()

# Cell 3: Evaluate Baseline
from src.models.evaluate import evaluate_model
from src.utils.config_loader import load_config

config = load_config("configs/config.yaml")

baseline_results = evaluate_model(
    model_path=config.paths.baseline_model,
    conf=config.inference.default_conf,
    iou=config.inference.default_iou
)

print("Baseline Results:")
for k, v in baseline_results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
