"""
Complete pipeline runner script with GPU support.
Runs the entire project workflow from download to evaluation.

Usage:
    python scripts/run_pipeline.py --api-key YOUR_ROBOFLOW_API_KEY
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import subprocess
import torch


def check_gpu():
    """Check GPU availability and print info."""
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        print("✅ GPU is available and will be used!")
    else:
        print("❌ No GPU detected. Training will use CPU (very slow!)")
        print("   To use GPU, install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("="*60)


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")

    # Ensure we run from project root
    project_root = Path(__file__).parent.parent
    result = subprocess.run(cmd, shell=True, check=False, cwd=str(project_root))

    if result.returncode != 0:
        print(f"[WARNING] Step failed with code {result.returncode}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run complete project pipeline")
    parser.add_argument("--api-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    args = parser.parse_args()

    # Check GPU first
    check_gpu()

    steps = []

    # Step 1: Download dataset
    if not args.skip_download:
        steps.append((
            f"python -m src.data.download_dataset --api-key {args.api_key}",
            "Download Dataset from Roboflow"
        ))

    # Step 2: Prepare dataset
    steps.append((
        "python -m src.data.prepare_dataset",
        "Prepare Dataset (Custom Split + data.yaml)"
    ))

    # Step 3: Baseline training
    if not args.skip_training:
        steps.append((
            "python -m src.models.train_baseline",
            "Baseline Model Training (YOLOv8n, 640px) - GPU"
        ))

        # Step 4: Ablation A training
        steps.append((
            "python -m src.models.train_ablation_a",
            "Ablation A Training (YOLOv8n, 960px) - GPU"
        ))

    # Step 5: Evaluation
    steps.append((
        "python -m src.models.evaluate --model outputs/models/baseline_yolov8n_best.pt --output outputs/ablation/baseline_eval.json",
        "Baseline Evaluation - GPU"
    ))

    # Run all steps
    print("\n" + "="*60)
    print("BRAIN HEMORRHAGE DETECTION - FULL PIPELINE")
    print("="*60)

    success_count = 0
    for cmd, desc in steps:
        if run_command(cmd, desc):
            success_count += 1

    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE: {success_count}/{len(steps)} steps successful")
    print("="*60)


if __name__ == "__main__":
    main()
