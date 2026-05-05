"""
Dataset preparation script.
Creates custom split and data.yaml.

Usage:
    python -m src.data.prepare_dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.dataset_utils import create_data_split, create_data_yaml, analyze_dataset

def find_config():
    """Find config.yaml from multiple possible locations."""
    possible_paths = [
        "configs/config.yaml",
        Path(__file__).parent.parent.parent / "configs" / "config.yaml",
        Path.cwd() / "configs" / "config.yaml",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return str(path)

    raise FileNotFoundError("config.yaml not found. Please run from project root.")


def main():
    config_path = find_config()
    config = load_config(config_path)
    print(f"[INFO] Loaded config from: {config_path}")

    # Find downloaded dataset - check multiple locations
    possible_raw_dirs = [
        Path(config.paths.data_raw),
        Path.cwd() / "data" / "raw",
        Path.cwd(),  # Project root (where roboflow downloads to)
    ]

    source_dir = None
    for raw_dir in possible_raw_dirs:
        if not raw_dir.exists():
            continue

        # Look for brain-hemorrhage-detection-* folders
        dataset_dirs = list(raw_dir.glob("brain-hemorrhage-detection*"))
        if dataset_dirs:
            source_dir = dataset_dirs[0]
            print(f"[INFO] Found dataset at: {source_dir}")
            break

    if source_dir is None:
        print("[ERROR] No dataset found. Run download_dataset.py first.")
        print("[INFO] Searched in:")
        for d in possible_raw_dirs:
            print(f"  - {d}")
        return

    # Create custom split
    print("[INFO] Creating custom split...")
    counts = create_data_split(
        source_dir=str(source_dir),
        output_dir=config.paths.data_splits,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
        test_ratio=config.split.test_ratio,
        seed=config.split.seed
    )

    print(f"[INFO] Split created: {counts}")

    # Create data.yaml
    yaml_path = create_data_yaml(
        split_dir=config.paths.data_splits,
        class_names=config.dataset.classes,
        output_path="data.yaml"
    )
    print(f"[INFO] data.yaml created: {yaml_path}")

    # Analyze dataset
    print("[INFO] Analyzing dataset...")
    stats = analyze_dataset(
        split_dir=config.paths.data_splits,
        class_names=config.dataset.classes,
        split="train"
    )

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*60)

if __name__ == "__main__":
    main()
