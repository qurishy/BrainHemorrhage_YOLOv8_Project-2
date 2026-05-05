"""
Dataset utilities for brain hemorrhage detection.
Handles data splitting, label parsing, and statistics.
"""

import os
import shutil
import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split


def parse_yolo_label(label_path: Path) -> List[Tuple]:
    """
    Parse YOLO format label file.

    Args:
        label_path: Path to .txt label file

    Returns:
        List of tuples: (class_id, cx, cy, w, h)
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                boxes.append((cls_id, cx, cy, w, h))
    return boxes


def create_data_split(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> Dict[str, int]:
    """
    Create custom train/val/test split from dataset.

    Args:
        source_dir: Source directory with images and labels
        output_dir: Output directory for split data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
        extensions: Supported image extensions

    Returns:
        Dictionary with split counts
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    np.random.seed(seed)

    source = Path(source_dir)
    output = Path(output_dir)

    # Find all images across splits
    all_images = []
    for split in ["train", "valid", "test", "val"]:
        img_dir = source / split / "images"
        if img_dir.exists():
            for ext in extensions:
                all_images.extend(sorted(img_dir.glob(f"*{ext}")))

    # Remove duplicates
    all_images = list(dict.fromkeys(all_images))

    # Match with labels
    image_label_pairs = []
    for img_path in all_images:
        label_path = None
        for split in ["train", "valid", "test", "val"]:
            candidate = source / split / "labels" / f"{img_path.stem}.txt"
            if candidate.exists():
                label_path = candidate
                break
        if label_path:
            image_label_pairs.append((img_path, label_path))

    # Split data
    train_pairs, temp_pairs = train_test_split(
        image_label_pairs, test_size=(val_ratio + test_ratio), random_state=seed, shuffle=True
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=seed, shuffle=True
    )

    # Create directories and copy files
    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }

    counts = {}
    for split_name, pairs in splits.items():
        (output / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output / split_name / "labels").mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            shutil.copy(img_path, output / split_name / "images" / img_path.name)
            shutil.copy(lbl_path, output / split_name / "labels" / lbl_path.name)

        counts[split_name] = len(pairs)

    return counts


def create_data_yaml(
    split_dir: str,
    class_names: List[str],
    output_path: str = "data.yaml"
) -> Path:
    """
    Create YOLO data.yaml configuration file.

    Args:
        split_dir: Directory containing train/val/test splits
        class_names: List of class names
        output_path: Output YAML file path

    Returns:
        Path to created YAML file
    """
    split_path = Path(split_dir)

    data_dict = {
        "path": str(split_path.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names
    }

    yaml_path = split_path / output_path
    with open(yaml_path, "w") as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    return yaml_path


def analyze_dataset(
    split_dir: str,
    class_names: List[str],
    split: str = "train"
) -> Dict:
    """
    Analyze dataset statistics.

    Args:
        split_dir: Data split directory
        class_names: List of class names
        split: Which split to analyze (train/val/test)

    Returns:
        Dictionary with statistics
    """
    split_path = Path(split_dir) / split
    label_dir = split_path / "labels"

    all_boxes = []
    class_counts = Counter()
    img_with_labels = 0
    img_without_labels = 0

    for lbl_path in sorted(label_dir.glob("*.txt")):
        boxes = parse_yolo_label(lbl_path)
        if boxes:
            img_with_labels += 1
            for b in boxes:
                all_boxes.append(b)
                class_counts[class_names[b[0]]] += 1
        else:
            img_without_labels += 1

    if all_boxes:
        widths = [b[3] for b in all_boxes]
        heights = [b[4] for b in all_boxes]
        areas = [w * h for w, h in zip(widths, heights)]

        stats = {
            "total_images": img_with_labels + img_without_labels,
            "annotated_images": img_with_labels,
            "empty_images": img_without_labels,
            "total_instances": len(all_boxes),
            "class_distribution": dict(class_counts),
            "bbox_width_mean": float(np.mean(widths)),
            "bbox_width_std": float(np.std(widths)),
            "bbox_height_mean": float(np.mean(heights)),
            "bbox_height_std": float(np.std(heights)),
            "bbox_area_mean": float(np.mean(areas)),
            "bbox_area_std": float(np.std(areas)),
        }
    else:
        stats = {
            "total_images": img_with_labels + img_without_labels,
            "annotated_images": img_with_labels,
            "empty_images": img_without_labels,
            "total_instances": 0,
            "class_distribution": {},
        }

    return stats
