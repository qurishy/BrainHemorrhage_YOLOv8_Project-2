"""
Visualization utilities for brain hemorrhage detection.
Handles drawing bounding boxes, creating galleries, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd


def _get_class_colors(config_colors):
    """Helper to parse class colors from config."""
    colors = {}
    if hasattr(config_colors, 'items'):
        items = config_colors.items()
    elif isinstance(config_colors, dict):
        items = config_colors.items()
    else:
        # Fallback
        return {
            "EDH": (255, 0, 0),
            "IPH": (0, 255, 0),
            "IVH": (0, 0, 255),
            "SAH": (255, 255, 0),
            "SDH": (255, 0, 255),
        }

    for cls, hex_color in items:
        h = str(hex_color).lstrip('#')
        if len(h) == 6:
            colors[cls] = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        else:
            colors[cls] = (255, 255, 255)

    return colors


def draw_yolo_boxes(
    image_path: Path,
    label_path: Path,
    class_names: List[str],
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    bbox_thickness: int = 3,
    font_size: int = 16
) -> Image.Image:
    """
    Draw YOLO bounding boxes on an image.

    Args:
        image_path: Path to image file
        label_path: Path to YOLO label file
        class_names: List of class names
        class_colors: Dictionary mapping class names to RGB tuples
        bbox_thickness: Thickness of bounding box lines
        font_size: Font size for labels

    Returns:
        PIL Image with drawn boxes
    """
    from .dataset_utils import parse_yolo_label

    if class_colors is None:
        class_colors = {
            "EDH": (255, 0, 0),
            "IPH": (0, 255, 0),
            "IVH": (0, 0, 255),
            "SAH": (255, 255, 0),
            "SDH": (255, 0, 255),
        }

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    boxes = parse_yolo_label(label_path)

    for cls_id, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw/2) * w_img)
        y1 = int((cy - bh/2) * h_img)
        x2 = int((cx + bw/2) * w_img)
        y2 = int((cy + bh/2) * h_img)

        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        color = class_colors.get(cls_name, (255, 255, 255))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=bbox_thickness)

        # Draw label background
        label = f"{cls_name}"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w, y1], fill=color)
        draw.text((x1, max(0, y1 - text_h - 2)), label, fill=(255, 255, 255), font=font)

    return img


def create_label_gallery(
    image_paths: List[Path],
    label_dir: Path,
    class_names: List[str],
    output_path: Path,
    num_samples: int = 20,
    grid_rows: int = 4,
    grid_cols: int = 5,
    figsize: Tuple[int, int] = (20, 16),
    dpi: int = 150,
    seed: int = 42
) -> Path:
    """
    Create label validation gallery grid.

    Args:
        image_paths: List of image paths
        label_dir: Directory containing label files
        class_names: List of class names
        output_path: Path to save gallery image
        num_samples: Number of samples to display
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid
        figsize: Figure size (width, height)
        dpi: Resolution
        seed: Random seed

    Returns:
        Path to saved gallery
    """
    import random
    random.seed(seed)

    sample = random.sample(image_paths, min(num_samples, len(image_paths)))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, img_path in enumerate(sample):
        lbl_path = label_dir / f"{img_path.stem}.txt"
        img_with_boxes = draw_yolo_boxes(img_path, lbl_path, class_names)
        axes[idx].imshow(img_with_boxes)
        axes[idx].set_title(img_path.name, fontsize=9)
        axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(len(sample), grid_rows * grid_cols):
        axes[idx].axis("off")

    plt.suptitle("Label Validation Gallery", fontsize=16)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: Path,
    title: str = "Class Distribution"
) -> Path:
    """
    Plot class distribution bar chart.

    Args:
        class_counts: Dictionary mapping class names to counts
        output_path: Path to save plot
        title: Plot title

    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(8, 5))
    classes, counts = zip(*sorted(class_counts.items()))
    plt.bar(classes, counts, color="steelblue")
    plt.title(title)
    plt.xlabel("Hemorrhage Type")
    plt.ylabel("Instance Count")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_training_curves(
    results_csv_path: Path,
    output_dir: Path
) -> List[Path]:
    """
    Plot training curves from results.csv.

    Args:
        results_csv_path: Path to Ultralytics results.csv
        output_dir: Directory to save plots

    Returns:
        List of saved plot paths
    """
    df = pd.read_csv(results_csv_path)
    df.columns = [c.strip() for c in df.columns]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(df["epoch"], df["train/box_loss"], label="Box Loss")
    axes[0,0].plot(df["epoch"], df["train/cls_loss"], label="Cls Loss")
    axes[0,0].plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
    axes[0,0].set_title("Training Losses")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
    axes[0,1].plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss")
    axes[0,1].plot(df["epoch"], df["val/dfl_loss"], label="Val DFL Loss")
    axes[0,1].set_title("Validation Losses")
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].set_ylabel("Loss")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", color="green")
    axes[1,0].set_title("mAP@0.5 over Epochs")
    axes[1,0].set_xlabel("Epoch")
    axes[1,0].set_ylabel("mAP@0.5")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="blue")
    axes[1,1].set_title("mAP@0.5:0.95 over Epochs")
    axes[1,1].set_xlabel("Epoch")
    axes[1,1].set_ylabel("mAP@0.5:0.95")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_paths.append(path)

    return saved_paths


def plot_confidence_sweep(
    results: List[Dict],
    output_path: Path
) -> Path:
    """
    Plot confidence threshold sweep results.

    Args:
        results: List of dictionaries with keys: Confidence, Precision, Recall, F1
        output_path: Path to save plot

    Returns:
        Path to saved plot
    """
    df = pd.DataFrame(results)
    best = df.loc[df["F1"].idxmax()]

    plt.figure(figsize=(10, 6))
    plt.plot(df["Confidence"], df["Precision"], "o-", label="Precision")
    plt.plot(df["Confidence"], df["Recall"], "s-", label="Recall")
    plt.plot(df["Confidence"], df["F1"], "^-", label="F1", linewidth=2)
    plt.axvline(best["Confidence"], color="red", linestyle="--", 
                label=f"Best F1 (conf={best['Confidence']:.3f})")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title("Confidence Threshold Sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path
