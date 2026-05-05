"""
Notebook 1: Data Exploration & Label Validation
This script can be run as a Jupyter notebook cell-by-cell.
"""

# Cell 1: Setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from src.utils.config_loader import load_config
from src.utils.dataset_utils import analyze_dataset, parse_yolo_label
from src.utils.visualization_utils import create_label_gallery, plot_class_distribution
import matplotlib.pyplot as plt

config = load_config("configs/config.yaml")

# Cell 2: Dataset Analysis
stats = analyze_dataset(
    split_dir=config.paths.data_splits,
    class_names=config.dataset.classes,
    split="train"
)

print("Dataset Statistics:")
for k, v in stats.items():
    print(f"  {k}: {v}")

# Cell 3: Class Distribution Plot
plot_class_distribution(
    class_counts=stats["class_distribution"],
    output_path=Path(config.paths.outputs_figures) / "class_distribution.png",
    title="Brain Hemorrhage Class Distribution"
)

# Cell 4: Label Validation Gallery
from pathlib import Path
import random

random.seed(config.visualization.label_gallery.num_samples)
train_images = list(Path(config.paths.data_splits / "train" / "images").glob("*.jpg"))
if not train_images:
    train_images = list(Path(config.paths.data_splits / "train" / "images").glob("*.png"))

sample = random.sample(train_images, min(config.visualization.label_gallery.num_samples, len(train_images)))

create_label_gallery(
    image_paths=sample,
    label_dir=Path(config.paths.data_splits) / "train" / "labels",
    class_names=config.dataset.classes,
    output_path=Path(config.paths.outputs_label_gallery) / "validation_gallery.png",
    num_samples=config.visualization.label_gallery.num_samples,
    grid_rows=config.visualization.label_gallery.grid_rows,
    grid_cols=config.visualization.label_gallery.grid_cols,
    figsize=tuple(config.visualization.label_gallery.figsize),
    dpi=config.visualization.label_gallery.dpi,
    seed=config.split.seed
)

print("Label gallery created!")
