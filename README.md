# 🧠 Brain Hemorrhage Detection - YOLOv8 Project

**BM 480 Deep Learning Course - Project 3**

A complete, production-ready project for brain hemorrhage detection using YOLOv8 on the Roboflow dataset (~355 CT images, 5 classes: EDH, IPH, IVH, SAH, SDH).

---

## 📁 Project Structure

```
BrainHemorrhage_YOLOv8_Project/
├── configs/
│   └── config.yaml              # Central configuration file
├── data/
│   ├── raw/                     # Downloaded dataset
│   ├── processed/               # Processed data
│   └── splits/                  # Custom 80/10/10 split
├── docs/                        # Documentation
├── notebooks/
│   ├── 01_data_exploration.py   # Data analysis & label gallery
│   ├── 02_training.py           # Model training
│   └── 03_evaluation.py         # Evaluation & ablation
├── outputs/
│   ├── models/                  # Trained models
│   ├── figures/                 # Plots & graphs
│   ├── label_gallery/           # Label validation images
│   ├── error_analysis/          # FP/FN analysis
│   ├── ablation/                # Ablation study results
│   └── reports/                 # Final reports
├── scripts/
│   ├── run_pipeline.py          # Full pipeline runner
│   └── run_gui.py               # GUI launcher
├── src/
│   ├── data/
│   │   ├── download_dataset.py  # Roboflow download
│   │   └── prepare_dataset.py   # Split & data.yaml creation
│   ├── gui/
│   │   ├── tkinter/
│   │   │   └── gui_app.py       # Desktop GUI
│   │   └── gradio/
│   │       └── web_app.py       # Web GUI
│   ├── models/
│   │   ├── train_baseline.py    # Baseline training
│   │   ├── train_ablation_a.py  # Ablation A training
│   │   └── evaluate.py          # Model evaluation
│   └── utils/
│       ├── config_loader.py     # YAML config manager
│       ├── dataset_utils.py     # Data utilities
│       └── visualization_utils.py  # Plotting utilities
├── requirements.txt             # Base dependencies
├── requirements-gui.txt         # GUI dependencies
└── requirements-dev.txt         # Development dependencies
```

---

## 🚀 Quick Start

### 1. Clone/Setup Project
```bash
cd BrainHemorrhage_YOLOv8_Project
```

### 2. Install Dependencies
```bash
# Base installation (training + evaluation)
pip install -r requirements.txt

# With GUI support
pip install -r requirements-gui.txt

# Development (testing, notebooks)
pip install -r requirements-dev.txt
```

### 3. Configure Paths
Edit `configs/config.yaml` and update the model paths to your actual `.pt` files:
```yaml
paths:
  baseline_model: "C:/Users/erhad/.../baseline_yolov8n_best.pt"
  improved_model: "C:/Users/erhad/.../improved_yolov8s_best.pt"
```

### 4. Run Full Pipeline
```bash
# Download dataset, prepare, train, evaluate
python scripts/run_pipeline.py --api-key zyV3K3gkGtguGRLG9Uvw
```

### 5. Launch GUI
```bash
# Desktop GUI (offline)
python scripts/run_gui.py --type tkinter

# Web GUI (browser)
python scripts/run_gui.py --type gradio
```

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Central Config** | All paths & hyperparameters in `config.yaml` |
| **Modular Code** | Clean separation: data, models, utils, GUI |
| **Two GUIs** | Tkinter (desktop) + Gradio (web) |
| **Reproducible** | Fixed seed (42) for all experiments |
| **Ablation Ready** | Image size & threshold ablation scripts |
| **Error Analysis** | FP/FN visualization tools |

---

## 📊 Model Comparison

| Model | Image Size | Speed | Accuracy |
|-------|-----------|-------|----------|
| Baseline YOLOv8n | 640px | Fast | Standard |
| Improved YOLOv8s | 640px | Medium | Higher |

---

## 📝 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Windows/Linux/macOS

---

## 📚 Citation

BM 480 Deep Learning Course, 2025-2026 Bahar Dönemi.
