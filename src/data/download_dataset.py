"""
Dataset download script from Roboflow.

Usage:
    python -m src.data.download_dataset --api-key YOUR_KEY
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from roboflow import Roboflow
from src.utils.config_loader import load_config

def download_dataset(api_key: str, output_dir: str = "data/raw"):
    """
    Download brain hemorrhage dataset from Roboflow.

    Args:
        api_key: Roboflow API key
        output_dir: Directory to save dataset
    """
    config = load_config("configs/config.yaml")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(config.dataset.roboflow_workspace).project(config.dataset.roboflow_project)
    dataset = project.version(config.dataset.version).download(config.dataset.format)

    print(f"[INFO] Dataset downloaded to: {dataset.location}")
    return dataset.location

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Roboflow")
    parser.add_argument("--api-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    download_dataset(args.api_key, args.output)

if __name__ == "__main__":
    main()
