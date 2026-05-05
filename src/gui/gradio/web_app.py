"""
Brain Hemorrhage Detection - Gradio Web Interface with GPU support
Modern browser-based UI for trained YOLOv8 models.

Usage:
    python -m src.gui.gradio.web_app
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import os

from src.utils.config_loader import load_config

# Cache models
_loaded_models = {}


def find_config():
    """Find config.yaml from multiple possible locations."""
    possible_paths = [
        "configs/config.yaml",
        Path(__file__).parent.parent.parent.parent / "configs" / "config.yaml",
        Path.cwd() / "configs" / "config.yaml",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return str(path)

    raise FileNotFoundError("config.yaml not found. Please run from project root.")


def get_device(config):
    """Determine the device to use for inference."""
    # Check if force_device is set
    force_device = None
    try:
        force_device = config.device.force_device
    except (AttributeError, KeyError):
        pass

    if force_device and force_device != "auto":
        return force_device

    # Auto-detect
    if torch.cuda.is_available():
        device = "0"
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("[WARNING] No GPU detected. Using CPU.")
        return "cpu"


def get_model(config, model_name):
    """Load and cache model from config paths."""
    if model_name not in _loaded_models:
        if model_name == "Baseline YOLOv8n":
            path = config.paths.baseline_model
        elif model_name == "Improved YOLOv8s":
            path = config.paths.improved_model
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}. Update config.yaml paths.")

        _loaded_models[model_name] = YOLO(path)
    return _loaded_models[model_name]


def parse_colors(config):
    """Parse hex colors from config to PIL-compatible format."""
    colors = {}
    color_config = config.visualization.class_colors

    # Handle both Config object and dict
    if hasattr(color_config, 'items'):
        items = color_config.items()
    elif isinstance(color_config, dict):
        items = color_config.items()
    else:
        # Fallback colors
        return {
            "EDH": "#FF0000",
            "IPH": "#00FF00",
            "IVH": "#0000FF",
            "SAH": "#FFFF00",
            "SDH": "#FF00FF",
        }

    for cls, hex_color in items:
        h = str(hex_color).lstrip('#')
        colors[cls] = f"#{h}"

    return colors


def detect_hemorrhage(input_image, model_name, conf_threshold, iou_threshold):
    """
    Run detection and return annotated image + markdown results.

    Args:
        input_image: PIL.Image from Gradio
        model_name: Selected model name
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
    Returns:
        (annotated_image, result_markdown)
    """
    if input_image is None:
        return None, "**Please upload a brain CT image first.**"

    config = load_config(find_config())
    device = get_device(config)
    model = get_model(config, model_name)
    class_colors = parse_colors(config)

    # Run inference with GPU
    results = model(
        input_image, 
        conf=conf_threshold, 
        iou=iou_threshold, 
        device=device,  # <-- GPU SETTING
        verbose=False
    )
    result = results[0]

    # Draw annotations
    draw_img = input_image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_name = result.names[cls_id]

            color = class_colors.get(cls_name, "#FFFFFF")

            # Draw bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            # Label
            label = f"{cls_name} {conf:.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - th - 6, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th - 3), label, fill="white", font=font)

            detections.append({
                "Class": cls_name,
                "Confidence": f"{conf:.4f}",
                "X1": f"{x1:.1f}", "Y1": f"{y1:.1f}",
                "X2": f"{x2:.1f}", "Y2": f"{y2:.1f}",
            })

    # Build markdown table
    if len(detections) == 0:
        result_md = "**No hemorrhage detected** above the selected confidence threshold."
    else:
        header = "| # | Class | Confidence | X1 | Y1 | X2 | Y2 |\n|---|---|---|---|---|---|---|---|\n"
        rows = ""
        for i, d in enumerate(detections, 1):
            rows += f"| {i} | {d['Class']} | {d['Confidence']} | {d['X1']} | {d['Y1']} | {d['X2']} | {d['Y2']} |\n"
        result_md = f"**Detected {len(detections)} hemorrhage region(s):**\n\n" + header + rows

    return draw_img, result_md


def create_interface():
    """Create and return Gradio interface."""
    config = load_config(find_config())
    device = get_device(config)

    device_status = f"🟢 GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "🔴 CPU Mode"

    with gr.Blocks(title=config.gui.gradio.title, theme=gr.themes.Soft()) as demo:

        gr.Markdown(f"""
        # 🧠 {config.gui.gradio.title}
        ### {config.gui.gradio.description}

        **Device Status: {device_status}**

        Upload a brain CT image and select a trained model to detect hemorrhage regions.
        Supports **{', '.join(config.dataset.classes)}** classification.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Input Settings")
                input_image = gr.Image(type="pil", label="Upload CT Image", image_mode="RGB")

                model_dropdown = gr.Dropdown(
                    choices=["Baseline YOLOv8n", "Improved YOLOv8s"],
                    value="Baseline YOLOv8n",
                    label="Select Model"
                )

                conf_slider = gr.Slider(
                    minimum=0.01, maximum=1.0, value=config.inference.default_conf, step=0.01,
                    label="Confidence Threshold"
                )

                iou_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=config.inference.default_iou, step=0.05,
                    label="NMS IoU Threshold"
                )

                with gr.Row():
                    detect_btn = gr.Button("🔍 Run Detection", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Detection Output")
                output_image = gr.Image(type="pil", label="Annotated Result")
                output_text = gr.Markdown(label="Detection Details")

        # Event handlers
        detect_btn.click(
            fn=detect_hemorrhage,
            inputs=[input_image, model_dropdown, conf_slider, iou_slider],
            outputs=[output_image, output_text]
        )

        clear_btn.click(
            fn=lambda: (None, "Upload an image and click **Run Detection**."),
            inputs=None,
            outputs=[output_image, output_text]
        )

        gr.Markdown("""
        ---
        **💡 Tip:** Lower the confidence threshold to catch subtle hemorrhages. 
        Use the **Improved YOLOv8s** model for potentially higher accuracy.
        """)

    return demo


def main():
    config = load_config(find_config())
    device = get_device(config)
    demo = create_interface()
    demo.launch(
        share=config.gui.gradio.share,
        server_port=config.gui.gradio.server_port
    )

if __name__ == "__main__":
    main()
