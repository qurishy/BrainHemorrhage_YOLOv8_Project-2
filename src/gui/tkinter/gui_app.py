"""
Brain Hemorrhage Detection - Desktop GUI (Tkinter) with GPU support
Professional offline interface for trained YOLOv8 models.

Usage:
    python -m src.gui.tkinter.gui_app
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import os

from src.utils.config_loader import load_config


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


class HemorrhageDetectorGUI:
    """Professional desktop GUI for brain hemorrhage detection."""

    def __init__(self, root):
        # Load config with robust path finding
        config_path = find_config()
        self.config = load_config(config_path)
        self.device = get_device(self.config)

        self.root = root
        self.root.title("Brain Hemorrhage Detection - YOLOv8")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # State
        self.current_model = None
        self.model = None
        self.original_image = None
        self.annotated_image = None
        self.current_image_path = None
        self.detections = []
        self.class_colors = self._parse_colors()

        self._build_ui()
        self._load_default_model()

    def _parse_colors(self):
        """Parse hex colors from config to RGB tuples."""
        colors = {}
        color_config = self.config.visualization.class_colors

        # Handle both Config object and dict
        if hasattr(color_config, 'items'):
            items = color_config.items()
        elif isinstance(color_config, dict):
            items = color_config.items()
        else:
            # Fallback colors
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

    def _build_ui(self):
        # Top Control Bar
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        ttk.Label(control_frame, text="Model:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar(value="Baseline YOLOv8n")
        self.model_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.model_var, 
            values=["Baseline YOLOv8n", "Improved YOLOv8s"],
            state="readonly", 
            width=25
        )
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Button(control_frame, text="Load Custom", command=self._browse_model).pack(side=tk.LEFT, padx=5)
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(control_frame, text="Select Image", command=self._browse_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Detect", command=self._run_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Result", command=self._save_result).pack(side=tk.LEFT, padx=5)

        # Threshold controls
        ttk.Label(control_frame, text="Conf:").pack(side=tk.LEFT, padx=(20, 5))
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(control_frame, from_=0.01, to=1.0, increment=0.01, 
                   textvariable=self.conf_var, width=6).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="IoU:").pack(side=tk.LEFT, padx=(10, 5))
        self.iou_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(control_frame, from_=0.1, to=0.9, increment=0.05,
                   textvariable=self.iou_var, width=6).pack(side=tk.LEFT)

        # Device indicator
        device_text = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU Mode"
        self.device_label = ttk.Label(control_frame, text=device_text, foreground="green" if torch.cuda.is_available() else "red")
        self.device_label.pack(side=tk.RIGHT, padx=10)

        # Main Content
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image Canvas
        img_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding=5)
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(img_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right Panel
        right_frame = ttk.Frame(content_frame, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # Model Info
        info_frame = ttk.LabelFrame(right_frame, text="Model Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, state=tk.DISABLED, 
                                bg="#fafafa", font=("Consolas", 9))
        self.info_text.pack(fill=tk.X)

        # Detection Results
        result_frame = ttk.LabelFrame(right_frame, text="Detection Results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, 
                                                      font=("Consolas", 10), bg="#fafafa")
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready. Please select a model and image.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _log_info(self, text):
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.configure(state=tk.DISABLED)

    def _log_result(self, text):
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.see(tk.END)

    def _clear_results(self):
        self.result_text.delete(1.0, tk.END)

    def _get_model_path(self, model_name):
        """Get model path from config."""
        if model_name == "Baseline YOLOv8n":
            return self.config.paths.baseline_model
        elif model_name == "Improved YOLOv8s":
            return self.config.paths.improved_model
        return None

    def _load_default_model(self):
        path = self._get_model_path(self.model_var.get())
        if path and os.path.exists(path):
            self._load_model(path)
        else:
            self._log_info(f"Model not found at:\n{path}\n\nPlease update config.yaml paths or browse for model.")

    def _on_model_change(self, event=None):
        path = self._get_model_path(self.model_var.get())
        if path and os.path.exists(path):
            self._load_model(path)
        else:
            messagebox.showwarning("Model Not Found", f"Update paths in config.yaml:\n{path}")

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO Model", filetypes=[("PyTorch", "*.pt"), ("All", "*.*")]
        )
        if path:
            self._load_model(path)

    def _load_model(self, path):
        try:
            self.status_var.set(f"Loading {Path(path).name}...")
            self.root.update()
            self.model = YOLO(path)
            self.current_model = path
            device_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
            self._log_info(f"Model: {Path(path).stem}\nPath: {path}\nDevice: {device_info}\nStatus: Loaded\n\nReady for inference.")
            self.status_var.set(f"Loaded: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_var.set("Model loading failed.")

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All", "*.*")]
        )
        if path:
            self.current_image_path = path
            self.original_image = Image.open(path).convert("RGB")
            self.annotated_image = None
            self._clear_results()
            self._show_image(self.original_image)
            self.status_var.set(f"Image: {Path(path).name}")

    def _show_image(self, pil_img):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10:
            canvas_w, canvas_h = 800, 600

        img_w, img_h = pil_img.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        x_off = (canvas_w - new_w) // 2
        y_off = (canvas_h - new_h) // 2
        self.canvas.create_image(x_off, y_off, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image

    def _run_detection(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        if self.original_image is None:
            messagebox.showwarning("No Image", "Select an image first.")
            return

        try:
            self.status_var.set("Running inference...")
            self.root.update()
            self._clear_results()

            conf = self.conf_var.get()
            iou = self.iou_var.get()

            # Run inference with GPU
            results = self.model(
                self.current_image_path, 
                conf=conf, 
                iou=iou, 
                device=self.device,  # <-- GPU SETTING
                verbose=False
            )
            result = results[0]

            draw_img = self.original_image.copy()
            draw = ImageDraw.Draw(draw_img)

            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            self.detections = []
            count = 0

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf_val = float(box.conf[0].cpu().numpy())
                    cls_name = result.names[cls_id]

                    color = self.class_colors.get(cls_name, (255, 255, 255))

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    label = f"{cls_name} {conf_val:.2f}"
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.rectangle([x1, y1 - th - 4, x1 + tw, y1], fill=color)
                    draw.text((x1, max(0, y1 - th - 2)), label, fill=(255, 255, 255), font=font)

                    self.detections.append({
                        "class": cls_name, "confidence": conf_val,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                    count += 1

            self.annotated_image = draw_img
            self._show_image(draw_img)

            self._log_result(f"=== Detection Results ===")
            self._log_result(f"Image: {Path(self.current_image_path).name}")
            self._log_result(f"Model: {Path(self.current_model).stem}")
            self._log_result(f"Device: {self.device}")
            self._log_result(f"Conf: {conf} | IoU: {iou}")
            self._log_result(f"Detections: {count}\n")

            if count == 0:
                self._log_result("No hemorrhage detected.")
            else:
                for i, det in enumerate(self.detections, 1):
                    self._log_result(
                        f"{i}. {det['class']} | Conf: {det['confidence']:.4f} | "
                        f"Box: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]"
                    )

            self.status_var.set(f"Done. {count} object(s) found.")

        except Exception as e:
            messagebox.showerror("Inference Error", str(e))
            self.status_var.set("Detection failed.")

    def _save_result(self):
        if self.annotated_image is None:
            messagebox.showwarning("No Result", "Run detection first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
            initialfile=f"result_{Path(self.current_image_path).stem}.png"
        )
        if path:
            self.annotated_image.save(path)
            self.status_var.set(f"Saved: {path}")
            messagebox.showinfo("Saved", f"Result saved to:\n{path}")


def main():
    root = tk.Tk()
    app = HemorrhageDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
