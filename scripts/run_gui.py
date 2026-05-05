"""
GUI launcher script.

Usage:
    python scripts/run_gui.py --type tkinter    # Desktop GUI
    python scripts/run_gui.py --type gradio    # Web GUI
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch GUI application")
    parser.add_argument("--type", type=str, choices=["tkinter", "gradio"], 
                       default="tkinter", help="GUI type to launch")
    args = parser.parse_args()

    if args.type == "tkinter":
        from src.gui.tkinter.gui_app import main
        print("[INFO] Launching Tkinter Desktop GUI...")
        main()
    elif args.type == "gradio":
        from src.gui.gradio.web_app import main
        print("[INFO] Launching Gradio Web Interface...")
        main()

if __name__ == "__main__":
    main()
