"""
Quick test to verify Config class and color parsing works correctly.
Run this before launching the GUI to ensure everything is fixed.

Usage:
    python test_config.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config

def test_config():
    print("=" * 60)
    print("TESTING CONFIG LOADER")
    print("=" * 60)

    try:
        config = load_config("configs/config.yaml")
        print("✅ Config loaded successfully")

        # Test basic access
        print(f"  Project name: {config.project.name}")
        print(f"  Dataset classes: {config.dataset.classes}")

        # Test nested dict access (this was the bug)
        viz = config.visualization
        print(f"  Visualization bbox_thickness: {viz.bbox_thickness}")

        # Test .items() on nested config (THE FIX)
        colors = viz.class_colors
        print(f"  Color config type: {type(colors)}")

        if hasattr(colors, 'items'):
            print("✅ colors.items() is available")
            for cls, color in colors.items():
                print(f"    {cls}: {color}")
        else:
            print("❌ colors.items() NOT available - FIX NOT APPLIED")
            return False

        # Test model paths
        print(f"\n  Baseline model path: {config.paths.baseline_model}")
        print(f"  Improved model path: {config.paths.improved_model}")

        print("\n✅ ALL TESTS PASSED - GUI should work now!")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
