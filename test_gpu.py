"""
GPU Test Script for Brain Hemorrhage Detection Project
Run this to verify your GPU is properly configured.

Usage:
    python test_gpu.py
"""

import torch
import sys

def test_gpu():
    print("="*60)
    print("GPU TEST FOR BRAIN HEMORRHAGE DETECTION")
    print("="*60)

    # Check PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # GPU Info
        gpu_count = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {gpu_count}")

        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

        # Test tensor operations on GPU
        print("\n" + "="*60)
        print("RUNNING GPU TENSOR TEST")
        print("="*60)

        try:
            # Create tensor on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()

            # Perform operation
            z = torch.matmul(x, y)

            print(f"✅ Tensor created on GPU: {x.device}")
            print(f"✅ Matrix multiplication successful: {z.shape}")
            print(f"✅ GPU is working correctly!")

            # Test with Ultralytics YOLO
            print("\n" + "="*60)
            print("CHECKING ULTRALYTICS GPU SUPPORT")
            print("="*60)

            try:
                from ultralytics import YOLO
                model = YOLO("yolov8n.pt")
                print(f"✅ YOLO model loaded successfully")
                print(f"✅ Model will use GPU for training/inference")
            except Exception as e:
                print(f"⚠️ YOLO test failed: {e}")

            return True

        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False
    else:
        print("\n" + "="*60)
        print("❌ NO GPU DETECTED")
        print("="*60)
        print("\nTo use GPU, install CUDA-enabled PyTorch:")
        print("  For CUDA 11.8:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  For CUDA 12.1:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("\nOr use CPU (slower):")
        print("    Set device: 'cpu' in configs/config.yaml")

        return False

if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)
