import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.device import get_device, get_device_info
import torch

def main():
    info = get_device_info()
    device = get_device()
    
    print("=== Manual Device Check ===")
    print(f"Detected Device Type: {device.type}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"Device Name: {info['device_name']}")
    
    print("\nAttempting to create a small tensor on the device...")
    t = torch.ones(3, 3, device=device)
    print(f"Tensor created successfully! Tensor device: {t.device}")
    print("===========================")
    
if __name__ == "__main__":
    main()
