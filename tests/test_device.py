import unittest
import torch
import sys
import os

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.device import get_device, get_device_info

class TestDevice(unittest.TestCase):
    def test_get_device_returns_valid_type(self):
        """1. & 2. The device utility returns a valid torch.device. Either CPU or CUDA."""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ["cpu", "cuda"])

    def test_device_selection_logic(self):
        """3. & 4. Verifies the fallback and selection logic."""
        device = get_device()
        info = get_device_info()
        
        if not info["cuda_available"]:
            self.assertEqual(device.type, "cpu")
            self.assertEqual(info["device_type"], "cpu")
        else:
            self.assertEqual(device.type, "cuda")
            self.assertEqual(info["device_type"], "cuda")

    def test_tensor_creation_on_device(self):
        """5. & 6. A small PyTorch tensor can be created on the selected device. No specific GPU assumptions."""
        device = get_device()
        try:
            tensor = torch.tensor([1.0, 2.0], device=device)
            self.assertEqual(tensor.device.type, device.type)
        except Exception as e:
            self.fail(f"Failed to create tensor on device {device}: {e}")
            
    def test_device_info_structure(self):
        """Check the structure of the info dictionary."""
        info = get_device_info()
        self.assertIn("cuda_available", info)
        self.assertIn("device_name", info)
        self.assertIn("device_type", info)

if __name__ == "__main__":
    unittest.main()
