import unittest
from unittest.mock import patch
import torch
import sys
import os
import torch.nn as nn

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.device import resolve_device, get_device_info

class TestDevice(unittest.TestCase):

    def test_invalid_device(self):
        with self.assertRaisesRegex(ValueError, "Invalid device specified"):
            resolve_device("unknown_device")

    def test_explicit_cpu(self):
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")
        self.assertIsInstance(device, torch.device)

    @patch('src.utils.device.torch.cuda.is_available', return_value=False)
    def test_auto_no_cuda(self, mock_is_available):
        device = resolve_device("auto")
        self.assertEqual(device.type, "cpu")

    @patch('src.utils.device.torch.cuda.is_available', return_value=True)
    def test_auto_with_cuda(self, mock_is_available):
        device = resolve_device("auto")
        self.assertEqual(device.type, "cuda")

    @patch('src.utils.device.torch.cuda.is_available', return_value=False)
    def test_explicit_cuda_fails_when_unavailable(self, mock_is_available):
        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            resolve_device("cuda")

    @patch('src.utils.device.torch.cuda.is_available', return_value=True)
    def test_explicit_cuda_succeeds_when_available(self, mock_is_available):
        device = resolve_device("cuda")
        self.assertEqual(device.type, "cuda")

    def test_real_device_resolution(self):
        """Tests the actual hardware without mocks."""
        device = resolve_device("auto")
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ["cpu", "cuda"])
        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
        else:
            self.assertEqual(device.type, "cpu")

    def test_tensor_and_training_step_on_device(self):
        """A tiny training step works using the resolved device."""
        device = resolve_device("auto")

        model = nn.Linear(10, 2).to(device)
        inputs = torch.randn(5, 10).to(device)
        targets = torch.randn(5, 2).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        # Forward pass
        outputs = model(inputs)
        self.assertEqual(outputs.device.type, device.type)

        # Backward pass
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Verify gradients exist
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertEqual(param.grad.device.type, device.type)

    def test_device_info_structure(self):
        """Check the structure of the info dictionary."""
        info = get_device_info()
        self.assertIn("cuda_available", info)
        self.assertIn("device_name", info)
        self.assertIn("device_type", info)

if __name__ == "__main__":
    unittest.main()
