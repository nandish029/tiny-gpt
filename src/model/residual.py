import torch
import torch.nn as nn

class ResidualAdd(nn.Module):
    """
    A small component for residual addition.
    Performs element-wise addition of x and sublayer_output.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Original input tensor [..., D]
            sublayer_output: Processed tensor [..., D]
            
        Returns:
            Added tensor of the exact same shape
        """
        if x.shape != sublayer_output.shape:
            raise ValueError(f"Shape mismatch: x shape {x.shape} != sublayer_output shape {sublayer_output.shape}")
            
        return x + sublayer_output
