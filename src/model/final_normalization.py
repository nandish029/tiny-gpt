import torch.nn as nn
from src.model.normalization import LayerNorm, RMSNorm

class FinalLayerNorm(nn.Module):
    """
    Final normalization applied after the Transformer stack.
    Reuses LayerNorm by default or RMSNorm when norm_type='rmsnorm'.
    """
    def __init__(self, embedding_dim: int, norm_type: str = "layernorm"):
        super().__init__()
        self.ln = RMSNorm(embedding_dim) if norm_type == "rmsnorm" else LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, D]
        Returns:
            Normalized tensor of shape [B, T, D]
        """
        return self.ln(x)
