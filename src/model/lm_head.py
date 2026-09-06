import torch
import torch.nn as nn

class LanguageModelHead(nn.Module):
    """
    Language model output head.
    Projects the final hidden representation to vocabulary logits.
    """
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, D]
        Returns:
            Raw logits of shape [B, T, V]
        """
        return self.linear(x)
