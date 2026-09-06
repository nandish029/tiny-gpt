import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network for GPT.
    Applies two linear transformations with a GELU activation in between.
    """
    def __init__(self, embedding_dim: int, feed_forward_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feed_forward_dim = feed_forward_dim
        
        self.linear1 = nn.Linear(embedding_dim, feed_forward_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(feed_forward_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, D]
        Returns:
            Output tensor of shape [B, T, D]
        """
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
