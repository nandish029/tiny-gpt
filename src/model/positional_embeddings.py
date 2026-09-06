import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """
    A learned positional embedding layer for GPT-1.
    Provides positional context independent of the token IDs.
    """
    def __init__(self, max_context_length: int, embedding_dim: int):
        super().__init__()
        self.max_context_length = max_context_length
        self.embedding_dim = embedding_dim
        # Learned positional embeddings up to max_context_length
        self.embedding = nn.Embedding(
            num_embeddings=max_context_length,
            embedding_dim=embedding_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of token IDs with shape [B, T].
               B = batch size, T = sequence length.
        Returns:
            Tensor of positional embeddings with shape [B, T, embedding_dim].
        """
        B, T = x.shape
        if T > self.max_context_length:
            raise ValueError(
                f"Sequence length {T} exceeds maximum context length {self.max_context_length}"
            )
            
        # Create an integer tensor of positions [0, 1, 2, ..., T-1]
        positions = torch.arange(T, dtype=torch.long, device=x.device)
        
        # Broadcast positions to match batch size: shape becomes [B, T]
        positions = positions.unsqueeze(0).expand(B, T)
        
        # Look up positional embeddings: shape becomes [B, T, embedding_dim]
        return self.embedding(positions)
