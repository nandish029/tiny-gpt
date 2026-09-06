import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    A simple token embedding layer for GPT-1.
    Converts integer token IDs into dense vector representations.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Using standard PyTorch Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of token IDs with shape [B, T].
               B = batch size, T = context length.
        Returns:
            Tensor of embedded tokens with shape [B, T, embedding_dim].
        """
        return self.embedding(x)
