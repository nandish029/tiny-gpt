import torch
import torch.nn as nn
from src.model.embeddings import TokenEmbedding
from src.model.positional_embeddings import PositionalEmbedding

class GPTInputEmbedding(nn.Module):
    """
    Combined token and positional embedding layer for GPT.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_context_length: int,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_context_length = max_context_length
        self.pos_encoding = pos_encoding
        
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_embedding = (
            PositionalEmbedding(max_context_length, embedding_dim)
            if pos_encoding == "learned"
            else None
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of shape [B, T] containing integer token IDs.
        Returns:
            Combined embeddings of shape [B, T, D].
        """
        if token_ids.dim() != 2:
            raise ValueError(f"Expected token_ids to be 2D [B, T], got {token_ids.dim()}D")
            
        B, T = token_ids.shape
        if T > self.max_context_length:
            raise ValueError(f"Sequence length {T} exceeds max_context_length {self.max_context_length}")
            
        token_emb = self.token_embedding(token_ids)
        if self.positional_embedding is None:
            return token_emb
        pos_emb = self.positional_embedding(token_ids)
        
        return token_emb + pos_emb
