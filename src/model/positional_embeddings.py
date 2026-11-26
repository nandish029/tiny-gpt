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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    """Apply rotary position embeddings to Q and K of shape [B, H, T, head_dim]."""
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embedding (RoPE) cosines and sines."""

    def __init__(self, head_dim: int, max_context_length: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_context_length, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        cos = self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin
