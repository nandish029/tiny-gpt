import torch
import torch.nn as nn
from typing import Tuple
import math

class QKVProjections(nn.Module):
    """
    Computes Q, K, and V linear projections for self-attention.
    Maintains three separate Linear layers to keep the implementation explicit.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [B, T, D] where D is embedding_dim.
        Returns:
            Tuple of (Q, K, V), each of shape [B, T, D].
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        return q, k, v

class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention without causal masking.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor of shape [..., Tq, D]
            k: Key tensor of shape [..., Tk, D]
            v: Value tensor of shape [..., Tk, D]
            
        Returns:
            output: Attention output of shape [..., Tq, D]
            attention_weights: Attention probabilities of shape [..., Tq, Tk]
        """
        D = q.size(-1)
        Tq = q.size(-2)
        Tk = k.size(-2)
        
        # scores = Q @ K^T / sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = scores / math.sqrt(D)
        
        if causal:
            # Prevent query pos i from attending to key pos j > i
            mask = torch.tril(torch.ones(Tq, Tk, dtype=torch.bool, device=q.device))
            scaled_scores = scaled_scores.masked_fill(~mask, float('-inf'))
            
        # attention_weights = softmax(scaled_scores, dim=-1)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # output = attention_weights @ V
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for GPT.
    """
    def __init__(self, embedding_dim: int, num_heads: int, causal: bool = True):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
            
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.causal = causal
        
        self.qkv_proj = QKVProjections(embedding_dim)
        self.attention = ScaledDotProductAttention()
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, D]
        Returns:
            Output tensor of shape [B, T, D]
        """
        B, T, D = x.shape
        
        # 1. Q, K, V projections
        q, k, v = self.qkv_proj(x)
        
        # 2. Reshape and transpose to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3 & 4. Scaled dot-product attention
        out, _ = self.attention(q, k, v, causal=self.causal)
        
        # 5. Concatenate heads back to [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        # 6. Final output projection
        out = self.out_proj(out)
        
        return out
