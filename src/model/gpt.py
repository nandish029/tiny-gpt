import torch
import torch.nn as nn
from src.model.gpt_embeddings import GPTInputEmbedding
from src.model.transformer_stack import TransformerStack
from src.model.final_normalization import FinalLayerNorm
from src.model.lm_head import LanguageModelHead

class GPT(nn.Module):
    """
    Complete GPT decoder-only language model.
    Assembles tested components into a single forward pass.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_context_length: int,
        num_heads: int,
        feed_forward_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_context_length = max_context_length
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_layers = num_layers

        self.embedding = GPTInputEmbedding(vocab_size, embedding_dim, max_context_length)
        self.transformer_stack = TransformerStack(embedding_dim, num_heads, feed_forward_dim, num_layers)
        self.final_norm = FinalLayerNorm(embedding_dim)
        self.lm_head = LanguageModelHead(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [B, T] containing integer token IDs.
        Returns:
            Raw logits of shape [B, T, V].
        """
        B, T = input_ids.shape
        if T > self.max_context_length:
            raise ValueError(
                f"Sequence length {T} exceeds max_context_length {self.max_context_length}"
            )

        x = self.embedding(input_ids)
        x = self.transformer_stack(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
