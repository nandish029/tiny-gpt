import math
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
        norm_type: str = "layernorm",
        ffn_type: str = "gelu",
        pos_encoding: str = "learned",
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_context_length = max_context_length
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.pos_encoding = pos_encoding
        self.tie_weights = tie_weights

        self.embedding = GPTInputEmbedding(
            vocab_size, embedding_dim, max_context_length, pos_encoding=pos_encoding
        )
        self.transformer_stack = TransformerStack(
            embedding_dim,
            num_heads,
            feed_forward_dim,
            num_layers,
            norm_type=norm_type,
            ffn_type=ffn_type,
            pos_encoding=pos_encoding,
            max_context_length=max_context_length,
        )
        self.final_norm = FinalLayerNorm(embedding_dim, norm_type=norm_type)
        self.lm_head = LanguageModelHead(embedding_dim, vocab_size, bias=not tie_weights)
        if tie_weights:
            self.apply(self._init_tied_weights)
            # Re-scale residual output projections (GPT-2 / Llama style)
            res_std = 0.02 / math.sqrt(2 * num_layers)
            for name, p in self.named_parameters():
                if name.endswith(("out_proj.weight", "w2.weight", "linear2.weight")):
                    nn.init.normal_(p, mean=0.0, std=res_std)
            self.lm_head.linear.weight = self.embedding.token_embedding.embedding.weight

    @staticmethod
    def _init_tied_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
