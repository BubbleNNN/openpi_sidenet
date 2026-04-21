"""Text-conditioned cross-attention for producing modality-fused hidden states.

A set of learnable fusion vectors is combined with mean-pooled (projected)
text embeddings to form the query.  The keys / values come from the
concatenated-and-self-attended modality tokens produced upstream.
"""

import torch
from torch import nn


class TextConditionedCrossAttention(nn.Module):
    """Cross-attention whose query is a learnable vector + projected text."""

    def __init__(
        self,
        d_model: int,
        num_queries: int,
        text_embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.learnable_vectors = nn.Parameter(
            torch.zeros(1, num_queries, d_model)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_kv = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        modality_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            modality_tokens: ``(B, N_mod, d_model)`` from concat + self-attn.
            text_embeddings: ``(B, L, text_embed_dim)`` raw language embeddings.
            text_mask: ``(B, L)`` bool — True for valid tokens.

        Returns:
            ``(B, num_queries, d_model)``
        """
        B = modality_tokens.shape[0]

        # Project text tokens and mean-pool (masked).
        text_proj = self.text_projector(text_embeddings)  # (B, L, d_model)
        if text_mask is not None:
            mask_f = text_mask.unsqueeze(-1).to(dtype=text_proj.dtype)
            text_pooled = (text_proj * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            text_pooled = text_proj.mean(dim=1)
        # text_pooled: (B, d_model)

        # Query = learnable vectors + pooled text
        queries = self.learnable_vectors.expand(B, -1, -1) + text_pooled.unsqueeze(1)

        # Cross-attention
        q = self.norm_q(queries)
        kv = self.norm_kv(modality_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        out = queries + attn_out

        # FFN
        out = out + self.ffn(self.norm_ffn(out))
        return out
