"""Perceiver-style encoder with learnable query tokens and cross-attention.

Each branch uses this encoder to compress variable-length modality inputs
into a fixed number of output tokens whose dimension equals ``d_model``.
"""

import torch
from torch import nn


class PerceiverEncoder(nn.Module):
    """Per-modality perceiver encoder.

    Projects raw modality input to *d_model*, then uses learnable query
    tokens to cross-attend over the projected input.  The output has a
    fixed sequence length equal to *num_queries* regardless of input length.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_queries: int,
        num_heads: int,
        num_input_tokens: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if num_input_tokens <= 0:
            raise ValueError(f"num_input_tokens must be positive, got {num_input_tokens}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.input_proj = nn.Linear(input_dim, d_model)
        self.num_input_tokens = num_input_tokens
        self.num_layers = num_layers
        self.single_frame_tokenizer = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_input_tokens * d_model),
        )
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_queries, d_model) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_cross_q": nn.LayerNorm(d_model, eps=layer_norm_eps),
                        "norm_cross_kv": nn.LayerNorm(d_model, eps=layer_norm_eps),
                        "cross_attn": nn.MultiheadAttention(
                            d_model,
                            num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm_self": nn.LayerNorm(d_model, eps=layer_norm_eps),
                        "self_attn": nn.MultiheadAttention(
                            d_model,
                            num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm_ffn": nn.LayerNorm(d_model, eps=layer_norm_eps),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 4, d_model),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Modality input — ``(B, T, input_dim)`` or ``(B, input_dim)``.

        Returns:
            ``(B, num_queries, d_model)``
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, input_dim)

        # Align dtype / device to the projection layer.
        x = x.to(device=self.input_proj.weight.device, dtype=self.input_proj.weight.dtype)

        if x.shape[1] == 1:
            kv = self.single_frame_tokenizer(x[:, 0])
            kv = kv.view(x.shape[0], self.num_input_tokens, self.input_proj.out_features)
        else:
            kv = self.input_proj(x)  # (B, T, d_model)

        B = kv.shape[0]
        latents = self.latent_queries.expand(B, -1, -1)  # (B, num_queries, d_model)

        for block in self.blocks:
            kv_normed = block["norm_cross_kv"](kv)
            q = block["norm_cross_q"](latents)
            attn_out, _ = block["cross_attn"](q, kv_normed, kv_normed, need_weights=False)
            latents = latents + attn_out

            latent_normed = block["norm_self"](latents)
            self_attn_out, _ = block["self_attn"](
                latent_normed,
                latent_normed,
                latent_normed,
                need_weights=False,
            )
            latents = latents + self_attn_out

            latents = latents + block["ffn"](block["norm_ffn"](latents))

        return latents
