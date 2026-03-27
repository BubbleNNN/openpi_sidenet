from collections.abc import Sequence

import torch
from torch import nn


class CrossAttnFusion(nn.Module):
    """Fuse one or more modality token groups into external fusion latents.

    This module does not own the learnable fusion latents. The caller passes in
    the current fusion vectors and one or more modality token tensors. Fusion is
    performed by sequentially writing each modality into the shared latents:

        fusion <- fusion + gate_i * CrossAttn(LN(fusion), LN(modality_i))
        fusion <- fusion + MLP(LN(fusion))
    """

    def __init__(self, in_features, num_heads, dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            in_features,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(in_features, eps=layer_norm_eps)
        self.norm_kv = nn.LayerNorm(in_features, eps=layer_norm_eps)
        self.norm_ffn = nn.LayerNorm(in_features, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features * 4, in_features),
        )
        self.alpha_ffn = nn.Parameter(torch.zeros(1))

    def _fuse_once(self, fusion_vectors, modality_vectors, modality_mask=None, gate=None):
        if modality_vectors.ndim != 3:
            raise ValueError(
                "modality_vectors must have shape (batch_size, num_tokens, features)"
            )
        if fusion_vectors.ndim != 3:
            raise ValueError(
                "fusion_vectors must have shape (batch_size, num_fusion_tokens, features)"
            )
        if modality_vectors.shape[0] != fusion_vectors.shape[0]:
            raise ValueError("batch sizes of fusion_vectors and modality_vectors must match")
        if modality_vectors.shape[-1] != fusion_vectors.shape[-1]:
            raise ValueError("feature dims of fusion_vectors and modality_vectors must match")

        query = self.norm_q(fusion_vectors)
        memory = self.norm_kv(modality_vectors)
        attn_output, _ = self.attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=modality_mask,
            need_weights=False,
        )
        if gate is not None:
            if not torch.is_tensor(gate):
                gate = torch.tensor(gate, dtype=attn_output.dtype, device=attn_output.device)
            gate = gate.to(dtype=attn_output.dtype, device=attn_output.device)
            while gate.ndim < attn_output.ndim:
                gate = gate.unsqueeze(-1)
            attn_output = attn_output * gate
        # The attention branch is controlled by modality-level gates passed from
        # outside the module. The FFN branch keeps a trainable zero-init gate so
        # that this residual path can be opened gradually during training.
        fusion_vectors = fusion_vectors + attn_output
        fusion_vectors = fusion_vectors + self.alpha_ffn * self.mlp(self.norm_ffn(fusion_vectors))
        return fusion_vectors

    def forward(self, fusion_vectors, modality_vectors, modality_masks=None, gates=None):
        """Fuse one or multiple modality memories into shared fusion latents.

        Args:
            fusion_vectors: Tensor with shape (B, N_fusion, D).
            modality_vectors: Tensor with shape (B, N_mod, D) or a sequence of such tensors.
            modality_masks: Optional bool Tensor with shape (B, N_mod), or a sequence
                aligned with ``modality_vectors``. True entries are treated as padding.
            gates: Optional scalar/tensor gate, or a sequence aligned with
                ``modality_vectors``. Typical shapes are scalar, (B,), or (B, 1).

        Returns:
            Tensor with shape (B, N_fusion, D).
        """
        if torch.is_tensor(modality_vectors):
            return self._fuse_once(
                fusion_vectors,
                modality_vectors,
                modality_mask=modality_masks,
                gate=gates,
            )

        if not isinstance(modality_vectors, Sequence):
            raise TypeError(
                "modality_vectors must be a Tensor or a sequence of Tensors"
            )

        num_modalities = len(modality_vectors)
        if modality_masks is None:
            modality_masks = [None] * num_modalities
        elif torch.is_tensor(modality_masks):
            raise TypeError(
                "modality_masks must be a sequence when modality_vectors is a sequence"
            )
        if gates is None:
            gates = [None] * num_modalities
        elif torch.is_tensor(gates) or not isinstance(gates, Sequence):
            gates = [gates] * num_modalities

        if len(modality_masks) != num_modalities:
            raise ValueError("modality_masks must align with modality_vectors")
        if len(gates) != num_modalities:
            raise ValueError("gates must align with modality_vectors")
        # TODO: check the sequential bias
        # cannot be paralleled due to the need for a shared fusion vector across modalities
        # Here introduced the sequential bias, i.e f&t->tactile and tactile->f&t may get different outputs
        # not sure if this is desirable. Maybe in practice some modalities are more important than others
        for curr_vectors, curr_mask, curr_gate in zip(modality_vectors, modality_masks, gates):
            fusion_vectors = self._fuse_once(
                fusion_vectors,
                curr_vectors,
                modality_mask=curr_mask,
                gate=curr_gate,
            )

        # Alternative fusion without sequential bias:
        # Use the same initial fusion vectors as query for every modality, compute
        # one delta per modality independently, then sum all deltas at the end.
        # This removes the order dependence introduced by iterative updates above.
        #
        # Uncomment this block and comment out the sequential loop if you want to
        # switch to the permutation-friendlier version.
        #
        # initial_fusion_vectors = fusion_vectors
        # total_delta = torch.zeros_like(initial_fusion_vectors)
        #
        # for curr_vectors, curr_mask, curr_gate in zip(modality_vectors, modality_masks, gates):
        #     if curr_vectors.ndim != 3:
        #         raise ValueError(
        #             "modality_vectors must have shape (batch_size, num_tokens, features)"
        #         )
        #     if curr_vectors.shape[0] != initial_fusion_vectors.shape[0]:
        #         raise ValueError("batch sizes of fusion_vectors and modality_vectors must match")
        #     if curr_vectors.shape[-1] != initial_fusion_vectors.shape[-1]:
        #         raise ValueError("feature dims of fusion_vectors and modality_vectors must match")
        #
        #     query = self.norm_q(initial_fusion_vectors)
        #     memory = self.norm_kv(curr_vectors)
        #     curr_delta, _ = self.attn(
        #         query=query,
        #         key=memory,
        #         value=memory,
        #         key_padding_mask=curr_mask,
        #         need_weights=False,
        #     )
        #
        #     if curr_gate is not None:
        #         if not torch.is_tensor(curr_gate):
        #             curr_gate = torch.tensor(
        #                 curr_gate,
        #                 dtype=curr_delta.dtype,
        #                 device=curr_delta.device,
        #             )
        #         curr_gate = curr_gate.to(dtype=curr_delta.dtype, device=curr_delta.device)
        #         while curr_gate.ndim < curr_delta.ndim:
        #             curr_gate = curr_gate.unsqueeze(-1)
        #         curr_delta = curr_delta * curr_gate
        #
        #     total_delta = total_delta + curr_delta
        #
        # fusion_vectors = initial_fusion_vectors + total_delta
        # fusion_vectors = (
        #     fusion_vectors
        #     + self.alpha_ffn * self.mlp(self.norm_ffn(fusion_vectors))
        # )

        return fusion_vectors
