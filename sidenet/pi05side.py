from torch import nn
from torch import Tensor
from .sidenet import SideNet
from torch.nn import functional as F
from openpi.models.model import Observation
from openpi.training import config as training_config
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

import torch


def load_pi05_pytorch(config_name: str, weight_path: str):
    train_cfg = training_config.get_config(config_name)
    pi05_model = train_cfg.model.load_pytorch(train_cfg, weight_path)
    return pi05_model


class PI05withSideNet(nn.Module):
    """PI0.5 wrapped with SideNet (v2 perceiver-based architecture).

    Injection strategy: SideNet output tokens are **concatenated** with the
    action-expert suffix embeddings so that the action expert can attend to
    the fused multimodal information during both training and inference.
    """

    def __init__(
        self,
        sidenet_config_path: str,
        pi05_config_name: str,
        pi05_weights_path: str,
    ):
        super().__init__()
        self.sidenet = SideNet(sidenet_config_path)
        self.pi05_model = load_pi05_pytorch(pi05_config_name, pi05_weights_path)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _resolve_modality_input_dict(
        self,
        observation: Observation,
        modality_input_dict: dict | None,
    ) -> dict:
        if modality_input_dict is not None:
            resolved = modality_input_dict
        else:
            resolved = observation.modalities or {}
        if not resolved:
            raise ValueError(
                "No SideNet modalities were provided. Pass `modality_input_dict` "
                "explicitly or include modalities in `Observation.modalities`."
            )
        return resolved

    def _extract_text_embeddings(
        self,
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Get raw language embeddings from PaliGemma's embedding layer.

        Returns
        -------
        text_embs : ``(B, L, 2048)`` float32
        text_mask : ``(B, L)`` bool
        """
        text_embs = self.pi05_model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        text_embs = text_embs.to(dtype=torch.float32)
        return text_embs, lang_masks

    def _run_sidenet(
        self,
        modality_input_dict: dict,
        text_embs: Tensor,
        text_mask: Tensor,
    ) -> Tensor:
        """Run SideNet and return output tokens (B, num_fusion_queries, expert_width)."""
        return self.sidenet(
            modality_inputs=modality_input_dict,
            text_embeddings=text_embs,
            text_mask=text_mask,
        )

    def _prepend_side_tokens(
        self,
        side_tokens: Tensor,
        suffix_embs: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepend SideNet output tokens to the suffix sequence.

        The SideNet tokens form their own causal attention group so that:
          - prefix tokens cannot attend to them
          - they can attend to prefix tokens
          - action tokens can attend to them
        """
        side_tokens = side_tokens.to(device=suffix_embs.device, dtype=suffix_embs.dtype)
        B, num_side, _ = side_tokens.shape

        if side_tokens.shape[-1] != suffix_embs.shape[-1]:
            raise ValueError(
                f"SideNet output dim ({side_tokens.shape[-1]}) must match "
                f"suffix embedding dim ({suffix_embs.shape[-1]})"
            )

        suffix_embs = torch.cat([side_tokens, suffix_embs], dim=1)

        side_pad = torch.ones(B, num_side, dtype=suffix_pad_masks.dtype, device=suffix_pad_masks.device)
        suffix_pad_masks = torch.cat([side_pad, suffix_pad_masks], dim=1)

        # First SideNet token starts a new attention group (att=1);
        # remaining SideNet tokens stay in the same group (att=0).
        side_att = torch.zeros(B, num_side, dtype=suffix_att_masks.dtype, device=suffix_att_masks.device)
        side_att[:, 0] = 1
        suffix_att_masks = torch.cat([side_att, suffix_att_masks], dim=1)

        return suffix_embs, suffix_pad_masks, suffix_att_masks

    @staticmethod
    def _mean_l2_norm(tensor: Tensor | None) -> Tensor | None:
        if tensor is None:
            return None
        flat = tensor.detach().to(dtype=torch.float32).reshape(tensor.shape[0], -1)
        return torch.linalg.vector_norm(flat, dim=1).mean()

    def _collect_forward_metrics(
        self,
        *,
        actions: Tensor,
        pred_actions: Tensor,
        side_tokens: Tensor | None,
        loss_action_dim: int | None = None,
    ) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}
        if loss_action_dim is not None:
            actions = actions[..., :loss_action_dim]
            pred_actions = pred_actions[..., :loss_action_dim]
        action_mse = (
            pred_actions.detach().to(dtype=torch.float32)
            - actions.detach().to(dtype=torch.float32)
        ).square()
        metrics["train/action_mse/t0"] = action_mse[:, 0].mean()
        metrics["train/action_mse/t_last"] = action_mse[:, -1].mean()
        metrics["train/action_mse/mean_over_horizon"] = action_mse.mean()
        injection_norm = self._mean_l2_norm(side_tokens)
        if injection_norm is not None:
            metrics["sidenet/injection_norm"] = injection_norm
        return metrics

    # ------------------------------------------------------------------
    # training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        observation: Observation,
        actions: Tensor,
        modality_input_dict: dict | None = None,
        noise=None,
        time=None,
        train_pi05: bool = False,
        loss_action_dim: int | None = None,
        return_aux: bool = False,
    ) -> Tensor:
        """Training forward pass.

        SideNet output tokens are concatenated with ``suffix_embs`` before
        the joint backbone + expert forward, so the action expert can attend
        to the fused multimodal tokens.
        """
        for param in self.pi05_model.parameters():
            param.requires_grad_(train_pi05)

        modality_input_dict = self._resolve_modality_input_dict(observation, modality_input_dict)

        images, img_masks, lang_tokens, lang_masks, state = (
            self.pi05_model._preprocess_observation(observation, train=True)
        )

        # --- text embeddings for SideNet ---
        text_embs, text_mask = self._extract_text_embeddings(lang_tokens, lang_masks)

        if noise is None:
            noise = self.pi05_model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.pi05_model.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi05_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.pi05_model.embed_suffix(state, x_t, time)
        )

        # --- SideNet: run once and prepend to suffix ---
        side_tokens = self._run_sidenet(modality_input_dict, text_embs, text_mask)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self._prepend_side_tokens(
            side_tokens, suffix_embs, suffix_pad_masks, suffix_att_masks,
        )

        if (
            self.pi05_model.paligemma_with_expert.paligemma.language_model.layers[0]
            .self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self.pi05_model._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.pi05_model.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self.pi05_model._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond,
        )

        # Extract only the last action_horizon tokens (skip prepended SideNet tokens).
        suffix_out = suffix_out[:, -self.pi05_model.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.pi05_model.action_out_proj(suffix_out)

        v_t = self.pi05_model._apply_checkpoint(action_out_proj_func, suffix_out)

        if loss_action_dim is not None:
            if loss_action_dim <= 0:
                raise ValueError(f"loss_action_dim must be positive, got {loss_action_dim}")
            if loss_action_dim > actions.shape[-1]:
                raise ValueError(
                    f"loss_action_dim ({loss_action_dim}) exceeds action dim ({actions.shape[-1]})"
                )
            u_t = u_t[..., :loss_action_dim]
            v_t = v_t[..., :loss_action_dim]

        losses = F.mse_loss(u_t, v_t, reduction="none")

        if not return_aux:
            return losses

        pred_noise = noise[..., :loss_action_dim] if loss_action_dim is not None else noise
        pred_actions = pred_noise - v_t
        aux_metrics = self._collect_forward_metrics(
            actions=actions,
            pred_actions=pred_actions,
            side_tokens=side_tokens,
            loss_action_dim=loss_action_dim,
        )
        return losses, aux_metrics

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def sample_actions(
        self,
        device,
        observation: Observation,
        modality_input_dict: dict | None = None,
        noise=None,
        num_steps: int = 10,
    ) -> Tensor:
        """Run pi05 inference with SideNet suffix injection.

        SideNet is evaluated **once** before the denoising loop.  Its output
        tokens are prepended to the suffix at every denoising step.
        """
        modality_input_dict = self._resolve_modality_input_dict(observation, modality_input_dict)
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (
                bsize,
                self.pi05_model.config.action_horizon,
                self.pi05_model.config.action_dim,
            )
            noise = self.pi05_model.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            self.pi05_model._preprocess_observation(observation, train=False)
        )

        # --- text embeddings + SideNet (once) ---
        text_embs, text_mask = self._extract_text_embeddings(lang_tokens, lang_masks)
        side_tokens = self._run_sidenet(modality_input_dict, text_embs, text_mask)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi05_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )

        # Build prefix KV cache (unchanged — SideNet no longer touches prefix).
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.pi05_model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.pi05_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.pi05_model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Denoising loop
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time_val = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time_val >= -dt / 2:
            expanded_time = time_val.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                side_tokens=side_tokens,
            )
            x_t = x_t + dt * v_t
            time_val += dt

        return x_t

    def denoise_step(
        self,
        state: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        timestep: Tensor,
        side_tokens: Tensor | None = None,
    ) -> Tensor:
        """One denoising step with SideNet suffix injection."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.pi05_model.embed_suffix(state, x_t, timestep)
        )

        # Prepend cached SideNet tokens to suffix.
        if side_tokens is not None:
            suffix_embs, suffix_pad_masks, suffix_att_masks = self._prepend_side_tokens(
                side_tokens, suffix_embs, suffix_pad_masks, suffix_att_masks,
            )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self.pi05_model._prepare_attention_masks_4d(full_att_2d_masks)
        self.pi05_model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.pi05_model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        # Only take the last action_horizon tokens (skip prepended SideNet tokens).
        suffix_out = suffix_out[:, -self.pi05_model.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.pi05_model.action_out_proj(suffix_out)
