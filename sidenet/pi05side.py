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
    '''
    The implementation of PI05 model with SideNet
    '''
    def __init__(self, 
                 sidenet_config_path: str, 
                 pi05_config_name: str,
                 pi05_weights_path: str):
        super().__init__()
        self.sidenet = SideNet(sidenet_config_path)
        self.pi05_model = load_pi05_pytorch(pi05_config_name, pi05_weights_path)    
    
    # FIX: resolve SideNet inputs from the standard Observation interface unless
    # an explicit override dict is provided.
    def _resolve_modality_input_dict(
        self,
        observation: Observation,
        modality_input_dict: dict | None,
    ) -> dict:
        if modality_input_dict is not None:
            resolved_modalities = modality_input_dict
        else:
            resolved_modalities = observation.modalities or {}

        if not resolved_modalities:
            raise ValueError(
                "No SideNet modalities were provided. Pass `modality_input_dict` explicitly "
                "or include modalities in `Observation.modalities`."
            )
        return resolved_modalities
    
   
    
    def forward(
        self,
        observation: Observation,
        actions: Tensor,
        modality_input_dict: dict | None = None,
        noise=None,
        time=None,
        train_pi05: bool = False,
        use_backbone_injector: bool = True,
        use_expert_injector: bool = True,
    ) -> Tensor:
        """
        This function is used for training only. For inference, use 'sample action' method instead.
        The forward function will go through the full training forward pipeline of pi05
        while injecting the SideNet fusion vectors into the pi05 model at the specified positions. The loss is computed as the MSE between the predicted actions and the ground truth actions, weighted by the time step.
        By default, only the SideNet is trainable. Set `train_pi05=True` to
        jointly train the base pi05 model and the SideNet.
        Args:
            obervation: Observation class defined in openpi.models.model, contains:
                Observation[
                    images: dict[str, torch.Tensor],
                    image_masks: dict[str, torch.Tensor],
                    state: torch.Tensor,
                    tokenized_prompt: torch.Tensor | None,
                    ...
                    ]
            actions: torch.Tensor with shape (batch_size, action_horizon, action_dim)
            modality_input_dict: A dictionary, whose keys are modality names (e.g. tactile), and values are the raw input from sensors.

        """
        # Base pi05 trainability can still be toggled per forward call. SideNet
        # trainability is configured externally by the training script so that
        # specific branches can be frozen without being re-enabled here.
        for param in self.pi05_model.parameters():
            param.requires_grad_(train_pi05)
        # FIX: default to the modalities already stored inside Observation so
        # training and inference can stay on the standard Observation interface.
        modality_input_dict = self._resolve_modality_input_dict(observation, modality_input_dict)

        images, img_masks, lang_tokens, lang_masks, state = self.pi05_model._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.pi05_model.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.pi05_model.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi05_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.pi05_model.embed_suffix(state, x_t, time)

        side_outputs = self.sidenet(
            modality_inputs=modality_input_dict,
            return_intermediates=True,
        )
        side_injections = side_outputs["injections"]

        # Backbone-side injection:
        # treat the injected output as extra prefix tokens so that the base model
        # can consume them through the existing prefix attention path.
        # TODO: Delete assertions below when all exps are done.
        backbone_injection = side_injections.get("backbone") if use_backbone_injector else None
        if backbone_injection is not None:
            backbone_injection = backbone_injection.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
            if backbone_injection.ndim != 3:
                raise ValueError(
                    "backbone injection must have shape (batch_size, num_side_tokens, hidden_dim)"
                )
            if backbone_injection.shape[0] != prefix_embs.shape[0]:
                raise ValueError(
                    "backbone injection batch size must match prefix embeddings batch size"
                )
            if backbone_injection.shape[-1] != prefix_embs.shape[-1]:
                raise ValueError(
                    "backbone injection hidden dim must match prefix embedding hidden dim"
                )
            prefix_embs = torch.cat([prefix_embs, backbone_injection], dim=1)

            batch_size, num_side_tokens = backbone_injection.shape[:2]
            side_pad_masks = torch.ones(
                batch_size,
                num_side_tokens,
                dtype=prefix_pad_masks.dtype,
                device=prefix_pad_masks.device,
            )
            side_att_masks = torch.zeros(
                batch_size,
                num_side_tokens,
                dtype=prefix_att_masks.dtype,
                device=prefix_att_masks.device,
            )
            prefix_pad_masks = torch.cat([prefix_pad_masks, side_pad_masks], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, side_att_masks], dim=1)

        # Expert-side injection:
        # add SideNet output to the AdaRMSNorm condition so that the action expert
        # receives the extra modality signal without changing deeper helper code.
        expert_injection = side_injections.get("expert") if use_expert_injector else None
        if expert_injection is not None:
            expert_injection = expert_injection.to(device=adarms_cond.device, dtype=adarms_cond.dtype)
            if expert_injection.ndim != 2:
                raise ValueError(
                    "expert injection must have shape (batch_size, hidden_dim)"
                )
            if expert_injection.shape[0] != adarms_cond.shape[0]:
                raise ValueError(
                    "expert injection batch size must match adarms_cond batch size"
                )
            if expert_injection.shape[-1] != adarms_cond.shape[-1]:
                raise ValueError(
                    "expert injection hidden dim must match adarms_cond hidden dim"
                )
            adarms_cond = adarms_cond + expert_injection

        if (
            self.pi05_model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self.pi05_model._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
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
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.pi05_model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.pi05_model.action_out_proj(suffix_out)

        v_t = self.pi05_model._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")   

    def sample_actions(
        self,
        device,
        observation: Observation,
        modality_input_dict: dict | None = None,
        noise=None,
        num_steps: int = 10,
        use_backbone_injector: bool = True,
        use_expert_injector: bool = True,
    ) -> Tensor:
        """Run pi05 inference while injecting SideNet outputs.

        This mirrors `PI0Pytorch.sample_actions`, with SideNet integrated using
        the same two-path strategy as `forward()`:
        1. backbone injection -> append extra prefix tokens
        2. expert injection -> add to AdaRMSNorm condition at each denoise step
        """
        # FIX: default to the modalities already stored inside Observation so
        # policy inference can keep calling `sample_actions(observation, ...)`.
        modality_input_dict = self._resolve_modality_input_dict(observation, modality_input_dict)
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (
                bsize,
                self.pi05_model.config.action_horizon,
                self.pi05_model.config.action_dim,
            )
            noise = self.pi05_model.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self.pi05_model._preprocess_observation(
            observation,
            train=False,
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi05_model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )

        # SideNet is evaluated once during sampling and then reused across all
        # denoising steps.
        side_outputs = self.sidenet(
            modality_inputs=modality_input_dict,
            return_intermediates=True,
        )
        side_injections = side_outputs["injections"]

        # --- SideNet modification: inject backbone output as extra prefix tokens.
        backbone_injection = side_injections.get("backbone") if use_backbone_injector else None
        if backbone_injection is not None:
            backbone_injection = backbone_injection.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
            if backbone_injection.ndim != 3:
                raise ValueError(
                    "backbone injection must have shape (batch_size, num_side_tokens, hidden_dim)"
                )
            if backbone_injection.shape[0] != prefix_embs.shape[0]:
                raise ValueError(
                    "backbone injection batch size must match prefix embeddings batch size"
                )
            if backbone_injection.shape[-1] != prefix_embs.shape[-1]:
                raise ValueError(
                    "backbone injection hidden dim must match prefix embedding hidden dim"
                )

            prefix_embs = torch.cat([prefix_embs, backbone_injection], dim=1)

            batch_size, num_side_tokens = backbone_injection.shape[:2]
            side_pad_masks = torch.ones(
                batch_size,
                num_side_tokens,
                dtype=prefix_pad_masks.dtype,
                device=prefix_pad_masks.device,
            )
            side_att_masks = torch.zeros(
                batch_size,
                num_side_tokens,
                dtype=prefix_att_masks.dtype,
                device=prefix_att_masks.device,
            )
            prefix_pad_masks = torch.cat([prefix_pad_masks, side_pad_masks], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, side_att_masks], dim=1)

        # --- SideNet modification: cache expert-side condition once and reuse it
        # across all denoising steps.
        expert_injection = side_injections.get("expert") if use_expert_injector else None

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image/language/side-token KV cache once.
        prefix_att_2d_masks_4d = self.pi05_model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.pi05_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.pi05_model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                expert_injection=expert_injection,
            )

            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def denoise_step(
        self,
        state: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        timestep: Tensor,
        expert_injection: Tensor | None = None,
    ) -> Tensor:
        """Apply one denoising step with optional SideNet expert injection.

        This is copied from `PI0Pytorch.denoise_step` and extended only at the
        AdaRMSNorm condition path so SideNet can inject the cached expert-side
        signal during sampling.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.pi05_model.embed_suffix(
            state,
            x_t,
            timestep,
        )

        # --- SideNet modification: inject cached expert condition into the
        # action expert AdaRMSNorm pathway at every denoising step.
        if expert_injection is not None:
            expert_injection = expert_injection.to(device=adarms_cond.device, dtype=adarms_cond.dtype)
            if expert_injection.ndim != 2:
                raise ValueError(
                    "expert injection must have shape (batch_size, hidden_dim)"
                )
            if expert_injection.shape[0] != adarms_cond.shape[0]:
                raise ValueError(
                    "expert injection batch size must match adarms_cond batch size"
                )
            if expert_injection.shape[-1] != adarms_cond.shape[-1]:
                raise ValueError(
                    "expert injection hidden dim must match adarms_cond hidden dim"
                )
            adarms_cond = adarms_cond + expert_injection

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
        suffix_out = suffix_out[:, -self.pi05_model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.pi05_model.action_out_proj(suffix_out)
