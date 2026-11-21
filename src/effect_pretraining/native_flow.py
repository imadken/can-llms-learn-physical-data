"""Temporary numerical prefix around the unmodified native SmolVLA action path."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def attention_mask(padding: torch.Tensor, causal_blocks: torch.Tensor) -> torch.Tensor:
    cumulative = causal_blocks.cumsum(dim=1)
    permitted = cumulative[:, None, :] <= cumulative[:, :, None]
    return permitted & padding[:, None, :] & padding[:, :, None]


def pad_native(values: torch.Tensor, width: int) -> torch.Tensor:
    if values.shape[-1] > width:
        raise ValueError("Native coordinate count exceeds checkpoint width")
    return F.pad(values, (0, width - values.shape[-1]))


def native_flow_target(
    actions: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    tau = time[:, None, None]
    return (1 - tau) * actions + tau * noise, noise - actions


def physical_action_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average only the seven physical coordinates, never the padded slots."""
    return F.mse_loss(predicted[..., :7], target[..., :7])


class NumericalFlow(nn.Module):
    def __init__(self, native: nn.Module, projector: nn.Linear):
        super().__init__()
        self.native = native
        self.effect_projector = projector
        for parameter in native.parameters():
            parameter.requires_grad_(False)
        for name, parameter in native.named_parameters():
            if name.startswith(
                (
                    "vlm_with_expert.lm_expert.",
                    "action_in_proj.",
                    "action_out_proj.",
                    "action_time_mlp_in.",
                    "action_time_mlp_out.",
                )
            ):
                parameter.requires_grad_(True)

    def prefix(self, state: torch.Tensor, effect: torch.Tensor, masked: torch.Tensor):
        state_token = self.native.state_proj(pad_native(state, self.native.config.max_state_dim))
        effect_token = self.effect_projector(effect)
        embeddings = torch.stack([state_token, effect_token], dim=1)
        padding = torch.stack([torch.ones_like(masked, dtype=torch.bool), ~masked], dim=1)
        # Distinct causal blocks prevent the state token from relaying an effect.
        blocks = torch.ones_like(padding, dtype=torch.int64)
        return embeddings, padding, blocks

    def forward(
        self,
        state: torch.Tensor,
        effect: torch.Tensor,
        masked: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions = pad_native(actions, self.native.config.max_action_dim)
        if noise is None:
            noise = self.native.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.native.sample_time(actions.shape[0], actions.device)
        noisy, target = native_flow_target(actions, noise, time)
        prefix, prefix_padding, prefix_blocks = self.prefix(state, effect, masked)
        suffix, suffix_padding, suffix_blocks = self.native.embed_suffix(noisy, time)
        padding = torch.cat([prefix_padding, suffix_padding], dim=1)
        blocks = torch.cat([prefix_blocks, suffix_blocks], dim=1)
        (_, output), _ = self.native.vlm_with_expert.forward(
            attention_mask=attention_mask(padding, blocks),
            position_ids=padding.cumsum(dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            fill_kv_cache=False,
        )
        prediction = self.native.action_out_proj(output[:, -16:].float())
        return physical_action_loss(prediction, target)

    @torch.no_grad()
    def generate(
        self,
        state: torch.Tensor,
        effect: torch.Tensor,
        masked: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if noise.shape[1:] != (16, self.native.config.max_action_dim):
            raise ValueError("Generation requires the full native H=16 noise tensor")
        prefix, padding, blocks = self.prefix(state, effect, masked)
        _, cache = self.native.vlm_with_expert.forward(
            attention_mask=attention_mask(padding, blocks),
            position_ids=padding.cumsum(dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        values = noise.clone()
        for step in range(10):
            time = torch.full((len(state),), 1 - step / 10, device=state.device)
            velocity = self.native.denoise_step(padding, cache, values, time)
            values = values - 0.1 * velocity
        return values[..., :7]

    def transfer_weights(self) -> dict[str, torch.Tensor]:
        """Export only native action generation weights, excluding state/effect/VLM."""
        trainable = {name for name, p in self.native.named_parameters() if p.requires_grad}
        return {
            name: value.detach().cpu().clone()
            for name, value in self.native.state_dict().items()
            if name in trainable
        }


def load_native_policy(checkpoint: Path, vlm_assets: Path):
    """Strictly load the pinned public checkpoint using existing official LeRobot code."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_file

    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if config.type != "smolvla":
        raise ValueError("The locked checkpoint must be SmolVLA")
    config.vlm_model_name = str(vlm_assets)
    config.load_vlm_weights = False
    config.chunk_size = 16
    config.n_action_steps = 4
    config.num_steps = 10
    config.train_state_proj = False
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    config.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    config.normalization_mapping = {kind.value: NormalizationMode.IDENTITY for kind in FeatureType}
    policy = SmolVLAPolicy(config)
    policy.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    return policy
