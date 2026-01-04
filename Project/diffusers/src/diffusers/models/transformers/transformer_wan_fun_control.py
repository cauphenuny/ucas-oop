# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WanFunControl Transformer3DModel with camera control support.

This model extends the base WanTransformer3DModel to support camera-controlled video generation
by accepting additional control latents (e.g., Plücker ray embeddings) that are concatenated
with the input latents before patchification.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin
from ..cache_utils import CacheMixin
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm
from .transformer_wan import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)


logger = logging.get_logger(__name__)


class WanFunControlTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for camera-controlled video generation using Wan Fun-Control.

    This model extends WanTransformer3DModel to accept control latents (camera/Plücker embeddings)
    that are concatenated with input latents before patchification.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Number of attention heads.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input latents (before control concatenation).
        control_channels (`int`, defaults to `36`):
            The number of channels in the control latents (e.g., Plücker embeddings).
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `4096`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            Type of normalization for query/key.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        image_dim (`int`, *optional*, defaults to `None`):
            Dimension for image embeddings (for I2V models).
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections.
        rope_max_seq_len (`int`, defaults to `1024`):
            Maximum sequence length for rotary positional embeddings.
        pos_embed_seq_len (`int`, *optional*, defaults to `None`):
            Sequence length for positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        control_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Rotary position embeddings
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

        # 2. Patch embedding - accepts concatenated input+control channels
        total_in_channels = in_channels + control_channels
        self.patch_embedding = nn.Conv3d(total_in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 3. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def patchify(self, hidden_states: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Patchify the input with optional camera control latents.

        Args:
            hidden_states: Input latents of shape [batch, in_channels, frames, height, width]
            control_camera_latents_input: Optional control latents of shape [batch, control_channels, frames, height, width]

        Returns:
            Patchified tensor of shape [batch, inner_dim, patch_frames, patch_height, patch_width]
        """
        if control_camera_latents_input is not None:
            # Concatenate input and control latents along channel dimension
            hidden_states = torch.cat([hidden_states, control_camera_latents_input], dim=1)

        # Apply patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        control_camera_latents: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional camera control latents.

        Args:
            hidden_states: Input latents [batch, in_channels, frames, height, width]
            timestep: Timestep tensor
            encoder_hidden_states: Text encoder hidden states
            encoder_hidden_states_image: Optional image encoder hidden states
            control_camera_latents: Optional camera control latents [batch, control_channels, frames, height, width]
            return_dict: Whether to return a dict
            attention_kwargs: Additional attention arguments

        Returns:
            Model output tensor or dict
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. Rotary position embeddings
        rotary_emb = self.rope(hidden_states)

        # 2. Patchify with control latents
        hidden_states = self.patchify(hidden_states, control_camera_latents)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 3. Timestep handling
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        # 4. Condition embeddings
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 5. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 6. Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
