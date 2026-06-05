from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_time_text_embed import QwenTimeTextEmbed
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_rms_norm import QwenTransformerRMSNorm


class QwenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        patch_size: int = 2,
        zero_cond_t: bool = False,
    ) -> None:
        super().__init__()
        # Qwen-Image-Edit-2511 sets `zero_cond_t: true` in transformer/config.json: the conditioning
        # image latent tokens are modulated as clean (timestep 0) in every block, while the noise
        # latents + text stream keep the real denoise timestep. Off (2509 / Qwen-Image T2I) it is a
        # no-op. See diffusers `QwenImageTransformer2DModel`.
        self.zero_cond_t = zero_cond_t
        self.inner_dim = num_attention_heads * attention_head_dim
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_norm = QwenTransformerRMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        self.transformer_blocks = [QwenTransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim) for i in range(num_layers)]  # fmt: off
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        qwen_image_ids: mx.array | None = None,
        cond_image_grid: tuple[int, int, int] | None = None,
    ) -> mx.array:
        hidden_states = self.img_in(hidden_states)
        batch_size = hidden_states.shape[0]
        timestep = QwenTransformer._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)

        # zero_cond_t (Qwen-Image-Edit-2511): double the timestep -> [t, 0] and build a per-token
        # modulate_index (0 = noise latents -> real t, 1 = conditioning image -> t 0). With no
        # conditioning image (T2I) the flag is a no-op. Mirrors diffusers QwenImageTransformer2DModel.
        modulate_index = None
        if self.zero_cond_t and cond_image_grid is not None:
            timestep = mx.concatenate([timestep, timestep * 0], axis=0)
            modulate_index = QwenTransformer._build_modulate_index(config, cond_image_grid, batch_size)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        text_embeddings = self.time_text_embed(timestep, hidden_states)
        image_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self.pos_embed,
            config=config,
            cond_image_grid=cond_image_grid,
        )
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                modulate_index=modulate_index,
            )
        # norm_out (and the text stream, inside the block) use only the real-timestep half of temb.
        norm_embeddings = text_embeddings[:batch_size] if modulate_index is not None else text_embeddings
        hidden_states = self.norm_out(hidden_states, norm_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    @staticmethod
    def _build_modulate_index(
        config: Config,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]],
        batch_size: int,
    ) -> mx.array:
        # img_shapes = [(1, latent_h, latent_w)] + cond grids (see _compute_rotary_embeddings); the
        # noise tokens come first (index 0), every conditioning-image token follows (index 1).
        noise_len = (config.height // 16) * (config.width // 16)
        grids = cond_image_grid if isinstance(cond_image_grid, list) else [cond_image_grid]
        cond_len = sum(int(t) * int(h) * int(w) for (t, h, w) in grids)
        row = [0] * noise_len + [1] * cond_len
        index = mx.array([row], dtype=mx.int32)
        if batch_size > 1:
            index = mx.broadcast_to(index, (batch_size, noise_len + cond_len))
        return index

    @staticmethod
    def _apply_transformer_block(
        idx: int,
        block: QwenTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: tuple[mx.array, mx.array],
        modulate_index: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_emb=image_rotary_embeddings,
            block_idx=idx,
            modulate_index=modulate_index,
        )

    @staticmethod
    def _compute_timestep(
        t: int | float,
        config: Config,
    ) -> mx.array:
        if isinstance(t, int):
            if t < len(config.scheduler.sigmas):
                timestep_idx = t
                time_step = config.scheduler.sigmas[timestep_idx]
            else:
                timestep_idx = None
                for idx, ts in enumerate(config.scheduler.timesteps):
                    if abs(int(ts.item()) - t) < 1:
                        timestep_idx = idx
                        break
                if timestep_idx is None:
                    time_step = t / 1000.0
                else:
                    time_step = config.scheduler.sigmas[timestep_idx]
        else:
            timestep_idx = None
            time_step = t

        timestep = mx.array(np.full((1,), time_step, dtype=np.float32))
        return timestep

    @staticmethod
    def _compute_rotary_embeddings(
        encoder_hidden_states_mask: mx.array,
        pos_embed: QwenEmbedRopeMLX,
        config: Config,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> tuple[mx.array, mx.array]:
        latent_height = config.height // 16
        latent_width = config.width // 16

        if cond_image_grid is None:
            img_shapes = [(1, latent_height, latent_width)]
        else:
            if isinstance(cond_image_grid, list):
                img_shapes = [(1, latent_height, latent_width)] + cond_image_grid
            else:
                img_shapes = [(1, latent_height, latent_width), cond_image_grid]

        txt_seq_lens = [int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])]  # fmt: off
        img_rotary_emb, txt_rotary_emb = pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)
        return img_rotary_emb, txt_rotary_emb
