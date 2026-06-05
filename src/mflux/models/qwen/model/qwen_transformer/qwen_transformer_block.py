from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_transformer.qwen_feed_forward import QwenFeedForward


class QwenTransformerBlock(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128):
        super().__init__()

        self.img_mod_silu = nn.SiLU()
        self.img_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.img_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.attn = QwenAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)
        self.img_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.img_ff = QwenFeedForward(dim=dim)

        self.txt_mod_silu = nn.SiLU()
        self.txt_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.txt_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_ff = QwenFeedForward(dim=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        text_embeddings: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        block_idx: int | None = None,
        modulate_index: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        img_mod_params = self.img_mod_linear(self.img_mod_silu(text_embeddings))
        # zero_cond_t: img modulation uses the doubled temb [real_t ; zero_t] (selected per token by
        # modulate_index); the text stream uses only the real-timestep half.
        txt_embeddings = text_embeddings[: text_embeddings.shape[0] // 2] if modulate_index is not None else text_embeddings
        txt_mod_params = self.txt_mod_linear(self.txt_mod_silu(txt_embeddings))

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = QwenTransformerBlock._modulate(img_normed, img_mod1, modulate_index)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = QwenTransformerBlock._modulate(txt_normed, txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            img_modulated=img_modulated,
            txt_modulated=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            block_idx=block_idx,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(img_normed2, img_mod2, modulate_index)

        img_mlp_output = self.img_ff(img_modulated2)

        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_ff(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _modulate(
        x: mx.array, mod_params: mx.array, index: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(mod_params, 3, axis=-1)
        if index is None:
            return x * (1 + scale[:, None, :]) + shift[:, None, :], gate[:, None, :]
        # zero_cond_t: mod_params has a doubled batch [real_t ; zero_t]. Select per token by index
        # (0 = noise latents -> real timestep half, 1 = conditioning image -> zero timestep half).
        actual_batch = shift.shape[0] // 2
        index_expanded = index[:, :, None]

        def pick(arr: mx.array) -> mx.array:
            a0, a1 = arr[:actual_batch], arr[actual_batch:]
            return mx.where(index_expanded == 0, a0[:, None, :], a1[:, None, :])

        shift_r, scale_r, gate_r = pick(shift), pick(scale), pick(gate)
        return x * (1 + scale_r) + shift_r, gate_r
