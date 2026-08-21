# ruff: noqa: RUF002, RUF003, N803, N806

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from orbax.checkpoint import transform_utils as ocp_transform_utils

from openpi.models.pi0 import posemb_sincos
from openpi.shared import array_typing as at

TokenReducerKind = Literal["learned_cross_attn", "mean_segments", "fixed_query_cross_attn"]
ActionEncoderKind = Literal["gru", "transformer_block"]
VisualConditionKind = Literal[
    "residual",
    "linear_joint",
    "nonlinear_joint",
]

@dataclasses.dataclass(frozen=True)
class Pi0WorldModelConfig:

    vlm_hidden_dim: int = 2048
    token_dim: int = 1024
    num_condition_tokens: int = 4
    proprio_dim: int = 32
    action_dim: int = 32
    num_reducer_heads: int = 4
    num_future_heads: int = 4
    action_embed_dim: int = 128
    gru_hidden_dim: int = 256
    gru_num_layers: int = 2
    gru_inter_layer_dropout: float = 0.1
    action_encoder_kind: ActionEncoderKind = "gru"
    visual_condition_kind: VisualConditionKind = "residual"
    transformer_num_heads: int = 4
    transformer_ffn_multiplier: int = 4
    proprio_embed_dim: int = 128
    time_embed_dim: int = 128
    time_fourier_dim: int = 16
    ffn_multiplier: int = 4
    log_var_min: float = -20.0
    log_var_max: float = 20.0
    token_reducer_kind: TokenReducerKind = "learned_cross_attn"


@dataclasses.dataclass
class WorldModelOutput:

    mu: at.Float[at.Array, "b m d"]
    log_var: at.Float[at.Array, "b m d"]
    current_tokens: at.Float[at.Array, "b m d"] | None = None


def heteroscedastic_gaussian_nll(
    target: at.Float[at.Array, "b m d"],
    mu: at.Float[at.Array, "b m d"],
    log_var: at.Float[at.Array, "b m d"],
    *,
    mask: at.Bool[at.Array, "b m"] | None = None,
) -> at.Float[at.Array, ""]:
    inv_var = jnp.exp(-log_var)
    nll = 0.5 * (jnp.square(target - mu) * inv_var + log_var)
    if mask is None:
        return jnp.mean(nll)
    w = mask.astype(nll.dtype)[..., None]
    denom = jnp.maximum(jnp.sum(w), 1.0)
    return jnp.sum(nll * w) / denom


def logvar_calibration_loss(
    target: at.Float[at.Array, "b m d"],
    mu: at.Float[at.Array, "b m d"],
    log_var: at.Float[at.Array, "b m d"],
    *,
    eps: float = 1e-6,
) -> at.Float[at.Array, ""]:
    """Match ``log_var`` to per-dimension squared residual (frozen ``mu``), in log space.

    ``target`` is stop-gradient reduced VLM features at ``t+Δ``; ``mu`` should be detached so only
    ``logvar_head`` receives gradients when combined with ``detach_features_for_logvar`` on the WM forward.
    """
    mu_sg = jax.lax.stop_gradient(mu)
    sq_err = jnp.square(target - mu_sg)
    log_sq = jnp.log(jnp.maximum(sq_err, jnp.asarray(eps, dtype=sq_err.dtype)))
    return jnp.mean(jnp.square(log_var - log_sq))


def global_confidence_from_log_var(log_var: at.Float[at.Array, "b m d"]) -> at.Float[at.Array, " b"]:
    return -jnp.mean(log_var, axis=(1, 2))


class _GRUCell(nnx.Module):

    def __init__(self, in_features: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.w_ir = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.w_hr = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.w_iz = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.w_hz = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.w_in = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.w_hn = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array, h: jax.Array) -> jax.Array:
        r = jax.nn.sigmoid(self.w_ir(x) + self.w_hr(h))
        z = jax.nn.sigmoid(self.w_iz(x) + self.w_hz(h))
        n = jnp.tanh(self.w_in(x) + r * self.w_hn(h))
        return (1.0 - z) * n + z * h


class _MultiHeadAttention(nnx.Module):

    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.w_q = nnx.Linear(dim, dim, rngs=rngs)
        self.w_k = nnx.Linear(dim, dim, rngs=rngs)
        self.w_v = nnx.Linear(dim, dim, rngs=rngs)
        self.w_o = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(
        self,
        queries: jax.Array,
        kv: jax.Array,
        kv_mask: jax.Array | None,
    ) -> jax.Array:
        q = self.w_q(queries)
        k = self.w_k(kv)
        v = self.w_v(kv)
        q = einops.rearrange(q, "b t (h c) -> b h t c", h=self.num_heads)
        k = einops.rearrange(k, "b t (h c) -> b h t c", h=self.num_heads)
        v = einops.rearrange(v, "b t (h c) -> b h t c", h=self.num_heads)
        logits = jnp.einsum("bhtc,bhsc->bhts", q, k) / jnp.sqrt(self.head_dim).astype(q.dtype)
        if kv_mask is not None:
            big_neg = jnp.asarray(jnp.finfo(logits.dtype).min, dtype=logits.dtype)
            mask = kv_mask[:, None, None, :]
            logits = jnp.where(mask, logits, big_neg)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhts,bhsc->bhtc", weights, v)
        out = einops.rearrange(out, "b h t c -> b t (h c)")
        return self.w_o(out)

class LatestVisualGlobalCondition(nnx.Module):
    """将最新视觉 tokens 压缩为全局向量，并生成视觉残差条件。"""

    def __init__(
        self,
        cfg: Pi0WorldModelConfig,
        *,
        rngs: nnx.Rngs,
    ):
        d_vlm = cfg.vlm_hidden_dim
        d = cfg.token_dim

        # 一个可学习查询，从所有有效视觉 tokens 中提取全局视觉信息。
        self.visual_query = nnx.Param(
            jax.random.normal(rngs(), (1, d)) * 0.02
        )

        # SigLIP token 维度与 WM token 维度不同时进行投影。
        self.visual_kv_proj = (
            nnx.Linear(d_vlm, d, rngs=rngs)
            if d_vlm != d
            else None
        )

        self.visual_attention = _MultiHeadAttention(
            d,
            cfg.num_future_heads,
            rngs=rngs,
        )
        self.visual_norm = nnx.LayerNorm(
            d,
            rngs=rngs,
        )

        # 零初始化保证模型刚创建时：
        # delta_u == 0
        # delta_film == 0
        # 因而严格退化为原来的 FutureConditionHead。
        self.visual_u_proj = nnx.Linear(
            d,
            d,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.visual_film_proj = nnx.Linear(
            d,
            2 * d,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def pool_visual(
        self,
        latest_visual_tokens: jax.Array,
        latest_visual_mask: jax.Array,
    ) -> jax.Array:
        """将最新视觉 tokens 压缩为一个全局视觉向量 v_latest。"""

        batch_size = latest_visual_tokens.shape[0]

        visual_kv = (
            latest_visual_tokens
            if self.visual_kv_proj is None
            else self.visual_kv_proj(latest_visual_tokens)
        )

        visual_query = jnp.broadcast_to(
            self.visual_query.value[None, :, :],
            (
                batch_size,
                self.visual_query.value.shape[0],
                visual_kv.shape[-1],
            ),
        )

        pooled = self.visual_attention(
            visual_query,
            visual_kv,
            latest_visual_mask,
        )
        pooled = self.visual_norm(pooled[:, 0, :])

        # 当某个样本不存在有效视觉 token 时，必须返回全零向量。
        has_valid_visual = jnp.any(
            latest_visual_mask,
            axis=-1,
            keepdims=True,
        )

        return jnp.where(
            has_valid_visual,
            pooled,
            jnp.zeros_like(pooled),
        )

    def __call__(
        self,
        latest_visual_tokens: jax.Array,
        latest_visual_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        v_latest = self.pool_visual(
            latest_visual_tokens,
            latest_visual_mask,
        )

        delta_u = self.visual_u_proj(v_latest)
        delta_film = self.visual_film_proj(v_latest)

        # 即使输出层的 bias 在训练后变为非零，
        # 没有有效视觉 token 的样本仍必须返回零残差。
        has_valid_visual = jnp.any(
            latest_visual_mask,
            axis=-1,
            keepdims=True,
        )

        delta_u = jnp.where(
            has_valid_visual,
            delta_u,
            jnp.zeros_like(delta_u),
        )
        delta_film = jnp.where(
            has_valid_visual,
            delta_film,
            jnp.zeros_like(delta_film),
        )

        return delta_u, delta_film


class TokenReducer(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        d_vlm = cfg.vlm_hidden_dim
        d = cfg.token_dim
        m = cfg.num_condition_tokens
        self.queries = nnx.Param(jax.random.normal(rngs(), (m, d)) * 0.02)
        self.kv_proj = nnx.Linear(d_vlm, d, rngs=rngs) if d_vlm != d else None
        self.attn = _MultiHeadAttention(d, cfg.num_reducer_heads, rngs=rngs)
        self.norm = nnx.LayerNorm(d, rngs=rngs)
        ffn_h = cfg.ffn_multiplier * d
        self.ff1 = nnx.Linear(d, ffn_h, rngs=rngs)
        self.ff2 = nnx.Linear(ffn_h, d, rngs=rngs)

    def __call__(self, H_t: jax.Array, *, kv_mask: jax.Array | None = None) -> jax.Array:
        """H_t: (B, N, d_vlm) -> C_t: (B, M, d)."""
        b = H_t.shape[0]
        kv = H_t if self.kv_proj is None else self.kv_proj(H_t)
        q = jnp.broadcast_to(self.queries.value[None, :, :], (b, self.queries.value.shape[0], kv.shape[-1]))
        x = self.attn(q, kv, kv_mask)
        x = self.norm(x)
        y = self.ff1(x)
        y = nnx.swish(y)
        y = self.ff2(y)
        return x + y


class VlmToTokenMLP(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        d_in = cfg.vlm_hidden_dim
        d_out = cfg.token_dim
        h = cfg.ffn_multiplier * d_out
        self.fc1 = nnx.Linear(d_in, h, rngs=rngs)
        self.fc2 = nnx.Linear(h, d_out, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(nnx.swish(self.fc1(x)))


class MeanSegmentTokenReducer(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        del rngs
        self._m = cfg.num_condition_tokens

    def __call__(self, H_t: jax.Array, *, kv_mask: jax.Array | None = None) -> jax.Array:
        kv = H_t
        b, n, d = kv.shape
        m = self._m
        if kv_mask is None:
            kv_mask = jnp.ones((b, n), dtype=jnp.bool_)
        pad_len = int(math.ceil(n / m) * m)
        if pad_len > n:
            kv = jnp.pad(kv, ((0, 0), (0, pad_len - n), (0, 0)))
            kv_mask = jnp.pad(kv_mask, ((0, 0), (0, pad_len - n)), constant_values=False)
        s = pad_len // m
        x = einops.rearrange(kv, "b (m s) d -> b m s d", m=m, s=s)
        w = einops.rearrange(kv_mask, "b (m s) -> b m s", m=m, s=s).astype(kv.dtype)[..., None]
        denom = jnp.maximum(jnp.sum(w, axis=2), 1.0)
        return jnp.sum(x * w, axis=2) / denom


class FixedQueryCrossAttnTokenReducer(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        d_vlm = cfg.vlm_hidden_dim
        d = cfg.token_dim
        m = cfg.num_condition_tokens
        self.queries = nnx.Variable(jax.random.normal(rngs(), (m, d)) * 0.02)
        self.kv_proj = nnx.Linear(d_vlm, d, rngs=rngs) if d_vlm != d else None
        self.attn = _MultiHeadAttention(d, cfg.num_reducer_heads, rngs=rngs)
        self.norm = nnx.LayerNorm(d, rngs=rngs)
        ffn_h = cfg.ffn_multiplier * d
        self.ff1 = nnx.Linear(d, ffn_h, rngs=rngs)
        self.ff2 = nnx.Linear(ffn_h, d, rngs=rngs)

    def __call__(self, H_t: jax.Array, *, kv_mask: jax.Array | None = None) -> jax.Array:
        b = H_t.shape[0]
        kv = H_t if self.kv_proj is None else self.kv_proj(H_t)
        q = jnp.broadcast_to(self.queries.value[None, :, :], (b, self.queries.value.shape[0], kv.shape[-1]))
        q = jax.lax.stop_gradient(q)
        x = self.attn(q, kv, kv_mask)
        x = self.norm(x)
        y = self.ff1(x)
        y = nnx.swish(y)
        y = self.ff2(y)
        return x + y


def _make_token_reducer(cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs) -> nnx.Module:
    k = cfg.token_reducer_kind
    if k == "learned_cross_attn":
        return TokenReducer(cfg, rngs=rngs)
    if k == "mean_segments":
        return MeanSegmentTokenReducer(cfg, rngs=rngs)
    if k == "fixed_query_cross_attn":
        return FixedQueryCrossAttnTokenReducer(cfg, rngs=rngs)
    raise ValueError(f"Unknown token_reducer_kind={k!r}. Expected one of: learned_cross_attn, mean_segments, fixed_query_cross_attn.")


def _gru_layer_scan(cell: _GRUCell, x: jax.Array, prefix_mask: jax.Array) -> jax.Array:
    h0 = jnp.zeros((x.shape[0], cell.w_hr.out_features), dtype=x.dtype)

    def scan_fn(carry, xt):
        m = xt[1]
        h_prev = carry
        cand = cell(xt[0], h_prev)
        h_next = jnp.where(m[:, None], cand, h_prev)
        return h_next, h_next

    _, outputs = jax.lax.scan(
        scan_fn,
        h0,
        (jnp.swapaxes(x, 0, 1), jnp.swapaxes(prefix_mask, 0, 1)),
    )
    return jnp.swapaxes(outputs, 0, 1)


class ActionPrefixEncoder(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        adim = cfg.action_dim
        e = cfg.action_embed_dim
        h = cfg.gru_hidden_dim
        self.in_proj = nnx.Linear(adim, e, rngs=rngs)
        self.in_norm = nnx.LayerNorm(e, rngs=rngs)
        self.gru0 = _GRUCell(e, h, rngs=rngs)
        self.gru1 = _GRUCell(h, h, rngs=rngs)
        self.inter_drop = nnx.Dropout(cfg.gru_inter_layer_dropout, rngs=rngs)
        self.out_proj = nnx.Linear(2 * h, h, rngs=rngs)

    def __call__(
        self,
        action_prefix: jax.Array,
        prefix_mask: jax.Array,
        *,
        rngs: nnx.Rngs,
        train: bool = False,
    ) -> jax.Array:
        x = self.in_proj(action_prefix)
        x = nnx.swish(x)
        x = self.in_norm(x)
        x = jnp.where(prefix_mask[..., None], x, jnp.zeros_like(x))

        out0 = _gru_layer_scan(self.gru0, x, prefix_mask)
        out1_in = self.inter_drop(out0, deterministic=not train, rngs=rngs)
        outputs = _gru_layer_scan(self.gru1, out1_in, prefix_mask)

        lengths = jnp.sum(prefix_mask.astype(jnp.int32), axis=-1)
        lengths = jnp.maximum(lengths, 1)
        last_idx = lengths - 1
        bidx = jnp.arange(outputs.shape[0])
        last_h = outputs[bidx, last_idx]
        denom = jnp.maximum(jnp.sum(prefix_mask.astype(outputs.dtype), axis=-1, keepdims=True), 1.0)
        mean_h = jnp.sum(outputs * prefix_mask[..., None], axis=1) / denom
        return self.out_proj(jnp.concatenate([last_h, mean_h], axis=-1))


class _ActionTokenTransformerBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = _MultiHeadAttention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        hidden = ffn_mult * dim
        self.ff1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.ff2 = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(
        self,
        x: at.Float[at.Array, "b l d"],
        *,
        mask: at.Bool[at.Array, "b l"] | None,
    ) -> at.Float[at.Array, "b l d"]:
        y = self.norm1(x)
        y = self.attn(y, y, mask)
        x = x + y
        y = self.norm2(x)
        y = self.ff2(nnx.swish(self.ff1(y)))
        return x + y


class ActionPrefixTransformerEncoder(nnx.Module):
    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        adim = cfg.action_dim
        e = cfg.action_embed_dim
        h = cfg.gru_hidden_dim
        self.in_proj = nnx.Linear(adim, e, rngs=rngs)
        self.in_norm = nnx.LayerNorm(e, rngs=rngs)
        self.block = _ActionTokenTransformerBlock(
            e,
            cfg.transformer_num_heads,
            cfg.transformer_ffn_multiplier,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(2 * e, h, rngs=rngs)

    def __call__(
        self,
        action_prefix: at.Float[at.Array, "b l a"],
        prefix_mask: at.Bool[at.Array, "b l"],
        *,
        rngs: nnx.Rngs,
        train: bool = False,
    ) -> at.Float[at.Array, "b h"]:
        del rngs, train
        x = self.in_proj(action_prefix)
        x = nnx.swish(x)
        x = self.in_norm(x)
        x = jnp.where(prefix_mask[..., None], x, jnp.zeros_like(x))
        x = self.block(x, mask=prefix_mask)
        x = jnp.where(prefix_mask[..., None], x, jnp.zeros_like(x))

        lengths = jnp.sum(prefix_mask.astype(jnp.int32), axis=-1)
        lengths = jnp.maximum(lengths, 1)
        last_idx = lengths - 1
        bidx = jnp.arange(x.shape[0])
        last_h = x[bidx, last_idx]
        denom = jnp.maximum(jnp.sum(prefix_mask.astype(x.dtype), axis=-1, keepdims=True), 1.0)
        mean_h = jnp.sum(x * prefix_mask[..., None], axis=1) / denom
        return self.out_proj(jnp.concatenate([last_h, mean_h], axis=-1))


class ProprioEncoder(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        d = cfg.proprio_dim
        o = cfg.proprio_embed_dim
        self.fc1 = nnx.Linear(d, o, rngs=rngs)
        self.fc2 = nnx.Linear(o, o, rngs=rngs)

    def __call__(self, proprio: jax.Array) -> jax.Array:
        x = self.fc1(proprio)
        x = nnx.swish(x)
        return self.fc2(x)


class TimeEncoder(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        fdim = cfg.time_fourier_dim
        raw_dim = 3 + fdim
        o = cfg.time_embed_dim
        self.fourier_dim = fdim
        self.fc1 = nnx.Linear(raw_dim, o, rngs=rngs)
        self.fc2 = nnx.Linear(o, o, rngs=rngs)

    def __call__(self, delta_t: jax.Array) -> jax.Array:
        t = jnp.maximum(delta_t.astype(jnp.float32), 0.0)
        f = posemb_sincos(t, self.fourier_dim, min_period=1e-3, max_period=1e3)
        raw = jnp.concatenate(
            [t[:, None], jnp.log1p(t)[:, None], jnp.square(t)[:, None], f],
            axis=-1,
        )
        x = self.fc1(raw)
        x = nnx.swish(x)
        return self.fc2(x)
class NonlinearJointCondition(nnx.Module):
    """非线性联合建模动作、本体、时间和最新视觉条件。"""

    def __init__(
        self,
        cfg: Pi0WorldModelConfig,
        *,
        rngs: nnx.Rngs,
    ):
        d = cfg.token_dim
        g_dim = (
            cfg.gru_hidden_dim
            + cfg.proprio_embed_dim
            + cfg.time_embed_dim
        )
        joint_dim = g_dim + d
        hidden_dim = 2 * d

        self.joint_norm = nnx.LayerNorm(
            joint_dim,
            rngs=rngs,
        )
        self.joint_fc1 = nnx.Linear(
            joint_dim,
            hidden_dim,
            rngs=rngs,
        )
        self.joint_fc2 = nnx.Linear(
            hidden_dim,
            d,
            rngs=rngs,
        )

        self.u_head = nnx.Linear(
            d,
            d,
            rngs=rngs,
        )
        self.film_head = nnx.Linear(
            d,
            2 * d,
            rngs=rngs,
        )

    def __call__(
        self,
        global_vec: jax.Array,
        v_latest: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        joint_condition = jnp.concatenate(
            [global_vec, v_latest],
            axis=-1,
        )

        z = self.joint_norm(joint_condition)
        z = self.joint_fc1(z)
        z = nnx.swish(z)
        z = self.joint_fc2(z)

        u = self.u_head(z)
        film = self.film_head(z)

        return u, film
    
class LinearJointCondition(nnx.Module):
    """使用一次线性映射联合建模 g 与 v_latest。"""

    def __init__(
        self,
        cfg: Pi0WorldModelConfig,
        *,
        rngs: nnx.Rngs,
    ):
        d = cfg.token_dim
        g_dim = (
            cfg.gru_hidden_dim
            + cfg.proprio_embed_dim
            + cfg.time_embed_dim
        )
        joint_dim = g_dim + d

        self.u_proj = nnx.Linear(
            joint_dim,
            d,
            rngs=rngs,
        )
        self.film_proj = nnx.Linear(
            joint_dim,
            2 * d,
            rngs=rngs,
        )

    def __call__(
        self,
        global_vec: jax.Array,
        v_latest: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        joint_condition = jnp.concatenate(
            [global_vec, v_latest],
            axis=-1,
        )

        u = self.u_proj(joint_condition)
        film = self.film_proj(joint_condition)

        return u, film

class FutureConditionHead(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        d = cfg.token_dim
        g = cfg.proprio_embed_dim + cfg.time_embed_dim + cfg.gru_hidden_dim
        self.u_proj = nnx.Linear(g, d, rngs=rngs)
        self.film = nnx.Linear(g, 2 * d, rngs=rngs)
        self.block0 = _FutureTokenBlock(d, cfg.num_future_heads, cfg.ffn_multiplier, rngs=rngs)
        self.block1 = _FutureTokenBlock(d, cfg.num_future_heads, cfg.ffn_multiplier, rngs=rngs)
        self.mean_head = nnx.Linear(d, d, rngs=rngs)
        self.logvar_head = nnx.Linear(d, d, rngs=rngs)

        self.visual_condition_kind = cfg.visual_condition_kind

        if cfg.visual_condition_kind == "residual":
            self.joint_condition = None
        elif cfg.visual_condition_kind == "linear_joint":
            self.joint_condition = LinearJointCondition(
                cfg,
                rngs=rngs,
            )
        elif cfg.visual_condition_kind == "nonlinear_joint":
            self.joint_condition = NonlinearJointCondition(
                cfg,
                rngs=rngs,
            )
        else:
            raise ValueError(
                "Unknown visual_condition_kind="
                f"{cfg.visual_condition_kind!r}. "
                "Expected one of: residual, linear_joint."
            )

        self.log_var_min = cfg.log_var_min
        self.log_var_max = cfg.log_var_max

    def __call__(
        self,
        C_t: jax.Array,
        global_vec: jax.Array,
        *,
        v_latest: jax.Array | None = None,
        delta_u: jax.Array | None = None,
        delta_film: jax.Array | None = None,
        detach_features_for_logvar: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        if self.visual_condition_kind == "residual":
            u = self.u_proj(global_vec)
            film = self.film(global_vec)

            if delta_u is not None:
                u = u + delta_u

            if delta_film is not None:
                film = film + delta_film

        elif self.visual_condition_kind in (
            "linear_joint",
            "nonlinear_joint",
        ):
            if v_latest is None:
                raise ValueError(
                    f"{self.visual_condition_kind} "
                    "condition requires v_latest"
                )

            u, film = self.joint_condition(
                global_vec,
                v_latest,
            )

        else:
            raise ValueError(
                "Unknown visual_condition_kind="
                f"{self.visual_condition_kind!r}"
            )

        gamma, beta = jnp.split(
            film,
            2,
            axis=-1,
        )

        x = (
            gamma[:, None, :]
            * (C_t + u[:, None, :])
            + beta[:, None, :]
        )

        x = self.block0(x)
        x = self.block1(x)
        mu = self.mean_head(x)

        x_lv = (
            jax.lax.stop_gradient(x)
            if detach_features_for_logvar
            else x
        )
        log_var = jnp.clip(
            self.logvar_head(x_lv),
            self.log_var_min,
            self.log_var_max,
        )

        return mu, log_var


class _FutureTokenBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = _MultiHeadAttention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        hidden = ffn_mult * dim
        self.ff1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.ff2 = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.norm1(x)
        y = self.attn(y, y, None)
        x = x + y
        y = self.norm2(x)
        y = self.ff1(y)
        y = nnx.swish(y)
        y = self.ff2(y)
        return x + y


class Pi0FutureWorldModel(nnx.Module):

    def __init__(self, cfg: Pi0WorldModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.token_reducer = _make_token_reducer(cfg, rngs=rngs)
        if cfg.token_reducer_kind == "mean_segments" and cfg.vlm_hidden_dim != cfg.token_dim:
            self.reducer_vlm_to_token = VlmToTokenMLP(cfg, rngs=rngs)
        else:
            self.reducer_vlm_to_token = None
        if cfg.action_encoder_kind == "gru":
            self.action_encoder = ActionPrefixEncoder(cfg, rngs=rngs)
        elif cfg.action_encoder_kind == "transformer_block":
            self.action_encoder = ActionPrefixTransformerEncoder(cfg, rngs=rngs)
        else:
            raise ValueError(
                f"Unknown action_encoder_kind={cfg.action_encoder_kind!r}. Expected one of: gru, transformer_block."
            )
        self.proprio_encoder = ProprioEncoder(cfg, rngs=rngs)
        self.time_encoder = TimeEncoder(cfg, rngs=rngs)
        self.future_head = FutureConditionHead(cfg, rngs=rngs)

        self.latest_visual_global_condition = (
            LatestVisualGlobalCondition(
                cfg,
                rngs=rngs,
            )
        )

    def _reduced_to_model_tokens(
        self,
        H: at.Float[at.Array, "b n d_vlm"],
        *,
        kv_mask: at.Bool[at.Array, "b n"] | None,
    ) -> at.Float[at.Array, "b m d"]:
        raw = self.token_reducer(H, kv_mask=kv_mask)
        if self.reducer_vlm_to_token is not None:
            return self.reducer_vlm_to_token(raw)
        return raw

    @at.typecheck
    def reduce_tokens(
        self,
        H: at.Float[at.Array, "b n d_vlm"],
        *,
        kv_mask: at.Bool[at.Array, "b n"] | None = None,
    ) -> at.Float[at.Array, "b m d"]:
        return self._reduced_to_model_tokens(H, kv_mask=kv_mask)

    @at.typecheck
    def __call__(
        self,
        H_t: at.Float[at.Array, "b n d_vlm"],
        proprio: at.Float[at.Array, "b q"],
        action_prefix: at.Float[at.Array, "b l a"],
        prefix_mask: at.Bool[at.Array, "b l"],
        delta_t: at.Float[at.Array, " b"],
        *,
        kv_mask: at.Bool[at.Array, "b n"] | None = None,
        latest_visual_tokens: at.Float[
            at.Array,
            "b v d_vlm",
        ] | None = None,
        latest_visual_mask: at.Bool[
            at.Array,
            "b v",
        ] | None = None,
        rngs: nnx.Rngs,
        train: bool = False,
        return_current_tokens: bool = True,
        detach_wm_features_for_logvar: bool = False,
    ) -> WorldModelOutput:
        # C_t = self._reduced_to_model_tokens(H_t, kv_mask=kv_mask)
        # m = self.action_encoder(action_prefix, prefix_mask, rngs=rngs, train=train)
        # p = self.proprio_encoder(proprio)
        # e = self.time_encoder(delta_t)
        # g = jnp.concatenate([m, p, e], axis=-1)
        # mu, log_var = self.future_head(C_t, g, detach_features_for_logvar=detach_wm_features_for_logvar)
        # return WorldModelOutput(
        #     mu=mu,
        #     log_var=log_var,
        #     current_tokens=C_t if return_current_tokens else None,
        # )
        if (latest_visual_tokens is None) != (
            latest_visual_mask is None
        ):
            raise ValueError(
                "latest_visual_tokens and latest_visual_mask "
                "must be provided together"
            )
        if (
            self.cfg.visual_condition_kind
            in (
                "linear_joint",
                "nonlinear_joint",
            )
            and latest_visual_tokens is None
        ):
            raise ValueError(
                "visual_condition_kind="
                f"{self.cfg.visual_condition_kind!r} "
                "requires latest visual inputs"
            )

        C_t = self._reduced_to_model_tokens(
            H_t,
            kv_mask=kv_mask,
        )

        m = self.action_encoder(
            action_prefix,
            prefix_mask,
            rngs=rngs,
            train=train,
        )
        p = self.proprio_encoder(proprio)
        e = self.time_encoder(delta_t)

        g = jnp.concatenate(
            [m, p, e],
            axis=-1,
        )
        v_latest = None
        delta_u = None
        delta_film = None

        if latest_visual_tokens is not None:
            if self.cfg.visual_condition_kind == "residual":
                delta_u, delta_film = (
                    self.latest_visual_global_condition(
                        latest_visual_tokens,
                        latest_visual_mask,
                    )
                )

            elif self.cfg.visual_condition_kind in (
                "linear_joint",
                "nonlinear_joint",
            ):
                v_latest = (
                    self.latest_visual_global_condition.pool_visual(
                        latest_visual_tokens,
                        latest_visual_mask,
                    )
                )

        mu, log_var = self.future_head(
            C_t,
            g,
            v_latest=v_latest,
            delta_u=delta_u,
            delta_film=delta_film,
            detach_features_for_logvar=(
                detach_wm_features_for_logvar
            ),
        )

        return WorldModelOutput(
            mu=mu,
            log_var=log_var,
            current_tokens=(
                C_t
                if return_current_tokens
                else None
            ),
        )


def load_pi0_future_world_model(
    checkpoint_dir: str | pathlib.Path,
    *,
    config: Pi0WorldModelConfig | None = None,
    dtype: jnp.dtype | None = None,
) -> Pi0FutureWorldModel:
    from openpi.models import model as model_mod

    path = pathlib.Path(checkpoint_dir).resolve()
    cfg = config or Pi0WorldModelConfig()
    wm = Pi0FutureWorldModel(cfg, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(wm)
    params = model_mod.restore_params(path / "params", dtype=dtype)
    params = ocp_transform_utils.intersect_trees(state.to_pure_dict(), params)
    state.replace_by_pure_dict(params)
    return nnx.merge(graphdef, state)
