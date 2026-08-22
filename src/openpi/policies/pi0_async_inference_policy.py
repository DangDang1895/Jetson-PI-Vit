# ruff: noqa: SLF001, RUF002, RUF003
from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import time
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0_world_model import Pi0FutureWorldModel, global_confidence_from_log_var
import openpi.policies.policy as _policy
from openpi.policies import wm_inference_verify as _wm_verify
from openpi.policies.wm_multi_rollout_schedule import wm_multi_rollout_adaptive_max_rounds
import openpi.shared.nnx_utils as nnx_utils
import openpi.transforms as transforms

logger = logging.getLogger("openpi")


class _LowKappaFullPi0Fallback(Exception):
    """WM multi-rollout saw κ_r < κ_0 - kappa_delta at round ``r>=1`` (after WM, before that round's AE).

    ``wm_ae_rounds_completed`` counts full WM→AE cycles finished (rounds ``0..r-1``).
    ``kappa_per_round_np`` has length ``r+1`` (one κ per WM forward, including the round that triggered fallback).

    Low-replan two-phase (client): execute ``rollout_actions_model`` for ``rollout_len`` steps in sim, run full Pi0
    on the post-rollout image, then stitch ``glue_actions_model`` with ``full_pi0_actions[overlap:]`` (model-space
    rows before ``_output_transform`` on the server; the infer ``except`` path converts them for the payload).
    """

    __slots__ = (
        "wm_ae_rounds_completed",
        "kappa_per_round_np",
        "rollout_len",
        "rollout_actions_model",
        "glue_actions_model",
    )

    def __init__(
        self,
        *,
        wm_ae_rounds_completed: int,
        kappa_per_round_np: np.ndarray,
        rollout_len: int,
        rollout_actions_model: np.ndarray,
        glue_actions_model: np.ndarray,
    ) -> None:
        super().__init__()
        self.wm_ae_rounds_completed = int(wm_ae_rounds_completed)
        self.kappa_per_round_np = kappa_per_round_np
        self.rollout_len = int(rollout_len)
        self.rollout_actions_model = rollout_actions_model
        self.glue_actions_model = glue_actions_model


ASYNC_KEY = "openpi/async"
AeProprioSource = Literal["prefix_t", "future_rollout", "vlash_last_action"]

PER_STEP_ACTION_PREFIX_LEN = 10


@dataclasses.dataclass(frozen=True)
class _BrainSnapshot:
    """一次已经完成并可供小脑读取的大脑快照。"""

    episode_id: int
    source_step: int
    h_t: jax.Array
    prefix_mask: jax.Array
    kv_cache: Any
    proprio: jax.Array

def _rollforward_proprio_batched(state: jnp.ndarray, actions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    m = mask.astype(jnp.float32)[..., None]
    acc = jnp.sum(m * actions, axis=1)
    d = min(int(state.shape[-1]), int(acc.shape[-1]))
    if d <= 0:
        return state
    return state.at[..., :d].add(acc[..., :d])


def _last_valid_prefix_action_batched(actions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    m = mask.astype(jnp.float32)
    rev = m[:, ::-1]
    cum = jnp.cumsum(rev, axis=1)
    pick_rev = (cum == 1) & (rev > 0.5)
    pick = pick_rev[:, ::-1]
    has_any = jnp.sum(m, axis=1, keepdims=True) > 0
    weights = jnp.where(has_any, pick.astype(jnp.float32), jnp.zeros_like(pick).at[:, 0].set(1.0))
    return jnp.einsum("bla,bl->ba", actions, weights)


def _vlash_last_action_proprio_batched(state: jnp.ndarray, actions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    last_a = _last_valid_prefix_action_batched(actions, mask)
    d = min(int(state.shape[-1]), int(last_a.shape[-1]))
    if d <= 0:
        return state
    return state.at[..., :d].set(last_a[..., :d])


class Pi0AsyncInferencePolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        inner: _policy.Policy,
        *,
        pi0: Pi0,
        world_model: Pi0FutureWorldModel | None,
        action_norm: transforms.Normalize | None,
        model_action_dim: int,
        ae_proprio_source: AeProprioSource = "vlash_last_action",
    ):
        self._inner = inner
        self._pi0 = pi0
        self._world_model = world_model
        self._action_norm = action_norm
        self._model_action_dim = model_action_dim
        self._ae_proprio_source: AeProprioSource = ae_proprio_source
        self._prefix_states = nnx_utils.module_jit(pi0.prefix_hidden_states)
        self._sample_with_future = nnx_utils.module_jit(pi0.sample_actions)

        # 当前观测只执行一次共享 SigLIP。
        self._latest_visual = nnx_utils.module_jit(
            pi0.encode_visual_tokens
        )

        # 大脑复用共享 SigLIP 输出，生成 H 和 KV。
        self._build_brain_cache = nnx_utils.module_jit(
            pi0.prefix_hidden_states_and_kv_from_visual
        )

        # 小脑使用旧 Snapshot 中缓存的 KV 去噪。
        self._sample_with_cached_kv = nnx_utils.module_jit(
            pi0.sample_actions_from_cached_kv
        )



        if world_model is not None:
            wm_gd, wm_st = nnx.split(world_model)

            def _wm_jitted(h_t, proprio, action_prefix_pad, prefix_mask, delta_t, rng_key):
                wm = nnx.merge(wm_gd, wm_st)
                out = wm(
                    h_t,
                    proprio,
                    action_prefix_pad,
                    prefix_mask,
                    delta_t,
                    kv_mask=None,
                    rngs=nnx.Rngs(rng_key),
                    train=False,
                    return_current_tokens=False,
                )
                kappa = global_confidence_from_log_var(out.log_var)
                return out.mu, kappa
            def _wm_visual_jitted(
                h_t,
                proprio,
                action_prefix_pad,
                prefix_mask,
                delta_t,
                latest_visual_tokens,
                latest_visual_mask,
                rng_key,
            ):
                wm = nnx.merge(
                    wm_gd,
                    wm_st,
                )

                out = wm(
                    h_t,
                    proprio,
                    action_prefix_pad,
                    prefix_mask,
                    delta_t,
                    kv_mask=None,
                    latest_visual_tokens=(
                        latest_visual_tokens
                    ),
                    latest_visual_mask=(
                        latest_visual_mask
                    ),
                    rngs=nnx.Rngs(rng_key),
                    train=False,
                    return_current_tokens=False,
                )

                return out.mu

            self._wm_forward_visual = jax.jit(
                _wm_visual_jitted
            )

            self._wm_forward = jax.jit(_wm_jitted)
        else:
            self._wm_forward = None
            self._wm_forward_visual = None

        # 单线程后台大脑。避免多个大脑任务排队。
        self._brain_executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="openpi-brain",
            )
        )

        self._brain_future: (
            concurrent.futures.Future[_BrainSnapshot] | None
        ) = None

        self._brain_snapshot: _BrainSnapshot | None = None
        self._per_step_episode_id: int | None = None

        # key 是实际执行动作的控制步：
        # history[k] 表示从 O_k 到 O_{k+1} 执行的动作 a_k。
        self._executed_action_history: dict[
            int,
            np.ndarray,
        ] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._inner.metadata

    def _build_brain_snapshot_task(
        self,
        *,
        episode_id: int,
        source_step: int,
        observation: _model.Observation,
        visual_tokens: jax.Array,
        visual_mask: jax.Array,
        proprio: jax.Array,
    ) -> _BrainSnapshot:
        """后台大脑：从共享视觉继续生成 H/KV。"""

        h_t, prefix_mask, kv_cache = (
            self._build_brain_cache(
                observation,
                visual_tokens,
                visual_mask,
            )
        )

        # JAX 默认异步派发。必须等待设备计算完成，
        # Future.done() 才能真正表示 H/KV 已经可用。
        h_t = jax.block_until_ready(h_t)
        prefix_mask = jax.block_until_ready(
            prefix_mask
        )
        kv_cache = jax.tree.map(
            jax.block_until_ready,
            kv_cache,
        )
        proprio = jax.block_until_ready(proprio)

        return _BrainSnapshot(
            episode_id=episode_id,
            source_step=source_step,
            h_t=h_t,
            prefix_mask=prefix_mask,
            kv_cache=kv_cache,
            proprio=proprio,
        )

    def _publish_finished_brain_snapshot(
        self,
    ) -> None:
        future = self._brain_future

        if future is None or not future.done():
            return

        self._brain_future = None
        snapshot = future.result()

        if snapshot.episode_id != self._per_step_episode_id:
            logger.info(
                "Discard stale brain snapshot: "
                "episode=%d source_step=%d",
                snapshot.episode_id,
                snapshot.source_step,
            )
            return

        current = self._brain_snapshot
        if (
            current is not None
            and snapshot.source_step
            <= current.source_step
        ):
            return

        # # 大脑推理延迟=9
        # if (
        #     current is not None
        #     and snapshot.source_step - current.source_step < 9
        # ):
        #     return

        self._brain_snapshot = snapshot

        logger.info(
            "Published brain snapshot: "
            "episode=%d source_step=%d",
            snapshot.episode_id,
            snapshot.source_step,
        )

    def _start_brain_update(
        self,
        *,
        episode_id: int,
        source_step: int,
        observation: _model.Observation,
        visual_tokens: jax.Array,
        visual_mask: jax.Array,
        proprio: jax.Array,
    ) -> None:
        if self._brain_future is not None:
            return

        self._brain_future = self._brain_executor.submit(
            self._build_brain_snapshot_task,
            episode_id=episode_id,
            source_step=source_step,
            observation=observation,
            visual_tokens=visual_tokens,
            visual_mask=visual_mask,
            proprio=proprio,
        )

    def _reset_per_step_state(
        self,
        episode_id: int,
    ) -> None:
        # episode 切换发生频率低，可以在这里等待旧任务退出，
        # 避免旧 episode 的大脑任务占用后台 worker。
        if self._brain_future is not None:
            try:
                self._brain_future.result(
                    timeout=600
                )
            except Exception:
                logger.exception(
                    "Previous brain update failed "
                    "during episode reset."
                )

        self._brain_future = None
        self._brain_snapshot = None
        self._executed_action_history.clear()
        self._per_step_episode_id = episode_id

        logger.info(
            "Reset per-step cerebellum state: "
            "episode=%d",
            episode_id,
        )

    def _build_fixed_action_prefix(
        self,
        *,
        snapshot: _BrainSnapshot,
        current_step: int,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        """构造从 Snapshot_s 到当前 O_t 的固定长度动作前缀。"""

        delta_steps = (
            current_step - snapshot.source_step
        )

        if delta_steps < 1:
            raise ValueError(
                "Per-step WM requires current_step > "
                "snapshot.source_step, got "
                f"current_step={current_step}, "
                f"source_step={snapshot.source_step}."
            )

        if delta_steps > PER_STEP_ACTION_PREFIX_LEN:
            raise ValueError(
                "Per-step WM snapshot exceeds the trained "
                "action-prefix range: "
                f"delta_steps={delta_steps}, "
                f"max={PER_STEP_ACTION_PREFIX_LEN}."
            )

        missing_steps = [
            step
            for step in range(
                snapshot.source_step,
                current_step,
            )
            if step not in self._executed_action_history
        ]

        if missing_steps:
            raise ValueError(
                "Missing actually executed actions for "
                f"steps={missing_steps}."
            )

        raw_actions = np.stack(
            [
                self._executed_action_history[step]
                for step in range(
                    snapshot.source_step,
                    current_step,
                )
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        if self._action_norm is not None:
            raw_actions = self._action_norm(
                {
                    "actions": raw_actions,
                }
            )["actions"]

        model_actions = transforms.pad_to_dim(
            raw_actions,
            self._model_action_dim,
            axis=-1,
        )
        model_actions = np.asarray(
            model_actions,
            dtype=np.float32,
        )

        action_prefix_pad = np.zeros(
            (
                PER_STEP_ACTION_PREFIX_LEN,
                self._model_action_dim,
            ),
            dtype=np.float32,
        )
        action_prefix_pad[:delta_steps] = (
            model_actions
        )

        prefix_mask = np.zeros(
            (
                PER_STEP_ACTION_PREFIX_LEN,
            ),
            dtype=bool,
        )
        prefix_mask[:delta_steps] = True

        return (
            jnp.asarray(
                action_prefix_pad
            )[np.newaxis, ...],
            jnp.asarray(
                prefix_mask
            )[np.newaxis, ...],
            jnp.asarray(
                [float(delta_steps)],
                dtype=jnp.float32,
            ),
        )

    def _infer_per_step_cerebellum(
        self,
        inputs: dict,
        meta: dict,
    ) -> dict:
        if self._wm_forward_visual is None:
            raise RuntimeError(
                "Per-step cerebellum requires a loaded "
                "world model."
            )

        episode_id = int(meta["episode_id"])
        current_step = int(meta["control_step"])

        if current_step < 0:
            raise ValueError(
                "control_step must be non-negative."
            )

        if episode_id != self._per_step_episode_id:
            self._reset_per_step_state(
                episode_id
            )

        # O_t 到达时，a_{t-1} 已经执行完成。
        if current_step > 0:
            last_action = meta.get(
                "last_executed_action"
            )
            if last_action is None:
                raise ValueError(
                    "Per-step request with control_step > 0 "
                    "requires last_executed_action."
                )

            self._executed_action_history[
                current_step - 1
            ] = np.asarray(
                last_action,
                dtype=np.float32,
            ).reshape(-1).copy()

        # 如果上一次后台大脑已经完成，在本轮小脑读取前发布。
        self._publish_finished_brain_snapshot()

        batched = jax.tree.map(
            lambda x: jnp.asarray(x)[
                np.newaxis,
                ...
            ],
            inputs,
        )
        observation = _model.Observation.from_dict(
            batched
        )

        # 当前 O_t 只执行一次共享 SigLIP。
        latest_visual_tokens, latest_visual_mask = (
            self._latest_visual(
                observation
            )
        )

        t0 = time.monotonic()

        # 初始 episode：缓存为空，等待正常大脑产生
        # H_0/KV_0，并直接用它生成第一块正常 Pi0 动作。
        if self._brain_snapshot is None:
            snapshot = self._build_brain_snapshot_task(
                episode_id=episode_id,
                source_step=current_step,
                observation=observation,
                visual_tokens=latest_visual_tokens,
                visual_mask=latest_visual_mask,
                proprio=batched["state"],
            )
            self._brain_snapshot = snapshot

            self._inner._rng, k_sample = (
                jax.random.split(
                    self._inner._rng,
                    2,
                )
            )

            actions_batched = (
                self._sample_with_cached_kv(
                    k_sample,
                    observation,
                    cached_kv_cache=(
                        snapshot.kv_cache
                    ),
                    cached_prefix_mask=(
                        snapshot.prefix_mask
                    ),
                    num_steps=(
                        self._inner._sample_kwargs.get(
                            "num_steps",
                            10,
                        )
                    ),
                    future_condition_tokens=None,
                )
            )

            used_snapshot = snapshot
            delta_steps = 0
            initial_full_pi0 = True

        else:
            # 本轮小脑固定读取当前已经完成的快照。
            used_snapshot = self._brain_snapshot

            if used_snapshot.source_step >= current_step:
                raise ValueError(
                    "Per-step snapshot must be older than "
                    "the current observation after the "
                    "initial request: "
                    f"source_step={used_snapshot.source_step}, "
                    f"current_step={current_step}."
                )

            # 当前视觉产生后，后台大脑也使用相同视觉
            # 计算 H_t/KV_t；小脑不等待它。
            self._start_brain_update(
                episode_id=episode_id,
                source_step=current_step,
                observation=observation,
                visual_tokens=latest_visual_tokens,
                visual_mask=latest_visual_mask,
                proprio=batched["state"],
            )

            (
                action_prefix_pad,
                action_prefix_mask,
                delta_t,
            ) = self._build_fixed_action_prefix(
                snapshot=used_snapshot,
                current_step=current_step,
            )

            delta_steps = int(
                current_step
                - used_snapshot.source_step
            )

            self._inner._rng, k_wm, k_sample = (
                jax.random.split(
                    self._inner._rng,
                    3,
                )
            )

            mu_t = self._wm_forward_visual(
                used_snapshot.h_t,
                used_snapshot.proprio,
                action_prefix_pad,
                action_prefix_mask,
                delta_t,
                latest_visual_tokens,
                latest_visual_mask,
                k_wm,
            )

            actions_batched = (
                self._sample_with_cached_kv(
                    k_sample,
                    observation,
                    cached_kv_cache=(
                        used_snapshot.kv_cache
                    ),
                    cached_prefix_mask=(
                        used_snapshot.prefix_mask
                    ),
                    num_steps=(
                        self._inner._sample_kwargs.get(
                            "num_steps",
                            10,
                        )
                    ),
                    future_condition_tokens=mu_t,
                )
            )

            initial_full_pi0 = False

        actions_batched = jax.block_until_ready(
            actions_batched
        )

        outputs = {
            "state": batched["state"],
            "actions": actions_batched,
        }
        outputs = jax.tree.map(
            lambda x: np.asarray(
                x[0, ...]
            ),
            outputs,
        )
        outputs = self._inner._output_transform(
            outputs
        )

        outputs["policy_timing"] = {
            "infer_ms": (
                time.monotonic() - t0
            )
            * 1000,
        }
        outputs[
            "openpi/per_step_cerebellum"
        ] = True
        outputs[
            "openpi/per_step_initial_full_pi0"
        ] = initial_full_pi0
        outputs[
            "openpi/brain_snapshot_source_step"
        ] = int(used_snapshot.source_step)
        outputs[
            "openpi/current_control_step"
        ] = current_step
        outputs[
            "openpi/wm_delta_steps"
        ] = delta_steps

        logger.info(
            "per_step_cerebellum: "
            "episode=%d current_step=%d "
            "snapshot_source_step=%d "
            "delta_steps=%d initial=%s "
            "brain_update_running=%s",
            episode_id,
            current_step,
            used_snapshot.source_step,
            delta_steps,
            initial_full_pi0,
            self._brain_future is not None,
        )

        return outputs

    @override
    def infer(self, obs: dict) -> dict:
        d = dict(obs)
        meta = d.pop(ASYNC_KEY, None)

        if (
            isinstance(meta, dict)
            and meta.get(
                "per_step_cerebellum"
            )
        ):
            if (
                self._world_model is None
                or self._wm_forward_visual is None
            ):
                raise RuntimeError(
                    "per_step_cerebellum requested "
                    "without a loaded world model."
                )

            inputs = self._inner._input_transform(
                d
            )

            return self._infer_per_step_cerebellum(
                inputs,
                meta,
            )


        if not isinstance(meta, dict) or not meta.get("use_world_model"):
            return self._inner.infer(d)
        if self._wm_forward is None or self._world_model is None:
            logger.warning(
                "openpi/async.use_world_model is set but no world model checkpoint was loaded; using standard Pi0."
            )
            return self._inner.infer(d)

        inputs = self._inner._input_transform(d)
        t0 = time.monotonic()
        used_wm_multi = isinstance(meta.get("wm_multi_rollout"), dict) and bool(meta["wm_multi_rollout"].get("enabled"))
        wm_extras: dict[str, Any] = {}
        try:
            if used_wm_multi:
                actions_batched, kappa_rounds, wm_extras = self._infer_wm_multi_rollout_ae(inputs, meta)
            else:
                actions_batched, kappa_rounds = self._infer_with_world_model(inputs, meta)
        except _LowKappaFullPi0Fallback as ex:
            logger.info(
                "wm_multi_rollout adaptive low_replan: κ dropped below κ₀ - kappa_delta "
                "(completed_wm_ae_rounds=%d wm_kappa_rounds=%s); two-phase full Pi0 "
                "(rollout_len=%d then glue+pi0[overlap:]).",
                ex.wm_ae_rounds_completed,
                np.asarray(ex.kappa_per_round_np, dtype=np.float64).reshape(-1).tolist(),
                ex.rollout_len,
            )
            stack = np.concatenate(
                [np.asarray(ex.rollout_actions_model, dtype=np.float32), np.asarray(ex.glue_actions_model, dtype=np.float32)],
                axis=0,
            )
            # ``_output_transform`` includes ``Unnormalize(..., strict=True)``, which requires every norm_stats key
            # (e.g. ``state`` and ``actions``) to be present; mirror the success-path dict.
            batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            state_for_out = np.asarray(batched["state"][0, ...])
            out_tf = self._inner._output_transform({"state": state_for_out, "actions": stack})
            act_all = np.asarray(out_tf["actions"], dtype=np.float32)
            n_roll = int(ex.rollout_len)
            roll_client = np.array(act_all[:n_roll], dtype=np.float32, copy=True)
            glue_client = np.array(act_all[n_roll:], dtype=np.float32, copy=True)
            return {
                "openpi/wm_low_replan_two_phase": True,
                "openpi/wm_low_replan_fallback_full_pi0": True,
                "openpi/wm_low_replan_partial_wm_ae_rounds": int(ex.wm_ae_rounds_completed),
                "openpi/wm_confidence_kappa": np.asarray(ex.kappa_per_round_np, dtype=np.float32).reshape(-1).copy(),
                "openpi/wm_low_replan_rollout_len": n_roll,
                "openpi/wm_low_replan_rollout_actions": roll_client,
                "openpi/wm_low_replan_glue_actions": glue_client,
                "policy_timing": {"infer_ms": (time.monotonic() - t0) * 1000},
            }
        except Exception:
            logger.exception("World model inference failed; falling back to standard Pi0.")
            out = self._inner.infer(d)
            if isinstance(out, dict):
                out = dict(out)
                out["openpi/wm_exception_fallback_full_pi0"] = True
                return out
            return out

        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        outputs = {"state": batched["state"], "actions": actions_batched}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._inner._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
        outputs["openpi/async_world_model"] = True
        if used_wm_multi:
            outputs["openpi/async_wm_multi_rollout"] = True
            if wm_extras.get("wm_stitch_n") is not None:
                outputs["openpi/wm_stitch_n"] = int(wm_extras["wm_stitch_n"])
            if wm_extras.get("adaptive_max_rounds") is not None:
                outputs["openpi/wm_adaptive_max_rounds"] = int(wm_extras["adaptive_max_rounds"])
            if wm_extras.get("adaptive_exec_len") is not None:
                outputs["openpi/wm_adaptive_exec_len"] = int(wm_extras["adaptive_exec_len"])
            if wm_extras.get("adaptive_early_stop"):
                outputs["openpi/wm_adaptive_early_stop"] = True
        kr = np.asarray(jax.device_get(kappa_rounds), dtype=np.float32).reshape(-1)
        outputs["openpi/wm_confidence_kappa"] = kr
        logger.info("wm_confidence_kappa_rounds=%s", kr.tolist())
        if wm_extras.get("adaptive_max_rounds") is not None:
            logger.info(
                "wm_multi_rollout_adaptive: max_rounds=%s early_stop=%s exec_len=%s",
                wm_extras.get("adaptive_max_rounds"),
                wm_extras.get("adaptive_early_stop"),
                wm_extras.get("adaptive_exec_len"),
            )
        return outputs

    def _infer_with_world_model(self, inputs: dict, meta: dict) -> tuple[jax.Array, jax.Array]:
        self._inner._rng, k_wm, k_sample = jax.random.split(self._inner._rng, 3)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(batched)
        h_t = self._prefix_states(observation)

        wm_proprio_t = meta.get("wm_proprio_t")
        if wm_proprio_t is None:
            proprio = batched["state"]
        else:
            wm_q = np.asarray(wm_proprio_t, dtype=np.float32).reshape(-1)
            # Normalize raw proprio (e.g. Libero 8-D) with ``state`` stats, then pad to model width for WM.
            if self._action_norm is not None:
                wm_q = self._action_norm({"state": wm_q})["state"]
            wm_q = transforms.pad_to_dim(wm_q, self._model_action_dim, axis=-1)
            proprio = jnp.asarray(wm_q)[jnp.newaxis, ...]

        ap = np.asarray(meta["action_prefix"], dtype=np.float32)
        if ap.ndim != 2:
            raise ValueError("action_prefix must be 2D (L, A)")
        # Same as proprio: normalize with ``actions`` stats at native width (e.g. Libero 7-D), then pad for WM.
        if self._action_norm is not None:
            ap = self._action_norm({"actions": ap})["actions"]
        ap = transforms.pad_to_dim(ap, self._model_action_dim, axis=-1)

        prefix_mask = np.asarray(meta["prefix_mask"], dtype=bool)
        if prefix_mask.shape[0] != ap.shape[0]:
            raise ValueError("prefix_mask length must match action_prefix length")

        delta_t = float(np.asarray(meta["delta_t"]).reshape(()))
        delta = jnp.asarray([delta_t], dtype=jnp.float32)
        ap_j = jnp.asarray(ap)[jnp.newaxis, ...]
        mask_j = jnp.asarray(prefix_mask)[jnp.newaxis, ...]
        mu, kappa = self._wm_forward(h_t, proprio, ap_j, mask_j, delta, k_wm)

        ae_src = meta.get("ae_proprio_source")
        if ae_src is not None:
            if ae_src not in ("prefix_t", "future_rollout", "vlash_last_action"):
                raise ValueError(
                    "openpi/async.ae_proprio_source must be 'prefix_t', 'future_rollout', or 'vlash_last_action', "
                    f"got {ae_src!r}"
                )
            effective_ae: AeProprioSource = ae_src
        else:
            effective_ae = self._ae_proprio_source

        obs_for_ae = observation
        if effective_ae == "future_rollout":
            state_ae = _rollforward_proprio_batched(batched["state"], ap_j, mask_j)
            obs_for_ae = _model.Observation.from_dict({**batched, "state": state_ae})
        elif effective_ae == "vlash_last_action":
            state_ae = _vlash_last_action_proprio_batched(batched["state"], ap_j, mask_j)
            obs_for_ae = _model.Observation.from_dict({**batched, "state": state_ae})

        if self._world_model is not None and _wm_verify.wm_inference_verify_mode() != "off":
            _wm_verify.run_wm_inference_verification(
                pi0=self._pi0,
                world_model=self._world_model,
                observation=obs_for_ae,
                mu=mu,
            )

        actions = self._sample_with_future(
            k_sample,
            obs_for_ae,
            num_steps=self._inner._sample_kwargs.get("num_steps", 10),
            future_condition_tokens=mu,
        )
        return actions, jnp.reshape(kappa, (-1,))

    def _observation_for_ae_from_prefix(
        self,
        *,
        batched: dict,
        observation: _model.Observation,
        ap_j: jnp.ndarray,
        mask_j: jnp.ndarray,
        effective_ae: AeProprioSource,
    ) -> _model.Observation:
        if effective_ae == "future_rollout":
            state_ae = _rollforward_proprio_batched(batched["state"], ap_j, mask_j)
            return _model.Observation.from_dict({**batched, "state": state_ae})
        if effective_ae == "vlash_last_action":
            state_ae = _vlash_last_action_proprio_batched(batched["state"], ap_j, mask_j)
            return _model.Observation.from_dict({**batched, "state": state_ae})
        return observation

    def _infer_wm_multi_rollout_ae(self, inputs: dict, meta: dict) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
        mr = meta["wm_multi_rollout"]
        num_rounds = int(mr["num_rounds"])
        delta_t_wm = float(mr["delta_t"])
        overlap = int(mr["overlap"])
        adaptive = bool(mr.get("adaptive_kappa"))
        low_replan = bool(mr.get("adaptive_kappa_low_replan"))
        kappa_th = float(mr.get("kappa_delta", 0.2))
        if num_rounds < 1:
            raise ValueError("wm_multi_rollout.num_rounds must be >= 1")
        if overlap < 1:
            raise ValueError("wm_multi_rollout.overlap must be >= 1 for non-empty first WM prefix")
        delta_idx = int(round(delta_t_wm))
        if delta_idx < 1:
            raise ValueError("wm_multi_rollout.delta_t must round to a positive integer index step")

        seed = meta.get("wm_seed_chunk")
        if seed is None:
            raise ValueError("wm_multi_rollout mode requires meta['wm_seed_chunk'] (H, A) from prefetch")
        seed_np = np.asarray(seed, dtype=np.float32)
        if seed_np.ndim != 2:
            raise ValueError("wm_seed_chunk must be 2D (H, A)")

        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(batched)
        h_t = self._prefix_states(observation)

        wm_proprio_t = meta.get("wm_proprio_t")
        if wm_proprio_t is None:
            proprio = batched["state"]
        else:
            wm_q = np.asarray(wm_proprio_t, dtype=np.float32).reshape(-1)
            if self._action_norm is not None:
                wm_q = self._action_norm({"state": wm_q})["state"]
            wm_q = transforms.pad_to_dim(wm_q, self._model_action_dim, axis=-1)
            proprio = jnp.asarray(wm_q)[jnp.newaxis, ...]

        if self._action_norm is not None:
            w_seed = self._action_norm({"actions": seed_np})["actions"]
            w_seed = transforms.pad_to_dim(w_seed, self._model_action_dim, axis=-1)
        else:
            w_seed = transforms.pad_to_dim(seed_np, self._model_action_dim, axis=-1)
        working = jnp.asarray(w_seed, dtype=jnp.float32)
        h = int(working.shape[0])
        if adaptive:
            max_r = wm_multi_rollout_adaptive_max_rounds(h=h, overlap=overlap, delta_idx=delta_idx)
            if overlap + (max_r - 1) * delta_idx > h:
                raise ValueError(
                    "wm_multi_rollout adaptive: internal max_r violates overlap+(N-1)*delta<=H "
                    f"(overlap={overlap}, max_r={max_r}, delta_idx={delta_idx}, H={h})"
                )
        else:
            max_r = num_rounds
            if overlap + (num_rounds - 1) * delta_idx > h:
                raise ValueError(
                    f"wm_multi_rollout: need overlap + (num_rounds-1)*delta_idx <= H, got "
                    f"overlap={overlap}, num_rounds={num_rounds}, delta_idx={delta_idx}, H={h}"
                )
        # AE merge uses ``start = overlap + r*delta``; require ``start < H`` on the last round r = max_r-1``.
        merge_round_cap = max(1, (h - overlap - 1) // delta_idx + 1)
        if max_r > merge_round_cap:
            logger.warning(
                "wm_multi_rollout: clamping max_r %d -> %d so overlap+r*delta < H on last merge (H=%d overlap=%d delta_idx=%d)",
                max_r,
                merge_round_cap,
                h,
                overlap,
                delta_idx,
            )
            max_r = merge_round_cap

        ae_src = meta.get("ae_proprio_source")
        if ae_src is not None:
            if ae_src not in ("prefix_t", "future_rollout", "vlash_last_action"):
                raise ValueError(
                    "openpi/async.ae_proprio_source must be 'prefix_t', 'future_rollout', or 'vlash_last_action', "
                    f"got {ae_src!r}"
                )
            effective_ae: AeProprioSource = ae_src
        else:
            effective_ae = self._ae_proprio_source

        delta = jnp.asarray([delta_t_wm], dtype=jnp.float32)
        logger.info(
            "wm_multi_rollout: adaptive=%s rounds=%d (max_r=%d) overlap=%d delta_t_wm=%s delta_idx=%d H=%d ae=%s",
            adaptive,
            num_rounds,
            max_r,
            overlap,
            delta_t_wm,
            delta_idx,
            h,
            effective_ae,
        )

        kappa_per_round: list[jax.Array] = []
        kappa0_f: float | None = None
        early_exec_len: int | None = None
        for r in range(max_r):
            self._inner._rng, k_wm, k_sample = jax.random.split(self._inner._rng, 3)
            if r == 0:
                if overlap > h:
                    raise RuntimeError(f"wm_multi_rollout: overlap {overlap} > H {h} at r={r}")
                ap_body = working[0:overlap]
            elif r >= 1 and mr.get("prev_chunk_tail") is not None:
                pct_np = np.asarray(mr["prev_chunk_tail"], dtype=np.float32)
                if pct_np.ndim != 2 or int(pct_np.shape[0]) != overlap:
                    raise ValueError(
                        "wm_multi_rollout.prev_chunk_tail must be 2D (overlap, A) with overlap="
                        f"{overlap}, got shape {pct_np.shape}"
                    )
                end_new = overlap + r * delta_idx
                if end_new > h:
                    raise RuntimeError(
                        f"wm_multi_rollout: prefix new-chunk slice end {end_new} > H={h} at r={r} overlap={overlap} delta_idx={delta_idx}"
                    )
                if self._action_norm is not None:
                    pt = self._action_norm({"actions": pct_np})["actions"]
                    pt = transforms.pad_to_dim(pt, self._model_action_dim, axis=-1)
                else:
                    pt = transforms.pad_to_dim(pct_np, self._model_action_dim, axis=-1)
                prev_rows = jnp.asarray(pt, dtype=jnp.float32)
                new_band = working[overlap:end_new]
                ap_body = jnp.concatenate([prev_rows, new_band], axis=0)
            else:
                lo0 = h - overlap
                if lo0 < 0 or lo0 > h:
                    raise RuntimeError(f"wm_multi_rollout: bad tail slice lo0={lo0} at r={r}")
                parts: list[jax.Array] = [working[lo0:h]]
                for j in range(1, r + 1):
                    lo_j = h - overlap - j * delta_idx
                    hi_j = h - overlap - (j - 1) * delta_idx
                    if lo_j < 0:
                        raise RuntimeError(
                            f"wm_multi_rollout: prefix block j={j} lo_j={lo_j}<0 at r={r} (H={h} overlap={overlap} delta_idx={delta_idx})"
                        )
                    if lo_j >= hi_j or hi_j > h:
                        raise RuntimeError(
                            f"wm_multi_rollout: bad prefix block j={j} lo={lo_j} hi={hi_j} at r={r} (H={h})"
                        )
                    parts.append(working[lo_j:hi_j])
                ap_body = jnp.concatenate(parts, axis=0)
            if ap_body.shape[0] == 0:
                raise RuntimeError(f"wm_multi_rollout: empty WM prefix at round r={r}")
            ap_j = ap_body[jnp.newaxis, ...]
            mask_j = jnp.ones((1, ap_body.shape[0]), dtype=jnp.bool_)
            mu, kappa = self._wm_forward(h_t, proprio, ap_j, mask_j, delta, k_wm)
            kappa_per_round.append(kappa.reshape(()))
            k_f = float(np.asarray(jax.device_get(kappa.reshape(()))))
            if adaptive:
                if r == 0:
                    kappa0_f = k_f
                elif kappa0_f is not None:
                    if low_replan:
                        if k_f < kappa0_f - kappa_th:
                            kappa_np = np.asarray(
                                jax.device_get(jnp.stack(kappa_per_round, axis=0)), dtype=np.float32
                            ).reshape(-1)
                            n_roll = int(overlap + int(r) * int(delta_idx))
                            if n_roll + overlap > int(h):
                                raise RuntimeError(
                                    "wm_multi_rollout low_replan: rollout_len+overlap exceeds H "
                                    f"(n_roll={n_roll}, overlap={overlap}, H={h}, r={r}, delta_idx={delta_idx})"
                                )
                            roll_w = working[:n_roll, ...]
                            glue_w = working[n_roll : n_roll + overlap, ...]
                            raise _LowKappaFullPi0Fallback(
                                wm_ae_rounds_completed=int(r),
                                kappa_per_round_np=kappa_np,
                                rollout_len=n_roll,
                                rollout_actions_model=np.asarray(jax.device_get(roll_w), dtype=np.float32),
                                glue_actions_model=np.asarray(jax.device_get(glue_w), dtype=np.float32),
                            )
                    elif k_f > kappa0_f + kappa_th:
                        early_exec_len = overlap + delta_idx * (r - 1)
                        break

            obs_for_ae = self._observation_for_ae_from_prefix(
                batched=batched,
                observation=observation,
                ap_j=ap_j,
                mask_j=mask_j,
                effective_ae=effective_ae,
            )
            if self._world_model is not None and _wm_verify.wm_inference_verify_mode() != "off":
                if adaptive and r == max_r - 1:
                    _wm_verify.run_wm_inference_verification(
                        pi0=self._pi0,
                        world_model=self._world_model,
                        observation=obs_for_ae,
                        mu=mu,
                    )
                elif not adaptive and r == num_rounds - 1:
                    _wm_verify.run_wm_inference_verification(
                        pi0=self._pi0,
                        world_model=self._world_model,
                        observation=obs_for_ae,
                        mu=mu,
                    )

            ae_actions = self._sample_with_future(
                k_sample,
                obs_for_ae,
                num_steps=self._inner._sample_kwargs.get("num_steps", 10),
                future_condition_tokens=mu,
            )
            start = overlap + r * delta_idx
            if start >= h:
                raise RuntimeError(f"wm_multi_rollout: merge start {start} >= H {h} at r={r}")
            working = working.at[start:h].set(ae_actions[0, start:h, ...])

        extras: dict[str, Any] = {"wm_stitch_n": int(len(kappa_per_round))}
        if adaptive:
            extras["adaptive_max_rounds"] = max_r
            extras["adaptive_early_stop"] = early_exec_len is not None
            extras["adaptive_exec_len"] = early_exec_len
            if early_exec_len is not None:
                el = int(early_exec_len)
                tail = el + overlap
                if tail < h:
                    working = working.at[tail:h].set(w_seed[tail:h, ...])

        return working[jnp.newaxis, ...], jnp.stack(kappa_per_round, axis=0), extras
