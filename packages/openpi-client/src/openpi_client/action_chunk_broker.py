from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
from collections import deque
from collections.abc import Callable
from typing import Dict, Optional

import numpy as np
import tree
from typing_extensions import Literal, override

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


def wm_stitch_n_from_policy_full(full: Dict) -> int | None:  # noqa: UP006
    """``openpi/wm_stitch_n`` from server (multi-round WM); ``None`` if absent."""
    v = full.get("openpi/wm_stitch_n")
    if v is None:
        return None
    return int(np.asarray(v, dtype=np.int64).reshape(()))


@dataclasses.dataclass(frozen=True)
class PrefetchContext:
    """Passed to ``prefetch_obs_hook`` when ``AsyncActionChunkBroker`` starts a background infer."""

    chunk_actions: np.ndarray
    chunk_start_step: int
    async_trigger_step: int
    action_horizon: int
    overlap_skip: int
    wm_stitch_n: int | None = None
    wm_overlap_exec_band: np.ndarray | None = None


class ActionChunkBroker(_base_policy.BasePolicy):
    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


def _snapshot_observation(obs: Dict) -> Dict:  # noqa: UP006
    def _copy(x):
        if isinstance(x, np.ndarray):
            return np.array(x, copy=True)
        return x

    return tree.map_structure(_copy, obs)


class AsyncActionChunkBroker(_base_policy.BasePolicy):
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        *,
        async_trigger_step: int = 4,
        overlap_skip: int = 0,
        chunk_exec_steps: int | None = None,
        allow_trigger_at_chunk_start: bool = False,
        prefetch_obs_hook: Callable[[Dict, PrefetchContext], Dict] | None = None,
        observation_state_chunk_index: int | None = None,
        prefetch_handover_true_state: bool = False,
        handover_snapshot_hook: Callable[[Dict, PrefetchContext], Dict] | None = None,
        handover_merged_state_fn: Optional[Callable[[Dict, Dict, np.ndarray], np.ndarray]] = None,
    ) -> None:
        if not (0 <= overlap_skip < action_horizon):
            raise ValueError(
                f"overlap_skip must be in [0, action_horizon), got overlap_skip={overlap_skip}, "
                f"action_horizon={action_horizon}."
            )
        if allow_trigger_at_chunk_start:
            valid_trigger = 0 <= async_trigger_step <= action_horizon
            trigger_msg = (
                "async_trigger_step must satisfy 0 <= async_trigger_step <= action_horizon "
                f"(got async_trigger_step={async_trigger_step}, action_horizon={action_horizon})."
            )
        else:
            valid_trigger = 0 < async_trigger_step <= action_horizon
            trigger_msg = (
                "async_trigger_step must satisfy 0 < async_trigger_step <= action_horizon "
                f"(got async_trigger_step={async_trigger_step}, action_horizon={action_horizon})."
            )
        if not valid_trigger:
            raise ValueError(trigger_msg)
        if chunk_exec_steps is not None and not (1 <= chunk_exec_steps <= action_horizon):
            raise ValueError(
                "chunk_exec_steps must be in [1, action_horizon] or None, "
                f"got chunk_exec_steps={chunk_exec_steps}, action_horizon={action_horizon}."
            )
        if observation_state_chunk_index is not None and not (0 <= observation_state_chunk_index < action_horizon):
            raise ValueError(
                "observation_state_chunk_index must be in [0, action_horizon) or None, "
                f"got {observation_state_chunk_index}, action_horizon={action_horizon}."
            )
        if prefetch_handover_true_state and prefetch_obs_hook is not None:
            raise ValueError("prefetch_handover_true_state=True is incompatible with prefetch_obs_hook.")
        if handover_snapshot_hook is not None and not prefetch_handover_true_state:
            raise ValueError("handover_snapshot_hook requires prefetch_handover_true_state=True.")
        if handover_merged_state_fn is not None and not prefetch_handover_true_state:
            raise ValueError("handover_merged_state_fn requires prefetch_handover_true_state=True.")

        self._policy = policy
        self._action_horizon = action_horizon
        self._async_trigger_step = async_trigger_step
        self._overlap_skip = overlap_skip
        self._chunk_exec_steps = chunk_exec_steps
        self._allow_trigger_at_chunk_start = allow_trigger_at_chunk_start
        self._prefetch_obs_hook = prefetch_obs_hook
        self._observation_state_chunk_index = observation_state_chunk_index
        self._prefetch_handover_true_state = prefetch_handover_true_state
        self._handover_snapshot_hook = handover_snapshot_hook
        self._handover_merged_state_fn = handover_merged_state_fn

        self._cur_step: int = 0
        self._chunk_start_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None
        self._pending_future: concurrent.futures.Future | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._first_chunk_after_reset = True
        self._state_history: list[np.ndarray] = []
        self._saved_visual_snap_at_k: Dict | None = None
        self._saved_handover_action_prefix: np.ndarray | None = None

    def _drain_pending(self) -> None:
        if self._pending_future is None:
            return
        fut = self._pending_future
        self._pending_future = None
        try:
            fut.result(timeout=600)
        except Exception:
            pass

    def _build_obs_in(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._observation_state_chunk_index is None:
            return obs
        st = np.array(obs["observation/state"], copy=True)
        self._state_history.append(st)
        idx = self._observation_state_chunk_index
        if len(self._state_history) > idx:
            obs_in = dict(obs)
            obs_in["observation/state"] = np.array(self._state_history[idx], copy=True)
            return obs_in
        return obs

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        exec_limit = self._chunk_exec_steps if self._chunk_exec_steps is not None else self._action_horizon
        if self._last_results is None:
            if self._observation_state_chunk_index is not None:
                self._state_history = []
            if self._pending_future is not None:
                self._last_results = self._pending_future.result()
                self._pending_future = None
                if self._first_chunk_after_reset or self._chunk_exec_steps is not None:
                    self._cur_step = 0
                    self._chunk_start_step = 0
                else:
                    self._cur_step = self._overlap_skip
                    self._chunk_start_step = self._overlap_skip

        obs_in = self._build_obs_in(obs)

        if self._last_results is None:
            if self._prefetch_handover_true_state and self._saved_visual_snap_at_k is not None:
                merged = _snapshot_observation(self._saved_visual_snap_at_k)
                if self._handover_merged_state_fn is not None:
                    if self._saved_handover_action_prefix is None:
                        raise RuntimeError("handover merge: missing saved action prefix (broker internal error)")
                    merged["observation/state"] = np.asarray(
                        self._handover_merged_state_fn(merged, obs_in, self._saved_handover_action_prefix),
                        dtype=np.float32,
                    )
                else:
                    merged["observation/state"] = np.array(obs_in["observation/state"], copy=True)
                infer_obs = merged
                self._saved_visual_snap_at_k = None
                self._saved_handover_action_prefix = None
            else:
                infer_obs = obs_in
            self._last_results = self._policy.infer(infer_obs)
            if self._first_chunk_after_reset or self._chunk_exec_steps is not None:
                self._cur_step = 0
                self._chunk_start_step = 0
                self._first_chunk_after_reset = False
            else:
                self._cur_step = self._overlap_skip
                self._chunk_start_step = self._overlap_skip

        if (
            not self._prefetch_handover_true_state
            and self._pending_future is None
            and self._cur_step == self._async_trigger_step
        ):
            snap = _snapshot_observation(obs_in)
            if self._prefetch_obs_hook is not None:
                actions = self._last_results["actions"]
                if not isinstance(actions, np.ndarray):
                    actions = np.asarray(actions)
                ctx = PrefetchContext(
                    chunk_actions=actions,
                    chunk_start_step=self._chunk_start_step,
                    async_trigger_step=self._async_trigger_step,
                    action_horizon=self._action_horizon,
                    overlap_skip=self._overlap_skip,
                )
                snap = self._prefetch_obs_hook(snap, ctx)
            self._pending_future = self._executor.submit(self._policy.infer, snap)
        elif self._prefetch_handover_true_state and self._cur_step == self._async_trigger_step:
            actions = self._last_results["actions"]
            if not isinstance(actions, np.ndarray):
                actions = np.asarray(actions)
            snap = _snapshot_observation(obs_in)
            if self._handover_snapshot_hook is not None:
                ctx = PrefetchContext(
                    chunk_actions=actions,
                    chunk_start_step=self._chunk_start_step,
                    async_trigger_step=self._async_trigger_step,
                    action_horizon=self._action_horizon,
                    overlap_skip=self._overlap_skip,
                )
                snap = self._handover_snapshot_hook(snap, ctx)
            self._saved_visual_snap_at_k = snap
            self._saved_handover_action_prefix = np.array(
                np.asarray(actions[self._async_trigger_step : self._action_horizon]),
                dtype=np.float32,
                copy=True,
            )

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1
        if self._cur_step >= exec_limit:
            self._last_results = None
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._drain_pending()
        self._last_results = None
        self._cur_step = 0
        self._chunk_start_step = 0
        self._first_chunk_after_reset = True
        self._state_history = []
        self._saved_visual_snap_at_k = None
        self._saved_handover_action_prefix = None

    def shutdown(self) -> None:
        self._drain_pending()
        self._executor.shutdown(wait=False)


def _split_leading_horizon(full: Dict, H: int) -> list[Dict]:  # noqa: UP006
    """Split policy outputs whose leading dim is ``H`` into ``H`` per-step dicts."""

    def one(i: int) -> Dict:
        def sel(x):
            if isinstance(x, np.ndarray) and x.ndim >= 1 and int(x.shape[0]) == H:
                # np.asarray(..., copy=) requires NumPy>=2; use np.array for 1.x compat.
                return np.array(x[i, ...], copy=True)
            return x

        return tree.map_structure(sel, full)

    return [one(i) for i in range(H)]

class PerStepCerebellumBroker(
    _base_policy.BasePolicy
):
    """每执行一步触发一次小脑推理，并按 Jetson-PI 方式补充动作尾部。"""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        *,
        async_key: str = "openpi/async",
        handover_mode: Literal[
            "tail_append",
            "replace_after_one_step",
        ] = "tail_append",
    ) -> None:
        if action_horizon < 2:
            raise ValueError(
                "action_horizon must be at least 2."
            )
        if handover_mode not in (
            "tail_append",
            "replace_after_one_step",
        ):
            raise ValueError(
                "Unknown per-step handover_mode="
                f"{handover_mode!r}. Expected "
                "'tail_append' or "
                "'replace_after_one_step'."
            )

        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._async_key = async_key
        self._handover_mode = handover_mode

        self._buf: deque[Dict] = deque()
        self._pending_future: (
            concurrent.futures.Future | None
        ) = None
        self._pending_control_step: int | None = None

        self._executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            )
        )

        self._episode_id = -1
        self._control_step = 0
        self._last_executed_action: (
            np.ndarray | None
        ) = None

    def _build_request(
        self,
        obs: Dict,
    ) -> Dict:
        request = _snapshot_observation(obs)

        meta = {
            "per_step_cerebellum": True,
            "episode_id": self._episode_id,
            "control_step": self._control_step,
        }

        if self._control_step > 0:
            if self._last_executed_action is None:
                raise RuntimeError(
                    "Missing last_executed_action for "
                    f"control_step={self._control_step}."
                )

            meta["last_executed_action"] = np.array(
                self._last_executed_action,
                dtype=np.float32,
                copy=True,
            )

        request[self._async_key] = meta
        return request

    def _validate_full(
        self,
        full: Dict,
    ) -> None:
        if "actions" not in full:
            raise RuntimeError(
                "Policy output does not contain actions."
            )

        actions = np.asarray(full["actions"])

        if (
            actions.ndim < 1
            or actions.shape[0]
            != self._action_horizon
        ):
            raise RuntimeError(
                "Expected an action chunk with leading "
                f"horizon={self._action_horizon}, "
                f"got shape={actions.shape}."
            )

    def _accept_initial_chunk(
        self,
        full: Dict,
    ) -> None:
        self._validate_full(full)

        parts = _split_leading_horizon(
            full,
            self._action_horizon,
        )
        self._buf.extend(parts)

    def _accept_handover_result(
        self,
        full: Dict,
    ) -> None:
        self._validate_full(full)

        if self._pending_control_step is not None:
            returned_step = full.get(
                "openpi/current_control_step"
            )

            if returned_step is not None:
                returned_step = int(
                    np.asarray(returned_step).reshape(())
                )

                if (
                    returned_step
                    != self._pending_control_step
                ):
                    raise RuntimeError(
                        "Per-step response is out of order: "
                        f"expected control_step="
                        f"{self._pending_control_step}, "
                        f"got {returned_step}."
                    )

        # parts = _split_leading_horizon(
        #     full,
        #     self._action_horizon,
        # )

        # # 固定一控制步推理延迟，并沿用 Jetson-PI：
        # # 保留旧 buffer，只补充新动作块最后一个动作。
        # self._buf.append(parts[-1])

        parts = _split_leading_horizon(
            full,
            self._action_horizon,
        )

        if self._handover_mode == "tail_append":
            # Jetson-PI 稳定性基线：
            # 保留旧 buffer，只补充新动作块最后一步。
            self._buf.append(parts[-1])

        elif (
            self._handover_mode
            == "replace_after_one_step"
        ):
            # 请求期间已经执行了一步旧动作。
            # 新动作块的第 0 项对应已经过去的时刻，
            # 因此丢弃第 0 项，并让第 1 项立即接管。
            self._buf.clear()
            self._buf.extend(parts[1:])

    def _consume_pending(self) -> None:
        if self._pending_future is None:
            return

        full = self._pending_future.result(
            timeout=600
        )

        self._pending_future = None
        self._accept_handover_result(full)
        self._pending_control_step = None

    def _start_per_step_infer(
        self,
        obs: Dict,
    ) -> None:
        if self._pending_future is not None:
            raise RuntimeError(
                "A per-step inference request is "
                "already running."
            )

        request = self._build_request(obs)
        self._pending_control_step = (
            self._control_step
        )
        self._pending_future = (
            self._executor.submit(
                self._policy.infer,
                request,
            )
        )

    @override
    def infer(
        self,
        obs: Dict,
    ) -> Dict:
        # 上一步启动的小脑推理应当在当前控制步接管前完成。
        self._consume_pending()

        if self._control_step == 0:
            # 初始缓存为空，第一次请求同步产生
            # H0/KV0 和完整的正常 Pi0 动作块。
            initial_request = self._build_request(obs)
            initial_full = self._policy.infer(
                initial_request
            )
            self._accept_initial_chunk(
                initial_full
            )
        else:
            # O_t 到达后启动小脑推理；随后返回旧 buffer
            # 中当前要执行的动作，实现推理与 env.step 重叠。
            self._start_per_step_infer(obs)

        if not self._buf:
            raise RuntimeError(
                "Per-step action buffer is empty."
            )

        out = self._buf.popleft()

        if "actions" not in out:
            raise RuntimeError(
                "Per-step output does not contain actions."
            )

        # 这是客户端接下来真正交给环境执行的动作。
        # 下一次请求将它作为 a_{t} 发送给服务端。
        self._last_executed_action = np.array(
            out["actions"],
            dtype=np.float32,
            copy=True,
        ).reshape(-1)

        self._control_step += 1
        return out

    @override
    def reset(self) -> None:
        if self._pending_future is not None:
            try:
                self._pending_future.result(
                    timeout=600
                )
            except Exception:
                logger.exception(
                    "Pending per-step inference failed "
                    "during reset."
                )

        self._pending_future = None
        self._pending_control_step = None
        self._buf.clear()

        self._episode_id += 1
        self._control_step = 0
        self._last_executed_action = None

        self._policy.reset()

    def shutdown(self) -> None:
        if self._pending_future is not None:
            try:
                self._pending_future.result(
                    timeout=600
                )
            except Exception:
                logger.exception(
                    "Pending per-step inference failed "
                    "during shutdown."
                )

        self._pending_future = None
        self._executor.shutdown(wait=False)

class AsyncActionBufferBroker(_base_policy.BasePolicy):

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        *,
        async_trigger_step: int = 4,
        overlap_skip: int = 0,
        chunk_exec_steps: int | None = None,
        allow_trigger_at_chunk_start: bool = False,
        prefetch_obs_hook: Callable[[Dict, PrefetchContext], Dict] | None = None,
        observation_state_chunk_index: int | None = None,
        wm_infer_complete_hook: Callable[[Dict], None] | None = None,
        low_replan_two_phase_fn: Callable[[Dict, Dict], Dict] | None = None,
    ) -> None:
        if chunk_exec_steps is not None:
            raise ValueError("AsyncActionBufferBroker does not support chunk_exec_steps; pass None.")
        if not (0 <= overlap_skip < action_horizon):
            raise ValueError(
                f"overlap_skip must be in [0, action_horizon), got overlap_skip={overlap_skip}, "
                f"action_horizon={action_horizon}."
            )
        if allow_trigger_at_chunk_start:
            valid_trigger = 0 <= async_trigger_step <= action_horizon
            trigger_msg = (
                "async_trigger_step must satisfy 0 <= async_trigger_step <= action_horizon "
                f"(got async_trigger_step={async_trigger_step}, action_horizon={action_horizon})."
            )
        else:
            valid_trigger = 0 < async_trigger_step <= action_horizon
            trigger_msg = (
                "async_trigger_step must satisfy 0 < async_trigger_step <= action_horizon "
                f"(got async_trigger_step={async_trigger_step}, action_horizon={action_horizon})."
            )
        if not valid_trigger:
            raise ValueError(trigger_msg)
        if observation_state_chunk_index is not None and not (0 <= observation_state_chunk_index < action_horizon):
            raise ValueError(
                "observation_state_chunk_index must be in [0, action_horizon) or None, "
                f"got {observation_state_chunk_index}, action_horizon={action_horizon}."
            )

        self._policy = policy
        self._action_horizon = action_horizon
        self._async_trigger_step = int(async_trigger_step)
        self._overlap_skip = int(overlap_skip)
        self._prefetch_obs_hook = prefetch_obs_hook
        self._observation_state_chunk_index = observation_state_chunk_index
        self._wm_infer_complete_hook = wm_infer_complete_hook
        self._low_replan_two_phase_fn = low_replan_two_phase_fn

        self._buf: deque[Dict] = deque()
        self._last_infer_full: np.ndarray | None = None
        self._last_wm_stitch_n: int | None = None
        self._last_wm_overlap_exec_band: np.ndarray | None = None
        self._pending_future: concurrent.futures.Future | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._state_history: list[np.ndarray] = []
        # κ-adaptive early WM stop (``openpi/wm_adaptive_early_stop``): enqueue only ``[0:L)``, stash
        # ``[L:L+overlap)`` for glue before ``[overlap:H)`` of the next infer; prefetch at obs after L pops.
        self._wm_adapt_pops_left: int = 0
        self._wm_adapt_overlap_parts: list[Dict] | None = None
        self._wm_adapt_prefetch_on_next: bool = False
        self._wm_adapt_glue_next: bool = False
        #: Snapshot of the dict passed to ``policy.infer`` for the in-flight prefetch (for WM low-replan two-phase).
        self._prefetch_two_phase_obs_snap: Dict | None = None

    def _maybe_apply_low_replan_two_phase(self, obs_snap: Dict, full: Dict) -> Dict:  # noqa: UP006
        if not bool(full.get("openpi/wm_low_replan_two_phase")):
            return full
        if self._low_replan_two_phase_fn is None:
            raise RuntimeError(
                "Policy returned openpi/wm_low_replan_two_phase=True but AsyncActionBufferBroker was constructed "
                "with low_replan_two_phase_fn=None."
            )
        return dict(self._low_replan_two_phase_fn(obs_snap, full))

    def _invoke_wm_infer_complete_hook(self, full: Dict) -> None:  # noqa: UP006
        if self._wm_infer_complete_hook is None:
            return
        try:
            self._wm_infer_complete_hook(full)
        except Exception:
            logger.exception("wm_infer_complete_hook raised")

    def _drain_pending(self) -> None:
        if self._pending_future is None:
            return
        fut = self._pending_future
        self._pending_future = None
        try:
            fut.result(timeout=600)
        except Exception:
            pass

    def _build_obs_in(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._observation_state_chunk_index is None:
            return obs
        st = np.array(obs["observation/state"], copy=True)
        self._state_history.append(st)
        idx = self._observation_state_chunk_index
        if len(self._state_history) > idx:
            obs_in = dict(obs)
            obs_in["observation/state"] = np.array(self._state_history[idx], copy=True)
            return obs_in
        return obs

    def _append_from_full_first(self, full: Dict) -> None:  # noqa: UP006
        parts = _split_leading_horizon(full, self._action_horizon)
        self._buf.extend(parts)
        acts = full.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=np.float32)
            H, O = self._action_horizon, self._overlap_skip
            self._last_infer_full = np.array(a, dtype=np.float32, copy=True)
            self._last_wm_stitch_n = wm_stitch_n_from_policy_full(full)
            self._last_wm_overlap_exec_band = np.array(a[H - O : H], dtype=np.float32, copy=True)

    def _append_from_full_merged(self, full: Dict) -> None:  # noqa: UP006
        parts = _split_leading_horizon(full, self._action_horizon)
        for j in range(self._overlap_skip, self._action_horizon):
            self._buf.append(parts[j])
        acts = full.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=np.float32)
            H, O = self._action_horizon, self._overlap_skip
            self._last_infer_full = np.array(a, dtype=np.float32, copy=True)
            self._last_wm_stitch_n = wm_stitch_n_from_policy_full(full)
            self._last_wm_overlap_exec_band = np.array(a[H - O : H], dtype=np.float32, copy=True)

    def _append_adaptive_wm_early(self, full: Dict) -> None:  # noqa: UP006
        """Partial enqueue + stash overlap band for glue-merge after prefetch (obs after L pops)."""
        H = self._action_horizon
        o = self._overlap_skip
        if not bool(full.get("openpi/wm_adaptive_early_stop")):
            raise RuntimeError("internal: _append_adaptive_wm_early without adaptive flag")
        L = int(full["openpi/wm_adaptive_exec_len"])
        if not (0 < L < H):
            raise RuntimeError(f"wm_adaptive_exec_len must be in (0,H), got L={L}, H={H}")
        if L + o > H:
            raise RuntimeError(f"wm_adaptive L+overlap must be <= H, got L={L}, overlap={o}, H={H}")
        parts = _split_leading_horizon(full, H)
        for i in range(L):
            self._buf.append(parts[i])
        self._wm_adapt_overlap_parts = [parts[i] for i in range(L, L + o)]
        self._wm_adapt_pops_left = L
        self._wm_adapt_prefetch_on_next = False
        self._wm_adapt_glue_next = False
        acts = full.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=np.float32)
            self._last_infer_full = np.array(a, dtype=np.float32, copy=True)
            self._last_wm_stitch_n = wm_stitch_n_from_policy_full(full)
            self._last_wm_overlap_exec_band = np.array(a[L : L + o], dtype=np.float32, copy=True)

    def _append_adaptive_glue_merge(self, full: Dict) -> None:  # noqa: UP006
        """After early-stop prefetch: old ``[L:L+overlap)`` at front, then new ``[overlap:H)`` at end."""
        if self._wm_adapt_overlap_parts is None:
            raise RuntimeError("internal: adaptive glue without stashed overlap parts")
        H = self._action_horizon
        o = self._overlap_skip
        parts = _split_leading_horizon(full, H)
        for p in reversed(self._wm_adapt_overlap_parts):
            self._buf.appendleft(p)
        for j in range(o, H):
            self._buf.append(parts[j])
        self._wm_adapt_overlap_parts = None
        self._wm_adapt_pops_left = 0
        acts = full.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=np.float32)
            H, O = self._action_horizon, self._overlap_skip
            self._last_infer_full = np.array(a, dtype=np.float32, copy=True)
            self._last_wm_stitch_n = wm_stitch_n_from_policy_full(full)
            self._last_wm_overlap_exec_band = np.array(a[H - O : H], dtype=np.float32, copy=True)

    def _consume_infer_full(self, full: Dict) -> None:  # noqa: UP006
        if bool(full.get("openpi/wm_adaptive_early_stop")):
            self._append_adaptive_wm_early(full)
        else:
            self._append_from_full_merged(full)

    def _blocking_refill(self, obs: Dict) -> None:  # noqa: UP006
        obs_in = self._build_obs_in(obs)
        full = self._policy.infer(obs_in)
        full = self._maybe_apply_low_replan_two_phase(_snapshot_observation(obs_in), full)
        self._invoke_wm_infer_complete_hook(full)
        if bool(full.get("openpi/wm_adaptive_early_stop")):
            self._append_adaptive_wm_early(full)
        elif self._last_infer_full is not None:
            self._append_from_full_merged(full)
        else:
            self._append_from_full_first(full)

    def _start_prefetch(self, obs: Dict) -> None:  # noqa: UP006
        obs_in = self._build_obs_in(obs)
        snap = _snapshot_observation(obs_in)
        if self._prefetch_obs_hook is not None:
            if self._last_infer_full is None:
                raise RuntimeError("AsyncActionBufferBroker: prefetch_hook requires prior infer (last_infer_full).")
            ctx = PrefetchContext(
                chunk_actions=self._last_infer_full,
                chunk_start_step=0,
                async_trigger_step=self._async_trigger_step,
                action_horizon=self._action_horizon,
                overlap_skip=self._overlap_skip,
                wm_stitch_n=self._last_wm_stitch_n,
                wm_overlap_exec_band=self._last_wm_overlap_exec_band,
            )
            snap = self._prefetch_obs_hook(snap, ctx)
        self._prefetch_two_phase_obs_snap = _snapshot_observation(snap)
        self._pending_future = self._executor.submit(self._policy.infer, snap)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._wm_adapt_prefetch_on_next:
            self._start_prefetch(obs)
            self._wm_adapt_prefetch_on_next = False
            self._wm_adapt_glue_next = True

        if self._pending_future is not None and self._pending_future.done():
            full = self._pending_future.result(timeout=600)
            self._pending_future = None
            snap = self._prefetch_two_phase_obs_snap
            self._prefetch_two_phase_obs_snap = None
            obs_in = self._build_obs_in(obs)
            full = self._maybe_apply_low_replan_two_phase(
                snap if snap is not None else _snapshot_observation(obs_in), full
            )
            self._invoke_wm_infer_complete_hook(full)
            if self._wm_adapt_glue_next:
                self._append_adaptive_glue_merge(full)
                self._wm_adapt_glue_next = False
            else:
                self._consume_infer_full(full)

        while len(self._buf) == 0:
            if self._pending_future is not None:
                full = self._pending_future.result(timeout=600)
                self._pending_future = None
                snap = self._prefetch_two_phase_obs_snap
                self._prefetch_two_phase_obs_snap = None
                obs_in = self._build_obs_in(obs)
                full = self._maybe_apply_low_replan_two_phase(
                    snap if snap is not None else _snapshot_observation(obs_in), full
                )
                self._invoke_wm_infer_complete_hook(full)
                if self._wm_adapt_glue_next:
                    self._append_adaptive_glue_merge(full)
                    self._wm_adapt_glue_next = False
                else:
                    self._consume_infer_full(full)
            else:
                self._blocking_refill(obs)

        need = self._action_horizon - self._async_trigger_step
        if (
            self._pending_future is None
            and len(self._buf) == need
            and not self._wm_adapt_prefetch_on_next
            and self._wm_adapt_overlap_parts is None
        ):
            self._start_prefetch(obs)

        out = self._buf.popleft()
        if self._wm_adapt_pops_left > 0:
            self._wm_adapt_pops_left -= 1
            if self._wm_adapt_pops_left == 0:
                self._wm_adapt_prefetch_on_next = True
        return out

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._drain_pending()
        self._buf.clear()
        self._last_infer_full = None
        self._last_wm_stitch_n = None
        self._last_wm_overlap_exec_band = None
        self._pending_future = None
        self._state_history = []
        self._wm_adapt_pops_left = 0
        self._wm_adapt_overlap_parts = None
        self._wm_adapt_prefetch_on_next = False
        self._wm_adapt_glue_next = False
        self._prefetch_two_phase_obs_snap = None

    def shutdown(self) -> None:
        self._drain_pending()
        self._executor.shutdown(wait=False)
