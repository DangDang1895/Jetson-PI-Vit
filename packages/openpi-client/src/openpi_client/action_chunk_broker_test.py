import numpy as np

from openpi_client import action_chunk_broker


class _FakePerStepPolicy:
    def __init__(self, action_horizon: int):
        self.action_horizon = action_horizon
        self.requests = []
        self.reset_count = 0

    def infer(self, obs: dict) -> dict:
        self.requests.append(obs)

        meta = obs["openpi/async"]
        control_step = int(meta["control_step"])

        # 请求 t 生成：
        # [100*t, 100*t+1, ..., 100*t+9]
        action_ids = (
            control_step * 100
            + np.arange(
                self.action_horizon,
                dtype=np.float32,
            )
        )

        actions = np.stack(
            [
                action_ids,
                -action_ids,
            ],
            axis=-1,
        )

        return {
            "actions": actions,
            "openpi/current_control_step": (
                control_step
            ),
        }

    def reset(self) -> None:
        self.reset_count += 1


def _observation(step: int) -> dict:
    return {
        "observation/state": np.asarray(
            [float(step)],
            dtype=np.float32,
        ),
    }


def test_per_step_cerebellum_uses_tail_supplement():
    horizon = 10
    policy = _FakePerStepPolicy(horizon)

    broker = (
        action_chunk_broker.PerStepCerebellumBroker(
            policy=policy,
            action_horizon=horizon,
        )
    )

    try:
        broker.reset()

        executed_ids = []

        for step in range(11):
            out = broker.infer(
                _observation(step)
            )
            executed_ids.append(
                int(out["actions"][0])
            )

        # 初始完整动作块保持不变。
        assert executed_ids[:10] == list(
            range(10)
        )

        # control_step=1 的新动作块是：
        # [100, 101, ..., 109]
        #
        # Jetson-PI 尾部补充只追加最后的 109，
        # 因而第 11 个执行动作应当是 109，
        # 而不是用 100 替换旧动作。
        assert executed_ids[10] == 109

    finally:
        broker.shutdown()


def test_per_step_requests_send_executed_actions():
    horizon = 10
    policy = _FakePerStepPolicy(horizon)

    broker = (
        action_chunk_broker.PerStepCerebellumBroker(
            policy=policy,
            action_horizon=horizon,
        )
    )

    try:
        broker.reset()

        for step in range(4):
            broker.infer(
                _observation(step)
            )

    finally:
        # 等待最后一个后台请求完成，
        # 确保 requests 已经记录完整。
        broker.shutdown()

    assert len(policy.requests) == 4

    for step, request in enumerate(
        policy.requests
    ):
        meta = request["openpi/async"]

        assert meta["per_step_cerebellum"] is True
        assert meta["episode_id"] == 0
        assert meta["control_step"] == step

        if step == 0:
            assert (
                "last_executed_action"
                not in meta
            )
        else:
            # 在 O_step 到达前，实际执行的是
            # 初始动作块中的 a_{step-1}。
            expected = np.asarray(
                [
                    float(step - 1),
                    -float(step - 1),
                ],
                dtype=np.float32,
            )

            np.testing.assert_array_equal(
                meta["last_executed_action"],
                expected,
            )


def test_per_step_reset_starts_new_episode():
    horizon = 10
    policy = _FakePerStepPolicy(horizon)

    broker = (
        action_chunk_broker.PerStepCerebellumBroker(
            policy=policy,
            action_horizon=horizon,
        )
    )

    try:
        broker.reset()
        broker.infer(_observation(0))
        broker.infer(_observation(1))

        broker.reset()
        broker.infer(_observation(0))

    finally:
        broker.shutdown()

    episode_and_step = [
        (
            request["openpi/async"]["episode_id"],
            request["openpi/async"]["control_step"],
        )
        for request in policy.requests
    ]

    assert episode_and_step == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]

    new_episode_meta = policy.requests[
        2
    ]["openpi/async"]

    assert (
        "last_executed_action"
        not in new_episode_meta
    )

def test_per_step_replace_after_one_step():
    horizon = 10
    policy = _FakePerStepPolicy(horizon)

    broker = (
        action_chunk_broker.PerStepCerebellumBroker(
            policy=policy,
            action_horizon=horizon,
            handover_mode=(
                "replace_after_one_step"
            ),
        )
    )

    try:
        broker.reset()

        executed_ids = []

        for step in range(4):
            out = broker.infer(
                _observation(step)
            )
            executed_ids.append(
                int(out["actions"][0])
            )

    finally:
        broker.shutdown()

    # t0：初始 Pi0 动作 a0
    # t1：小脑推理期间继续执行旧 a1
    # t2：请求 t1 返回 [100,...,109]，
    #     丢弃 100，由 101 立即接管
    # t3：请求 t2 返回 [200,...,209]，
    #     丢弃 200，由 201 立即接管
    assert executed_ids == [
        0,
        1,
        101,
        201,
    ]

    # O3 请求中发送的上一条实际执行动作
    # 必须是 101，而不是旧动作 2。
    request_t3 = policy.requests[3]
    meta_t3 = request_t3["openpi/async"]

    np.testing.assert_array_equal(
        meta_t3["last_executed_action"],
        np.asarray(
            [101.0, -101.0],
            dtype=np.float32,
        ),
    )