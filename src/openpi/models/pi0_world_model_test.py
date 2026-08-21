import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest
import dataclasses
from openpi.models import pi0_world_model as wm


def _small_config() -> wm.Pi0WorldModelConfig:
    return wm.Pi0WorldModelConfig(
        vlm_hidden_dim=6,
        token_dim=4,
        num_future_heads=2,
        ffn_multiplier=2,
    )


def test_latest_visual_global_condition_ignores_masked_tokens():
    module = wm.LatestVisualGlobalCondition(
        _small_config(),
        rngs=nnx.Rngs(0),
    )

    visual_tokens = jnp.arange(
        18,
        dtype=jnp.float32,
    ).reshape(1, 3, 6)
    visual_mask = jnp.array([[True, True, False]])

    # 只修改被 mask 的第三个 token。
    changed_tokens = visual_tokens.at[:, 2, :].set(1_000_000.0)

    pooled = module.pool_visual(
        visual_tokens,
        visual_mask,
    )
    pooled_changed = module.pool_visual(
        changed_tokens,
        visual_mask,
    )

    assert pooled.shape == (1, 4)
    np.testing.assert_allclose(
        pooled,
        pooled_changed,
        rtol=1e-5,
        atol=1e-5,
    )


def test_latest_visual_global_condition_returns_zero_for_empty_visual():
    module = wm.LatestVisualGlobalCondition(
        _small_config(),
        rngs=nnx.Rngs(0),
    )

    visual_tokens = jnp.arange(
        18,
        dtype=jnp.float32,
    ).reshape(1, 3, 6)
    visual_mask = jnp.zeros((1, 3), dtype=jnp.bool_)

    pooled = module.pool_visual(
        visual_tokens,
        visual_mask,
    )
    delta_u, delta_film = module(
        visual_tokens,
        visual_mask,
    )

    np.testing.assert_array_equal(
        pooled,
        jnp.zeros((1, 4), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        delta_u,
        jnp.zeros((1, 4), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        delta_film,
        jnp.zeros((1, 8), dtype=jnp.float32),
    )


def test_latest_visual_residual_outputs_are_zero_initialized():
    module = wm.LatestVisualGlobalCondition(
        _small_config(),
        rngs=nnx.Rngs(0),
    )

    visual_tokens = jnp.arange(
        18,
        dtype=jnp.float32,
    ).reshape(1, 3, 6)
    visual_mask = jnp.ones((1, 3), dtype=jnp.bool_)

    delta_u, delta_film = module(
        visual_tokens,
        visual_mask,
    )

    np.testing.assert_array_equal(
        delta_u,
        jnp.zeros((1, 4), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        delta_film,
        jnp.zeros((1, 8), dtype=jnp.float32),
    )

def test_future_condition_head_zero_visual_residual_preserves_output():
    cfg = _small_config()
    head = wm.FutureConditionHead(
        cfg,
        rngs=nnx.Rngs(1),
    )

    C_t = jnp.arange(
        8,
        dtype=jnp.float32,
    ).reshape(1, 2, 4)

    global_dim = (
        cfg.gru_hidden_dim
        + cfg.proprio_embed_dim
        + cfg.time_embed_dim
    )
    global_vec = jnp.linspace(
        -1.0,
        1.0,
        global_dim,
        dtype=jnp.float32,
    )[None, :]

    base_mu, base_log_var = head(
        C_t,
        global_vec,
    )

    residual_mu, residual_log_var = head(
        C_t,
        global_vec,
        delta_u=jnp.zeros((1, 4), dtype=jnp.float32),
        delta_film=jnp.zeros((1, 8), dtype=jnp.float32),
    )

    np.testing.assert_array_equal(
        residual_mu,
        base_mu,
    )
    np.testing.assert_array_equal(
        residual_log_var,
        base_log_var,
    )


def test_future_condition_head_applies_visual_residual():
    cfg = _small_config()
    head = wm.FutureConditionHead(
        cfg,
        rngs=nnx.Rngs(2),
    )

    C_t = jnp.arange(
        8,
        dtype=jnp.float32,
    ).reshape(1, 2, 4)

    global_dim = (
        cfg.gru_hidden_dim
        + cfg.proprio_embed_dim
        + cfg.time_embed_dim
    )
    global_vec = jnp.linspace(
        -1.0,
        1.0,
        global_dim,
        dtype=jnp.float32,
    )[None, :]

    base_mu, base_log_var = head(
        C_t,
        global_vec,
    )

    delta_u = jnp.array(
        [[0.1, -0.2, 0.3, -0.4]],
        dtype=jnp.float32,
    )
    delta_film = jnp.array(
        [[
            0.2,
            -0.1,
            0.3,
            -0.2,
            0.4,
            -0.3,
            0.1,
            -0.4,
        ]],
        dtype=jnp.float32,
    )

    residual_mu, residual_log_var = head(
        C_t,
        global_vec,
        delta_u=delta_u,
        delta_film=delta_film,
    )

    assert not np.allclose(
        residual_mu,
        base_mu,
    )
    assert not np.allclose(
        residual_log_var,
        base_log_var,
    )
def _world_model_inputs(
    cfg: wm.Pi0WorldModelConfig,
) -> dict[str, jnp.ndarray]:
    return {
        "H_t": jnp.arange(
            18,
            dtype=jnp.float32,
        ).reshape(1, 3, 6),
        "proprio": jnp.zeros(
            (1, cfg.proprio_dim),
            dtype=jnp.float32,
        ),
        "action_prefix": jnp.zeros(
            (1, 2, cfg.action_dim),
            dtype=jnp.float32,
        ),
        "prefix_mask": jnp.array(
            [[True, True]],
            dtype=jnp.bool_,
        ),
        "delta_t": jnp.array(
            [1.0],
            dtype=jnp.float32,
        ),
    }


def test_empty_visual_stays_zero_after_residual_bias_changes():
    cfg = _small_config()
    module = wm.LatestVisualGlobalCondition(
        cfg,
        rngs=nnx.Rngs(3),
    )

    module.visual_u_proj.bias.value = jnp.ones(
        (cfg.token_dim,),
        dtype=jnp.float32,
    )
    module.visual_film_proj.bias.value = jnp.ones(
        (2 * cfg.token_dim,),
        dtype=jnp.float32,
    )

    visual_tokens = jnp.arange(
        18,
        dtype=jnp.float32,
    ).reshape(1, 3, 6)
    visual_mask = jnp.zeros((1, 3), dtype=jnp.bool_)

    delta_u, delta_film = module(
        visual_tokens,
        visual_mask,
    )

    np.testing.assert_array_equal(
        delta_u,
        jnp.zeros((1, 4), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        delta_film,
        jnp.zeros((1, 8), dtype=jnp.float32),
    )


def test_world_model_zero_initialized_visual_path_preserves_output():
    cfg = _small_config()
    model = wm.Pi0FutureWorldModel(
        cfg,
        rngs=nnx.Rngs(4),
    )
    inputs = _world_model_inputs(cfg)

    latest_visual_tokens = jnp.arange(
        24,
        dtype=jnp.float32,
    ).reshape(1, 4, 6)
    latest_visual_mask = jnp.ones(
        (1, 4),
        dtype=jnp.bool_,
    )

    base_output = model(
        **inputs,
        rngs=nnx.Rngs(5),
    )
    visual_output = model(
        **inputs,
        latest_visual_tokens=latest_visual_tokens,
        latest_visual_mask=latest_visual_mask,
        rngs=nnx.Rngs(5),
    )

    np.testing.assert_array_equal(
        visual_output.current_tokens,
        base_output.current_tokens,
    )
    np.testing.assert_array_equal(
        visual_output.mu,
        base_output.mu,
    )
    np.testing.assert_array_equal(
        visual_output.log_var,
        base_output.log_var,
    )


def test_world_model_applies_activated_visual_residual_without_changing_C_t():
    cfg = _small_config()
    model = wm.Pi0FutureWorldModel(
        cfg,
        rngs=nnx.Rngs(6),
    )
    inputs = _world_model_inputs(cfg)

    # 模拟视觉残差层经过训练后的状态。
    model.latest_visual_global_condition.visual_u_proj.bias.value = (
        jnp.array(
            [0.1, -0.2, 0.3, -0.4],
            dtype=jnp.float32,
        )
    )
    model.latest_visual_global_condition.visual_film_proj.bias.value = (
        jnp.array(
            [
                0.2,
                -0.1,
                0.3,
                -0.2,
                0.4,
                -0.3,
                0.1,
                -0.4,
            ],
            dtype=jnp.float32,
        )
    )

    latest_visual_tokens = jnp.arange(
        24,
        dtype=jnp.float32,
    ).reshape(1, 4, 6)
    latest_visual_mask = jnp.ones(
        (1, 4),
        dtype=jnp.bool_,
    )

    base_output = model(
        **inputs,
        rngs=nnx.Rngs(7),
    )
    visual_output = model(
        **inputs,
        latest_visual_tokens=latest_visual_tokens,
        latest_visual_mask=latest_visual_mask,
        rngs=nnx.Rngs(7),
    )

    # 最新视觉不能覆写源状态 C_t。
    np.testing.assert_array_equal(
        visual_output.current_tokens,
        base_output.current_tokens,
    )

    # 但是必须通过 delta_u/delta_film 改变未来预测。
    assert not np.allclose(
        visual_output.mu,
        base_output.mu,
    )


def test_world_model_requires_visual_tokens_and_mask_together():
    cfg = _small_config()
    model = wm.Pi0FutureWorldModel(
        cfg,
        rngs=nnx.Rngs(8),
    )
    inputs = _world_model_inputs(cfg)

    latest_visual_tokens = jnp.arange(
        24,
        dtype=jnp.float32,
    ).reshape(1, 4, 6)

    with pytest.raises(
        ValueError,
        match="latest_visual_tokens and latest_visual_mask must be provided together",
    ):
        model(
            **inputs,
            latest_visual_tokens=latest_visual_tokens,
            rngs=nnx.Rngs(9),
        )
def test_linear_joint_condition_uses_latest_visual():
    cfg = dataclasses.replace(
        _small_config(),
        visual_condition_kind="linear_joint",
    )
    model = wm.Pi0FutureWorldModel(
        cfg,
        rngs=nnx.Rngs(10),
    )
    inputs = _world_model_inputs(cfg)

    visual_mask = jnp.ones(
        (1, 4),
        dtype=jnp.bool_,
    )

    visual_tokens_a = jnp.arange(
        24,
        dtype=jnp.float32,
    ).reshape(1, 4, 6)

    visual_tokens_b = visual_tokens_a.at[
        0,
        0,
        :,
    ].set(
        jnp.array(
            [
                100.0,
                -50.0,
                25.0,
                -12.0,
                6.0,
                -3.0,
            ],
            dtype=jnp.float32,
        )
    )

    output_a = model(
        **inputs,
        latest_visual_tokens=visual_tokens_a,
        latest_visual_mask=visual_mask,
        rngs=nnx.Rngs(11),
    )
    output_b = model(
        **inputs,
        latest_visual_tokens=visual_tokens_b,
        latest_visual_mask=visual_mask,
        rngs=nnx.Rngs(11),
    )

    # 最新视觉不能改变源状态 C_t。
    np.testing.assert_array_equal(
        output_a.current_tokens,
        output_b.current_tokens,
    )

    # 但在线性联合条件中，不同视觉必须产生不同预测。
    assert not np.allclose(
        output_a.mu,
        output_b.mu,
    )


def test_linear_joint_condition_requires_latest_visual():
    cfg = dataclasses.replace(
        _small_config(),
        visual_condition_kind="linear_joint",
    )
    model = wm.Pi0FutureWorldModel(
        cfg,
        rngs=nnx.Rngs(12),
    )
    inputs = _world_model_inputs(cfg)

    with pytest.raises(
        ValueError,
        match=(
            "visual_condition_kind='linear_joint' "
            "requires latest visual inputs"
        ),
    ):
        model(
            **inputs,
            rngs=nnx.Rngs(13),
        )

def test_nonlinear_joint_condition_models_cross_condition_interaction():
    cfg = dataclasses.replace(
        _small_config(),
        visual_condition_kind="nonlinear_joint",
    )
    condition = wm.NonlinearJointCondition(
        cfg,
        rngs=nnx.Rngs(14),
    )

    global_dim = (
        cfg.gru_hidden_dim
        + cfg.proprio_embed_dim
        + cfg.time_embed_dim
    )

    g0 = jnp.zeros(
        (1, global_dim),
        dtype=jnp.float32,
    )
    g1 = g0.at[0, 0].set(1.0)
    g1 = g1.at[0, 1].set(-0.5)

    v0 = jnp.zeros(
        (1, cfg.token_dim),
        dtype=jnp.float32,
    )
    v1 = jnp.array(
        [[0.2, -0.4, 0.6, -0.8]],
        dtype=jnp.float32,
    )

    u00, _ = condition(g0, v0)
    u10, _ = condition(g1, v0)
    u01, _ = condition(g0, v1)
    u11, _ = condition(g1, v1)

    # 纯线性映射满足：
    # f(g1, v1) - f(g1, v0) - f(g0, v1) + f(g0, v0) == 0
    #
    # 非线性联合模块应当能够产生非零交互项。
    interaction = (
        u11
        - u10
        - u01
        + u00
    )

    assert not np.allclose(
        interaction,
        jnp.zeros_like(interaction),
        rtol=1e-5,
        atol=1e-6,
    )


def test_nonlinear_joint_condition_uses_latest_visual():
    cfg = dataclasses.replace(
        _small_config(),
        visual_condition_kind="nonlinear_joint",
    )
    model = wm.Pi0FutureWorldModel(
        cfg,
        rngs=nnx.Rngs(15),
    )
    inputs = _world_model_inputs(cfg)

    visual_mask = jnp.ones(
        (1, 4),
        dtype=jnp.bool_,
    )

    visual_tokens_a = jnp.arange(
        24,
        dtype=jnp.float32,
    ).reshape(1, 4, 6)

    visual_tokens_b = visual_tokens_a.at[
        0,
        1,
        :,
    ].set(
        jnp.array(
            [
                -100.0,
                50.0,
                -25.0,
                12.0,
                -6.0,
                3.0,
            ],
            dtype=jnp.float32,
        )
    )

    output_a = model(
        **inputs,
        latest_visual_tokens=visual_tokens_a,
        latest_visual_mask=visual_mask,
        rngs=nnx.Rngs(16),
    )
    output_b = model(
        **inputs,
        latest_visual_tokens=visual_tokens_b,
        latest_visual_mask=visual_mask,
        rngs=nnx.Rngs(16),
    )

    np.testing.assert_array_equal(
        output_a.current_tokens,
        output_b.current_tokens,
    )

    assert not np.allclose(
        output_a.mu,
        output_b.mu,
    )


def test_nonlinear_joint_condition_requires_latest_visual():
    cfg = dataclasses.replace(
        _small_config(),
        visual_condition_kind="nonlinear_joint",
    )
    model = wm.Pi0FutureWorldModel(
        cfg,
        rngs=nnx.Rngs(17),
    )
    inputs = _world_model_inputs(cfg)

    with pytest.raises(
        ValueError,
        match=(
            "visual_condition_kind='nonlinear_joint' "
            "requires latest visual inputs"
        ),
    ):
        model(
            **inputs,
            rngs=nnx.Rngs(18),
        )
def test_linear_joint_legacy_warm_start_preserves_base_projection():
    cfg = dataclasses.replace(
        _small_config(),
        visual_condition_kind="linear_joint",
    )
    model = wm.Pi0FutureWorldModel(
        cfg,
        rngs=nnx.Rngs(19),
    )
    head = model.future_head

    global_dim = (
        cfg.gru_hidden_dim
        + cfg.proprio_embed_dim
        + cfg.time_embed_dim
    )

    global_vec = jnp.linspace(
        -1.0,
        1.0,
        global_dim,
        dtype=jnp.float32,
    )[None, :]

    visual_vec = jnp.linspace(
        1.0,
        -1.0,
        cfg.token_dim,
        dtype=jnp.float32,
    )[None, :]

    # 旧 FutureConditionHead 的输出。
    expected_u = head.u_proj(global_vec)
    expected_film = head.film(global_vec)

    # 将旧投影权重迁移到新的 linear-joint 层。
    wm.initialize_linear_joint_from_legacy_head(model)

    actual_u, actual_film = head.joint_condition(
        global_vec,
        visual_vec,
    )

    # 视觉权重为零时，新联合层必须保持旧模型输出。
    np.testing.assert_allclose(
        actual_u,
        expected_u,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        actual_film,
        expected_film,
        rtol=1e-6,
        atol=1e-6,
    )

    # 新增的视觉权重区域必须全零。
    np.testing.assert_array_equal(
        head.joint_condition.u_proj.kernel.value[global_dim:],
        jnp.zeros(
            (cfg.token_dim, cfg.token_dim),
            dtype=head.joint_condition.u_proj.kernel.value.dtype,
        ),
    )
    np.testing.assert_array_equal(
        head.joint_condition.film_proj.kernel.value[global_dim:],
        jnp.zeros(
            (cfg.token_dim, 2 * cfg.token_dim),
            dtype=head.joint_condition.film_proj.kernel.value.dtype,
        ),
    )