# Latest Visual Linear-Joint Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the world-model mean path to predict current reduced condition tokens from an older brain state, executed actions, source proprioception, elapsed steps, and visual tokens from the current observation.

**Architecture:** Keep Pi0, shared SigLIP, the action expert, TokenReducer, and log-variance head frozen. Extract target-observation image tokens through frozen SigLIP, pool them with `LatestVisualGlobalCondition`, combine the pooled vector with `[action, proprio, time]` through `LinearJointCondition`, and train the remaining WM mean path against frozen `TokenReducer(H_target)`.

**Tech Stack:** Python 3.11, JAX, Flax NNX, Optax, pytest, Tyro CLI, Bash.

---

## File map

- Modify `src/openpi/models/pi0.py`: expose frozen SigLIP image tokens and masks without running the VLM language stream.
- Modify `src/openpi/models/pi0_world_model.py`: add a tested legacy-to-linear-joint warm-start helper.
- Modify `src/openpi/models/pi0_world_model_test.py`: verify warm-start equality and zero visual columns.
- Modify `src/openpi/training/world_model_training.py`: add training configuration, direct WM checkpoint loading, and pass target visual tokens into the no-logvar training step.
- Modify `src/openpi/training/world_model_training_four_stage.py`: construct a linear-joint WM, use the direct pretrained checkpoint, and select an exact mean-path trainable filter.
- Create `scripts/train_wm_libero_spatial_linear_joint.sh`: provide a dedicated stage-2-only launcher without changing the original four-stage recipe.
- Do not modify `src/openpi/training/world_model_data.py`: its existing `O_0`, `O_delta`, `actions[:delta]` alignment is already correct.

### Task 1: Test and add legacy linear-joint warm start

**Files:**
- Modify: `src/openpi/models/pi0_world_model_test.py`
- Modify: `src/openpi/models/pi0_world_model.py`

- [ ] **Step 1: Add the failing test**

Append to `src/openpi/models/pi0_world_model_test.py`:

```python
def test_linear_joint_legacy_warm_start_preserves_base_projection():
    cfg = dataclasses.replace(
        _small_config(),
        visual_condition_kind="linear_joint",
    )
    model = wm.Pi0FutureWorldModel(cfg, rngs=nnx.Rngs(19))
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

    expected_u = head.u_proj(global_vec)
    expected_film = head.film(global_vec)

    wm.initialize_linear_joint_from_legacy_head(model)

    actual_u, actual_film = head.joint_condition(
        global_vec,
        visual_vec,
    )
    np.testing.assert_allclose(actual_u, expected_u, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_film, expected_film, rtol=1e-6, atol=1e-6)

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
```

- [ ] **Step 2: Run the focused test and verify failure**

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  src/openpi/models/pi0_world_model_test.py::test_linear_joint_legacy_warm_start_preserves_base_projection
```

Expected: failure because `initialize_linear_joint_from_legacy_head` does not exist.

- [ ] **Step 3: Add the warm-start helper**

Add immediately before `load_pi0_future_world_model` in `src/openpi/models/pi0_world_model.py`:

```python
def initialize_linear_joint_from_legacy_head(
    model: Pi0FutureWorldModel,
) -> None:
    """Initialize a linear-joint head from the loaded legacy g-only projections.

    Call this exactly once after loading a checkpoint that predates
    LinearJointCondition. Never call it when resuming a trained linear-joint
    checkpoint because it deliberately zeros the learned visual columns.
    """
    if model.cfg.visual_condition_kind != "linear_joint":
        raise ValueError(
            "legacy linear-joint initialization requires "
            "visual_condition_kind='linear_joint'"
        )

    head = model.future_head
    joint = head.joint_condition
    if not isinstance(joint, LinearJointCondition):
        raise TypeError("linear_joint model does not contain LinearJointCondition")

    old_u_kernel = head.u_proj.kernel.value
    old_film_kernel = head.film.kernel.value
    visual_dim = model.cfg.token_dim

    joint.u_proj.kernel.value = jnp.concatenate(
        [
            old_u_kernel,
            jnp.zeros(
                (visual_dim, old_u_kernel.shape[1]),
                dtype=old_u_kernel.dtype,
            ),
        ],
        axis=0,
    )
    joint.u_proj.bias.value = head.u_proj.bias.value

    joint.film_proj.kernel.value = jnp.concatenate(
        [
            old_film_kernel,
            jnp.zeros(
                (visual_dim, old_film_kernel.shape[1]),
                dtype=old_film_kernel.dtype,
            ),
        ],
        axis=0,
    )
    joint.film_proj.bias.value = head.film.bias.value
```

- [ ] **Step 4: Run all world-model tests**

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  src/openpi/models/pi0_world_model_test.py
```

Expected: 15 tests pass; existing unrelated warnings may remain.

### Task 2: Expose current-observation visual tokens from Pi0

**Files:**
- Modify: `src/openpi/models/pi0.py`

- [ ] **Step 1: Add the visual-token method**

Add immediately before `prefix_hidden_states` in `src/openpi/models/pi0.py`:

```python
    @at.typecheck
    def encode_visual_tokens(
        self,
        observation: _model.Observation,
    ) -> tuple[
        at.Float[at.Array, "b v d"],
        at.Bool[at.Array, "b v"],
    ]:
        """Encode only observation images with the shared frozen SigLIP."""
        observation = _model.preprocess_observation(
            None,
            observation,
            train=False,
        )

        visual_tokens = []
        visual_masks = []
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(
                observation.images[name],
                train=False,
            )
            visual_tokens.append(image_tokens)
            visual_masks.append(
                einops.repeat(
                    observation.image_masks[name],
                    "b -> b v",
                    v=image_tokens.shape[1],
                )
            )

        if not visual_tokens:
            raise ValueError("observation contains no images")

        return (
            jnp.concatenate(visual_tokens, axis=1),
            jnp.concatenate(visual_masks, axis=1),
        )
```

- [ ] **Step 2: Compile the model file**

```bash
.venv/bin/python -m py_compile src/openpi/models/pi0.py
```

Expected: exit code 0.

### Task 3: Pass target visual tokens into stage-2 condition training

**Files:**
- Modify: `src/openpi/training/world_model_training.py`

- [ ] **Step 1: Extend `train_step_stage1_no_logvar`**

Inside `loss_fn`, immediately after computing `h_t`, `h_f`, and `target`, add:

```python
        latest_visual_tokens, latest_visual_mask = (
            bundle.pi0.encode_visual_tokens(observation_f)
        )
        latest_visual_tokens = jax.lax.stop_gradient(
            latest_visual_tokens
        )
```

Then extend the existing `bundle.wm(...)` call with:

```python
            latest_visual_tokens=latest_visual_tokens,
            latest_visual_mask=latest_visual_mask,
```

The resulting loss body must preserve:

```python
        log_var_sg = jax.lax.stop_gradient(out.log_var)
        return wm_mod.heteroscedastic_gaussian_nll(
            target,
            out.mu,
            log_var_sg,
        )
```

- [ ] **Step 2: Confirm the data loader remains unchanged**

Do not modify `src/openpi/training/world_model_data.py`. Verify the existing lines still read:

```python
out0 = per_step[0]
out_d = per_step[delta]
prefix = actions_seq[:delta]
```

- [ ] **Step 3: Compile the training module**

```bash
.venv/bin/python -m py_compile \
  src/openpi/training/world_model_training.py
```

Expected: exit code 0.

### Task 4: Add explicit linear-joint configuration and legacy checkpoint loading

**Files:**
- Modify: `src/openpi/training/world_model_training.py`
- Modify: `src/openpi/training/world_model_training_four_stage.py`

- [ ] **Step 1: Add configuration fields**

In `WorldModelTrainConfig`, next to `action_encoder_kind`, add:

```python
    visual_condition_kind: wm_mod.VisualConditionKind = "residual"
```

Next to the existing WM initialization fields, add:

```python
    wm_init_from_checkpoint: str | None = None
```

- [ ] **Step 2: Add a direct-checkpoint bundle loader**

Add after `load_bundle_with_wm_export` in `world_model_training.py`:

```python
def load_bundle_with_wm_checkpoint(
    bundle: Pi0WorldModelTrainBundle,
    checkpoint_dir: str | pathlib.Path,
    *,
    initialize_linear_joint_from_legacy: bool,
) -> Pi0WorldModelTrainBundle:
    loaded_wm = wm_mod.load_pi0_future_world_model(
        checkpoint_dir,
        config=bundle.wm.cfg,
    )
    if initialize_linear_joint_from_legacy:
        wm_mod.initialize_linear_joint_from_legacy_head(loaded_wm)
    return Pi0WorldModelTrainBundle(bundle.pi0, loaded_wm)
```

- [ ] **Step 3: Pass the condition kind into both WM constructors**

In both `world_model_training.py` and `world_model_training_four_stage.py`, add to `Pi0WorldModelConfig(...)`:

```python
        visual_condition_kind=cfg.visual_condition_kind,
```

- [ ] **Step 4: Add the direct initialization branch in four-stage training**

In `world_model_training_four_stage.py`, before processing any resume or initialization branch, add:

```python
    init_mode_count = sum(
        value is not None
        for value in (
            cfg.resume_four_stage_orbax_ckpt_root,
            cfg.resume_wm_export_step,
            cfg.wm_init_from_export_step,
            cfg.wm_init_from_checkpoint,
        )
    )
    if init_mode_count > 1:
        raise ValueError(
            "resume/init modes are mutually exclusive: choose exactly one of "
            "resume_four_stage_orbax_ckpt_root, resume_wm_export_step, "
            "wm_init_from_export_step, or wm_init_from_checkpoint"
        )
```

Then, after the existing `wm_init_from_export_step` branch, add:

```python
    elif cfg.wm_init_from_checkpoint is not None:
        if cfg.stage1_steps != 0:
            raise ValueError(
                "wm_init_from_checkpoint requires stage1_steps=0"
            )
        bundle = wm_train.load_bundle_with_wm_checkpoint(
            bundle,
            cfg.wm_init_from_checkpoint,
            initialize_linear_joint_from_legacy=(
                cfg.visual_condition_kind == "linear_joint"
            ),
        )
        params = nnx.state(bundle)
        skip_stage1_for_wm_export = True
        logger.info(
            "Loaded legacy WM init from %s with visual_condition_kind=%s",
            cfg.wm_init_from_checkpoint,
            cfg.visual_condition_kind,
        )
```

Do not execute `initialize_linear_joint_from_legacy_head` in resume branches.

- [ ] **Step 5: Compile both training modules**

```bash
.venv/bin/python -m py_compile \
  src/openpi/training/world_model_training.py \
  src/openpi/training/world_model_training_four_stage.py
```

Expected: exit code 0.

### Task 5: Freeze exactly the stable representation and unused heads

**Files:**
- Modify: `src/openpi/training/world_model_training_four_stage.py`

- [ ] **Step 1: Add the linear-joint mean-path filter**

Add after `trainable_filter_four_stage2_no_logvar_head`:

```python
def trainable_filter_four_stage2_linear_joint_mean() -> nnx.filterlib.Filter:
    """Train the active linear-joint WM mean path only."""
    return nnx.All(
        nnx.Param,
        nnx_utils.PathRegex(r"wm/.*"),
        nnx.Not(nnx_utils.PathRegex(r"wm/token_reducer/.*")),
        nnx.Not(nnx_utils.PathRegex(r"wm/reducer_vlm_to_token/.*")),
        nnx.Not(nnx_utils.PathRegex(r"wm/.*/logvar_head/.*")),
        nnx.Not(nnx_utils.PathRegex(r"wm/future_head/u_proj/.*")),
        nnx.Not(nnx_utils.PathRegex(r"wm/future_head/film/.*")),
        nnx.Not(
            nnx_utils.PathRegex(
                r"wm/latest_visual_global_condition/visual_u_proj/.*"
            )
        ),
        nnx.Not(
            nnx_utils.PathRegex(
                r"wm/latest_visual_global_condition/visual_film_proj/.*"
            )
        ),
    )
```

- [ ] **Step 2: Select the filter only for linear-joint stage 2**

Replace the non-logvar stage-2 selection with:

```python
    if cfg.wm_logvar_only_finetune:
        s2_flt = wm_train.trainable_filter_wm_logvar_head_only()
        s2_tag = "four_s2_wm_logvar_only"
    elif cfg.visual_condition_kind == "linear_joint":
        s2_flt = trainable_filter_four_stage2_linear_joint_mean()
        s2_tag = "four_s2_linear_joint_mean"
    else:
        s2_flt = trainable_filter_four_stage2_no_logvar_head()
        s2_tag = "four_s2_wm_lcond"
```

This keeps Pi0, SigLIP, the action expert, TokenReducer, reducer projection, log-var head, and inactive condition projections outside the optimizer.

- [ ] **Step 3: Inspect actual selected parameter paths**

Before the training loop, temporarily or permanently log `state.params.filter(s2_flt).flat_state()` paths. Verify that active paths include:

```text
wm/action_encoder/*
wm/proprio_encoder/*
wm/time_encoder/*
wm/latest_visual_global_condition/visual_query
wm/latest_visual_global_condition/visual_attention/*
wm/latest_visual_global_condition/visual_norm/*
wm/future_head/joint_condition/*
wm/future_head/block0/*
wm/future_head/block1/*
wm/future_head/mean_head/*
```

and exclude all frozen paths listed above.

### Task 6: Add a dedicated training launcher

**Files:**
- Create: `scripts/train_wm_libero_spatial_linear_joint.sh`

- [ ] **Step 1: Create the launcher**

Create `scripts/train_wm_libero_spatial_linear_joint.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${PI0_CHECKPOINT:?Set PI0_CHECKPOINT to the pi05_libero checkpoint}"
: "${WM_INIT_FROM_CHECKPOINT:?Set WM_INIT_FROM_CHECKPOINT to the legacy future_correction_module directory}"
: "${OPENPI_LIBERO_LOCAL_DATASET_DIR:?Set OPENPI_LIBERO_LOCAL_DATASET_DIR}"
: "${PY:?Set PY to the project Python executable}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"

export STAGE2_STEPS="${STAGE2_STEPS:-15000}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
export EXP_NAME="${EXP_NAME:-wm_linear_joint_pi05_libero_spatial_s2_${STAGE2_STEPS}_bs${BATCH_SIZE}_${RUN_TS}}"
LOG="${WM_LOG_FILE:-${REPO_ROOT}/logs/${EXP_NAME}.log}"
mkdir -p "$(dirname "${LOG}")" "${REPO_ROOT}/checkpoints"

echo "LOG=${LOG}"
echo "EXP_NAME=${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PI0_CHECKPOINT=${PI0_CHECKPOINT}"
echo "WM_INIT_FROM_CHECKPOINT=${WM_INIT_FROM_CHECKPOINT}"
echo "STAGE2_STEPS=${STAGE2_STEPS} BATCH_SIZE=${BATCH_SIZE}"

set -o pipefail
"${PY}" -u "${REPO_ROOT}/scripts/train_world_model_four_stage.py" \
  --data-config-name pi05_libero \
  --assets-base-dir "${OPENPI_LIBERO_LOCAL_DATASET_DIR}" \
  --checkpoint-base-dir "${REPO_ROOT}/checkpoints" \
  --exp-name "${EXP_NAME}" \
  --pi0-checkpoint "${PI0_CHECKPOINT}" \
  --wm-init-from-checkpoint "${WM_INIT_FROM_CHECKPOINT}" \
  --visual-condition-kind linear_joint \
  --libero-task-index-min 0 \
  --libero-task-index-max 10 \
  --stage1-steps 0 \
  --stage2-steps "${STAGE2_STEPS}" \
  --stage3-steps 0 \
  --stage4-steps 0 \
  --max-delta-t 10 \
  --handover-horizon-min 10 \
  --handover-horizon-max 10 \
  --token-reducer-kind learned_cross_attn \
  --action-encoder-kind transformer_block \
  --wm-gru-hidden-dim 384 \
  --wm-gru-num-layers 3 \
  --wm-num-future-heads 8 \
  --wm-num-reducer-heads 8 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --seed 42 \
  --no-wandb-enabled \
  2>&1 | stdbuf -oL -eL tee -a "${LOG}"

exit "${PIPESTATUS[0]}"
```

Do not pass `--full-llm-trainable`, stage-3 joint-loss flags, confidence flags, or inference overlap flags.

The current general script is named spatial but selects task indices `30..39`; the dedicated spatial launcher must use `0..9`, represented by Tyro bounds `0` and `10`.

- [ ] **Step 2: Check shell syntax**

```bash
bash -n scripts/train_wm_libero_spatial_linear_joint.sh
```

Expected: exit code 0.

### Task 7: Verify training integration

**Files:**
- Test all modified files.

- [ ] **Step 1: Run static checks**

```bash
git diff --check
.venv/bin/python -m py_compile \
  src/openpi/models/pi0.py \
  src/openpi/models/pi0_world_model.py \
  src/openpi/training/world_model_training.py \
  src/openpi/training/world_model_training_four_stage.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run model tests**

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  src/openpi/models/pi0_world_model_test.py
```

Expected: all tests pass.

- [ ] **Step 3: Run a one-step GPU smoke train**

```bash
cd /chencen005/users/hs/Works/Jetson-PI-Vit

export PI0_CHECKPOINT=/chencen005/users/hs/Works/Jetson-PI-Vit/checkpoints/Pretrained/pi05_libero
export WM_INIT_FROM_CHECKPOINT=/chencen005/users/hs/Works/Jetson-PI-Vit/checkpoints/Pretrained/future_correction_module
export OPENPI_LIBERO_LOCAL_DATASET_DIR=/chencen005/datasets/libero
export PY=/chencen005/users/hs/Works/Jetson-PI-Vit/.venv/bin/python
export CUDA_VISIBLE_DEVICES=4
export STAGE2_STEPS=1
export BATCH_SIZE=1
export NUM_WORKERS=0
export SAVE_INTERVAL=1

bash scripts/train_wm_libero_spatial_linear_joint.sh
```

Expected log evidence:

```text
visual_condition_kind=linear_joint
stage_tag=four_s2_linear_joint_mean
latest visual token shape is non-empty
finite loss at the first completed step
world_model_step_1 checkpoint saved
```

- [ ] **Step 4: Load the saved checkpoint with linear-joint config**

Construct `Pi0WorldModelConfig` with the same dimensions and `visual_condition_kind="linear_joint"`, then call `load_pi0_future_world_model` on `world_model_step_1`. Do not run legacy warm-start initialization on this trained checkpoint.

- [ ] **Step 5: Commit only after the user reviews the diff**

```bash
git status --short
git diff --check
git diff -- \
  src/openpi/models/pi0.py \
  src/openpi/models/pi0_world_model.py \
  src/openpi/models/pi0_world_model_test.py \
  src/openpi/training/world_model_training.py \
  src/openpi/training/world_model_training_four_stage.py \
  scripts/train_wm_libero_spatial_linear_joint.sh \
  docs/2026-08-21-latest-visual-linear-joint-training-logic.md
```

Do not commit until the user explicitly requests it.
