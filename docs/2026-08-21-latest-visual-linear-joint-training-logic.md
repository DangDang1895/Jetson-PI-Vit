# 最新视觉线性联合条件训练逻辑

## 1. 文档目的

本文固定当前 `latest-visual-residual` 分支下一阶段的训练定义。训练目标是让未来纠正模块使用旧大脑特征、从旧状态到当前状态之间已经执行的动作，以及当前最新视觉，重建当前条件特征。

本文只描述训练，不包含异步推理、动作块重叠、`OVERLAP`、动作裁剪、置信度门控或多轮 rollout。上述机制属于推理与动作交接层，不能改变本文的训练时间对齐。

当前选择的视觉条件结构为：

```python
visual_condition_kind = "linear_joint"
```

## 2. 时间与动作约定

机器人时间关系统一定义为：

```text
O_k --a_k--> O_(k+1)
```

对任意当前时刻 `t`，随机采样时间间隔：

```text
delta = Δ >= 1
source = t - Δ
target = t
```

从 `O_(t-Δ)` 到 `O_t` 实际执行的动作恰好为：

```python
action_prefix = [
    a_(t-Δ),
    a_(t-Δ+1),
    ...,
    a_(t-1),
]
```

动作前缀长度必须等于 `Δ`。不能包含 `a_t`，因为 `a_t` 会将状态从 `O_t` 推进到 `O_(t+1)`。

## 3. 训练目标

模型学习：

```text
旧大脑特征 H_(t-Δ)
+ 源本体状态 q_(t-Δ)
+ 已执行动作 [a_(t-Δ), ..., a_(t-1)]
+ 时间间隔 Δ
+ 当前最新视觉 V_t
-> 当前条件特征 μ_t ≈ C_t
```

数学形式为：

```text
μ_t = WM(
    H_(t-Δ),
    q_(t-Δ),
    actions[t-Δ:t],
    Δ,
    V_t,
)
```

其中：

```text
H_(t-Δ) = frozen_pi0.prefix_hidden_states(O_(t-Δ))
V_t     = frozen_siglip(O_t.images)
H_t     = frozen_pi0.prefix_hidden_states(O_t)
C_t     = frozen_token_reducer(H_t)
```

监督目标是压缩后的 `C_t`，不是完整的 `H_t`。

## 4. 数据样本构造

当前 `WorldModelLeRobotDataset` 已经具备正确的源观测、目标观测和动作前缀时间关系：

```python
delta = sample_integer(1, max_delta_t)

observation_source = per_step[0]
observation_target = per_step[delta]

action_prefix = actions_seq[:delta]
```

等价于将 `t-Δ` 平移为窗口索引 `0`：

```text
O_0 --a_0--> O_1 -- ... --a_(Δ-1)--> O_Δ
```

数据集不需要新增第三个观测，也不需要新增 `OVERLAP`。`_wm_obs_future` 在本训练中同时承担两个角色：

1. 提供当前最新图像 `O_t.images`；
2. 通过冻结的 Pi0 和 TokenReducer 生成监督目标 `C_t`。

动作 padding 和 mask 保持原逻辑：

```python
action_prefix_pad[:delta] = actions_seq[:delta]
prefix_mask[:delta] = True
prefix_mask[delta:] = False
```

## 5. 模型前向流程

### 5.1 固定源条件 token

```python
h_source = stop_gradient(
    pi0.prefix_hidden_states(observation_source)
)

c_source = wm.reduce_tokens(h_source)
```

`c_source` 表示旧状态，不被最新视觉提前覆写。

### 5.2 编码当前最新视觉

使用 Pi0 内共享且冻结的 SigLIP，对 `observation_target.images` 编码：

```python
latest_visual_tokens, latest_visual_mask = (
    pi0.encode_visual_tokens(observation_target)
)
```

随后由可训练的 `LatestVisualGlobalCondition` 使用 masked learned-query pooling 得到：

```python
v_latest = wm.latest_visual_global_condition.pool_visual(
    latest_visual_tokens,
    latest_visual_mask,
)
```

### 5.3 构造非视觉全局条件

```python
m = action_encoder(action_prefix_pad, prefix_mask)
p = proprio_encoder(observation_source.state)
e = time_encoder(delta_t)
g = concatenate([m, p, e])
```

第一版保留原 Jetson-PI 的 proprio 时间来源，使用源观测 `q_(t-Δ)`。本实验只新增当前视觉条件，不同时改变 proprio 的时间语义。

### 5.4 线性联合视觉条件

```python
g_joint = concatenate([g, v_latest])
u = linear_joint_u(g_joint)
film = linear_joint_film(g_joint)
gamma, beta = split(film)
```

随后保持原 FutureConditionHead 的 token 演化结构：

```python
x = gamma[:, None, :] * (c_source + u[:, None, :]) + beta[:, None, :]
x = block0(x)
x = block1(x)
mu = mean_head(x)
log_var = logvar_head(x)
```

## 6. 监督目标与损失

当前观测产生固定监督目标：

```python
h_target = stop_gradient(
    pi0.prefix_hidden_states(observation_target)
)

target = stop_gradient(
    wm.reduce_tokens(h_target)
)
```

第一版保留原异方差高斯 NLL，但停止 `log_var` 梯度：

```python
log_var_sg = stop_gradient(output.log_var)

loss = heteroscedastic_gaussian_nll(
    target,
    output.mu,
    log_var_sg,
)
```

本阶段不训练置信度，不使用 kappa，不用该损失进行 log-var 校准。

## 7. 冻结与训练范围

本实验使用：

```text
stage1_steps = 0
stage2_steps > 0
stage3_steps = 0
stage4_steps = 0
visual_condition_kind = linear_joint
```

参数分为“参与训练”“主动冻结”和“当前分支不使用”三类。

### 7.1 参与训练：完整有效的均值演化路径

| 模块 | 参数路径 | 参数来源 | 作用 |
| --- | --- | --- | --- |
| ActionEncoder | `wm/action_encoder/*` | 原 WM 参数，继续微调 | 将 `actions[t-Δ:t]` 和 mask 编码成动作摘要 `m` |
| ProprioEncoder | `wm/proprio_encoder/*` | 原 WM 参数，继续微调 | 将源观测的 proprio 编码成 `p` |
| TimeEncoder | `wm/time_encoder/*` | 原 WM 参数，继续微调 | 将 `delta_t` 编码成 `e` |
| 视觉 query | `wm/latest_visual_global_condition/visual_query` | 新参数 | 从最新视觉 tokens 中查询任务相关信息 |
| 视觉维度投影 | `wm/latest_visual_global_condition/visual_kv_proj/*` | 新参数；同维时模块可为空 | 将 SigLIP hidden dim 映射到 WM token dim |
| 视觉注意力 | `wm/latest_visual_global_condition/visual_attention/*` | 新参数 | learned query 对有效视觉 tokens 做 masked attention pooling |
| 视觉归一化 | `wm/latest_visual_global_condition/visual_norm/*` | 新参数 | 归一化全局视觉向量 `v_latest` |
| 联合内容投影 | `wm/future_head/joint_condition/u_proj/*` | 新结构，由旧 `u_proj` 热启动 | `[m,p,e,v_latest] -> u` |
| 联合 FiLM 投影 | `wm/future_head/joint_condition/film_proj/*` | 新结构，由旧 `film` 热启动 | `[m,p,e,v_latest] -> gamma,beta` |
| FutureTokenBlock 0 | `wm/future_head/block0/*` | 原 WM 参数，继续微调 | 对条件调制后的 token 建模 |
| FutureTokenBlock 1 | `wm/future_head/block1/*` | 原 WM 参数，继续微调 | 继续建模 token 间关系 |
| MeanHead | `wm/future_head/mean_head/*` | 原 WM 参数，继续微调 | 输出 `mu_t` 并拟合冻结目标 `C_t` |

完整可训练数据流为：

```text
actions -> ActionEncoder -> m -----------┐
proprio -> ProprioEncoder -> p ----------|
delta_t -> TimeEncoder -> e -------------|-> [m,p,e,v_latest]
visual tokens -> VisualPooling -> v_latest┘          |
                                                     v
                                          joint u_proj / film_proj
                                                     |
C_source -> FiLM -> block0 -> block1 -> mean_head -> mu_t
```

其中，真正替代原 Jetson-PI `u_proj(g)` 和 `film(g)` 的新联合参数是：

```text
wm/future_head/joint_condition/u_proj/*
wm/future_head/joint_condition/film_proj/*
```

### 7.2 主动冻结：参与前向但不更新

| 模块 | 参数路径 | 冻结原因 |
| --- | --- | --- |
| Pi0 主体 | `pi0/*` | 固定大脑和动作模型 |
| 共享 SigLIP | `pi0/PaliGemma/img/*` | 固定最新视觉 token 所在的特征空间 |
| PaliGemma LLM | `pi0/PaliGemma/llm/*` | 固定 `H_(t-Δ)` 和 `H_t` 的表示空间 |
| Pi0 Action Expert | `pi0/state_proj/*`、`pi0/action_*/*` 等 | 当前不训练动作损失 |
| TokenReducer | `wm/token_reducer/*` | 固定 `C_source` 与监督目标 `C_t` 的 token 空间 |
| reducer 投影 | `wm/reducer_vlm_to_token/*` | 属于固定 TokenReducer 输出空间 |
| LogVarHead | `wm/future_head/logvar_head/*` | 当前只训练均值 `mu`，不训练不确定性 |

对应的固定教师路径为：

```python
h_source = stop_gradient(pi0.prefix_hidden_states(O_source))
h_target = stop_gradient(pi0.prefix_hidden_states(O_target))
target = stop_gradient(token_reducer(h_target))
latest_visual_tokens = stop_gradient(siglip(O_target.images))
```

损失中的 log-var 同样停止梯度：

```python
log_var_sg = stop_gradient(output.log_var)
```

`logvar_head` 的权重保持不变。由于它接收的上游 `block0/block1` 输出会随训练变化，log-var 数值本身仍可能变化，但损失不会沿 log-var 路径反向传播。

### 7.3 当前分支不使用：从优化器排除的备用参数

| 参数 | 参数路径 | 不使用原因 |
| --- | --- | --- |
| 旧内容投影 | `wm/future_head/u_proj/*` | linear-joint 前向改用 `joint_condition.u_proj` |
| 旧 FiLM 投影 | `wm/future_head/film/*` | linear-joint 前向改用 `joint_condition.film_proj` |
| residual 内容投影 | `wm/latest_visual_global_condition/visual_u_proj/*` | 只供 residual 模式生成 `delta_u` |
| residual FiLM 投影 | `wm/latest_visual_global_condition/visual_film_proj/*` | 只供 residual 模式生成 `delta_film` |

旧 `future_head.u_proj/film` 只在加载 legacy checkpoint 后用于初始化新联合层：

```text
旧 [m,p,e] 权重
        |
        v copy
新 joint_condition 的 [m,p,e] 权重区域

新 joint_condition 的视觉权重区域 = 0
```

迁移完成后，旧投影不再出现在 linear-joint 的计算图中。将这些备用参数从优化器排除，不是冻结有效演化能力，而是避免为无梯度分支创建优化器状态或施加无意义的 weight decay。

### 7.4 训练范围结论

```text
训练：动作、本体、时间、最新视觉到 mu_t 的完整有效均值演化路径
冻结：Pi0、SigLIP、LLM、Action Expert、TokenReducer、logvar_head
排除：linear-joint 模式不会调用的旧投影与 residual 投影
```

## 8. 旧 checkpoint 热启动

训练从原 future correction module 热启动。旧模型的 `u_proj(g)` 和 `film(g)` 要迁移到新增联合层的非视觉权重区域：

```text
joint_u.kernel[g rows]      <- old_u.kernel
joint_u.kernel[visual rows] <- 0
joint_u.bias                <- old_u.bias

joint_film.kernel[g rows]      <- old_film.kernel
joint_film.kernel[visual rows] <- 0
joint_film.bias                <- old_film.bias
```

这样初始化后，对任意视觉输入都有：

```text
linear_joint([g, v]) = old_projection(g)
```

旧 WM 行为在训练第 0 步得到保留，随后整个均值路径可以联合适配当前视觉。

该迁移只用于从不含联合视觉层的旧 checkpoint 初始化新训练；恢复已经训练过的 linear-joint checkpoint 时禁止再次执行，否则会覆盖已经学到的视觉权重。

## 9. 具体样例

设：

```text
t = 3
Δ = 3
```

样本时间线为：

```text
O_0 --a_0--> O_1 --a_1--> O_2 --a_2--> O_3
```

训练输入：

```text
H_0
q_0
[a_0, a_1, a_2]
delta_t = 3
V_3 = SigLIP(O_3.images)
```

训练目标：

```text
C_3 = TokenReducer(H_3)
```

最终学习：

```text
WM(H_0, q_0, [a_0, a_1, a_2], 3, V_3) -> μ_3 ≈ C_3
```

## 10. 第一版训练配置

仅运行条件均值训练阶段：

```text
stage1_steps = 0
stage2_steps > 0
stage3_steps = 0
stage4_steps = 0
visual_condition_kind = linear_joint
```

从预训练 future correction module 初始化，使用 `train_step_stage1_no_logvar` 的扩展版本完成训练。

## 11. 验收条件

训练流程只有同时满足以下条件才算打通：

1. 数据集仍产生 `O_0`、`O_Δ` 和 `actions[:Δ]`，没有训练期 `OVERLAP`；
2. 最新视觉 tokens 来自 `observation_target`，源 `H` 和 proprio 来自 `observation_source`；
3. target 为冻结的 `TokenReducer(H_target)`；
4. `visual_condition_kind` 在训练配置、模型构造和 checkpoint 中均为 `linear_joint`；
5. Pi0、SigLIP、TokenReducer 和 `logvar_head` 没有梯度更新；
6. ActionEncoder、ProprioEncoder、TimeEncoder、视觉 pooling、LinearJointCondition、FutureTokenBlocks 和 `mean_head` 有有效梯度；
7. 旧 checkpoint 初始化后，视觉权重为零且初始输出与原 WM 一致；
8. 改变有效最新视觉 tokens 会改变 `mu`，改变被 mask 的视觉 tokens 不会改变 `mu`；
9. 单步 smoke training 能完成前向、反向和 checkpoint 保存；
10. 保存的 WM checkpoint 可以按 `linear_joint` 配置重新加载。

## 12. 明确不在本阶段处理的内容

```text
异步动作交接
OVERLAP 与动作块裁剪
跨请求 H/KV 缓存
置信度门控
kappa
log-var 校准
多轮 rollout
动作损失联合微调
推理服务接线
```

这些内容不能反向改变本文确定的训练时间关系。
