# TA-VLA 仿真强化 baseline

这是一个只新增文件的 baseline 包。它不改变现有 `RealSimEnv`、当前
`rl_games/train.py`、远程 Server client 或已有任务注册。正式实验前先用
`pure_smoke.py` 和单环境 residual PPO 验证。

## 已包含模块

- `tavla_baseline/wrench.py`：固定的
  `wrench_base -> -wrench_base -> affine adapter -> effort`，顺序固定为
  `[Fx,Fy,Fz,Tx,Ty,Tz]`。adapter 的 JSON 是数值真值，`.pt` 只作为配套
  checkpoint 文件存在性和复现实验标识，不执行不可信 pickle。
- `observations.py`：actor/critic 分离和特权状态泄漏检查。
- `rewards.py`：成功、深度进展、对准、力/力矩超阈值、动作跳变、碰撞、超时，
  以及连续保持成功和安全终止。
- `randomization.py`：标称、位姿、接触、传感器/延迟四阶段 curriculum。
- `states.py`：JSON 格式预插入状态数据库。
- `ppo.py`：独立 Gaussian residual PPO、GAE、非对称 critic、critic warm-up。
- `flow_noise.py` / `flow_ppo.py`：Conditional Flow Matching 内层积分的
  Gaussian transition、完整 flow log-prob 和 PPO ratio。
- `isaac_env.py`：新增 `TavlaAffineResidualEnv`。它通过子类接入 affine
  wrench，并让 residual PPO 的 7 个机械臂维度确实作用到 teacher target。

## 纯模块测试

```bash
cd /home/gujiawei/isaac_env
/home/gujiawei/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests/test_tavla_baseline.py
/home/gujiawei/miniconda3/envs/env_isaaclab/bin/python scripts/reinforcement_learning/tavla_baseline/pure_smoke.py
```

这两项不连接 Server，也不启动 Isaac Sim；它们验证 adapter 数值公式、符号、
观测边界、随机化阶段、奖励、终止和状态数据库。

## residual PPO baseline

当前远程 Server 是单环境状态ful 接口，所以 baseline 先固定为一个环境：

```bash
cd /home/gujiawei/isaac_env
./isaaclab.sh -p scripts/reinforcement_learning/tavla_baseline/train_residual_ppo.py \
  --headless \
  --device cuda:0 \
  --tavla-host 10.0.40.113 \
  --tavla-port 8000 \
  --updates 100 \
  --rollout-steps 128 \
  --output-dir logs/tavla_baseline/residual_ppo
```

它使用 `TavlaAffineResidualEnv`，不会改动正在运行的 `train` tmux。输出包括：

- `latest.pt`：actor、critic、两个 optimizer 和配置；
- `metrics.jsonl`：policy/value loss、entropy、KL、clip fraction、explained
  variance、teacher failure；
- `config.json`：本次 baseline 配置副本。

确定性评估：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/tavla_baseline/evaluate_residual_ppo.py \
  --headless \
  --checkpoint logs/tavla_baseline/residual_ppo/latest.pt \
  --episodes 100 \
  --output logs/tavla_baseline/residual_eval.json
```

## 预插入状态库

```bash
./isaaclab.sh -p scripts/reinforcement_learning/tavla_baseline/capture_preinsert_states.py \
  --headless --episodes 100 \
  --output outputs/tavla_baseline/preinsert_states.json
```

当前脚本先记录进入 engage 条件的状态。后续 reset 逻辑可通过这个 JSON 数据库
接入新的专用 PegInsert 环境，而不影响旧环境。

## Flow-Noise PPO

`train_flow_noise_ppo.py` 不接受远程 Server 作为 actor。它要求在 TA-VLA 模型
机器上提供一个 bridge factory：

```python
def make_bridge(config):
    return bridge
```

bridge 必须提供：

```text
sampler
critic
actor_parameters
critic_parameters
sample(observation) -> (action, old_log_prob, flow_trace)
evaluate(...)
update(observation, critic_observation, trace, old_log_prob, returns, advantages)
state_dict()
```

其中 `sampler` 是 `FlowNoiseSampler`，velocity 必须连接本地 TA-VLA action
expert。远程 Server 没有 velocity、transition log-prob 或梯度，因此不能用于
这个主方案。

## 当前配置注意事项

`force_soft_threshold`、`force_hard_threshold`、`torque_soft_threshold` 和
`torque_hard_threshold` 都是配置值，不能直接视为最终真机安全阈值。正式实验前
需要用成功真机轨迹的力/力矩分布重新填写，并在报告中记录来源。

