# Baseline 验收顺序

## 1. 不启动 Isaac Sim 的测试

```bash
cd /home/gujiawei/isaac_env
/home/gujiawei/miniconda3/envs/env_isaaclab/bin/python -m pytest -q tests/test_tavla_baseline.py
/home/gujiawei/miniconda3/envs/env_isaaclab/bin/python scripts/reinforcement_learning/tavla_baseline/pure_smoke.py
```

## 2. Server 协议和 affine residual PPO

```bash
./isaaclab.sh -p scripts/reinforcement_learning/tavla_baseline/train_residual_ppo_protocol.py \
  --headless --device cuda:0 \
  --tavla-host 10.0.40.113 --tavla-port 8000 \
  --updates 100 --rollout-steps 128 \
  --output-dir logs/tavla_baseline/residual_ppo_protocol
```

此入口使用：

```text
224x224 RGB
state=(8,)
effort=(1,6)
reset={"reset": True}
action=(50,8)
```

如果当前 Server 仍然只支持“断开并重连”而不接受 reset 消息，使用
`train_residual_ppo.py` 这个兼容入口；它仍然使用 affine adapter，但沿用已有
client 的重连 reset 行为。

## 3. 确定性评估

```bash
./isaaclab.sh -p scripts/reinforcement_learning/tavla_baseline/evaluate_residual_ppo.py \
  --headless --checkpoint logs/tavla_baseline/residual_ppo_protocol/latest.pt \
  --episodes 100 --output logs/tavla_baseline/residual_eval.json
```

## 4. 主方案 Flow-Noise PPO

这个阶段必须把 bridge 和 TA-VLA 本地 action expert 放到 TA-VLA 机器上。现有
远程 Server 只有最终 action，没有 flow velocity、transition log-prob 或梯度，
所以不把远程 Server 伪装成 Flow-Noise actor。

