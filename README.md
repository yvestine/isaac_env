# Isaac Sim RealSim：rollout、PI0 与 TAVLA 测试

本仓库保留当前使用的一套链路：

- 真实轨迹在 Isaac Sim 中的关节回放与视觉 profile 生成；
- PI0 WebSocket 推理；
- TAVLA 10 帧 × 10 维力趋势推理；
- 40 个真实数据初始位置上的 8 组串行评估与统一统计；
- 仿真/真机因果力对齐和可视化。

旧 6 维 TAVLA、旧残差 PPO baseline、重复的 `Factory/` Franka 资产和一次性 traj0 实验脚本已从主树移出。背景实际引用的 `franka_new/` 资产仍保留，并已清除 USD 中的旧机器绝对路径。历史代码仍可从 Git 历史恢复，本机整理前快照位于 `backup/repo_cleanup_20261002_pre/`。

## 1. 新机器安装

要求：Ubuntu 22.04+、NVIDIA GPU/驱动、Miniconda、Git LFS。Isaac Sim 6.0 要求 Python 3.12 和 GLIBC 2.35+。

```bash
git clone https://github.com/yvestine/isaac_env.git
cd isaac_env

# 该变量表示你接受 NVIDIA Omniverse EULA。
OMNI_KIT_ACCEPT_EULA=yes bash scripts/setup_environment.sh
conda activate env_isaaclab
python scripts/check_repository.py --check-environment
```

安装脚本会自动完成：

1. 从本仓库的 Git LFS 拉取 USD、H5、MP4、PT 等文件；
2. 创建 `env_isaaclab`；
3. 安装与当前机器一致的 Python 3.12、Torch 2.10/CUDA 12.8、Isaac Sim 6.0；
4. 安装固定版本 `v3.0.0-beta2` 的 Isaac Lab；
5. 以 editable 模式安装三个 TacEx 包并执行完整文件检查。

机器人、背景、孔/peg、40 条 profile、aligned 力数据和力趋势 adapter 都在仓库/LFS 中，不需要再从其他目录复制资产。远程 PI0/TAVLA 模型权重仍运行在策略服务器上，不属于仿真端资产。

## 2. TAVLA smoke

单个 profile、8000 端口、最多 60 秒；目标目录会原位覆盖，不生成时间戳子目录：

```bash
POLICY_HOST=114.214.164.36 \
POLICY_PORT=8000 \
TAVLA_ACTION_START_INDEX=1 \
REPLAN_ACTIONS=5 \
EPISODE_LENGTH_S=60 \
bash scripts/run_sim_data_aligned_eval_all.sh \
  0 0 outputs/smoke_force_trend_8000_profile_0
```

该测试的视觉/reset profile 来自 `sim-data/traj_0`，力训练通道来自 `sim-data-aligned/traj_0/data.h5`。在线运行时读取 PhysX 力，经同一 causal alignment、affine adapter 和 force-trend encoder 生成 `[10,10]` effort；机械臂 action 执行仍使用原 TAVLA 控制逻辑。

## 3. 完整 8 × 40 串行评估

8 组配置为：端口 `8000/8001` × action 起点 `1/5` × 每次执行 `5/10` 步。每组使用同一批 40 个真实数据 profile 的初始关节、孔位和视觉域参数，不进行 OOD reset；每条最多 60 秒。

```bash
POLICY_HOST=114.214.164.36 \
bash scripts/run_tavla_8exp_40.sh outputs/tavla_8exp overwrite
```

中断后只继续未完成项：

```bash
POLICY_HOST=114.214.164.36 \
bash scripts/run_tavla_8exp_40.sh outputs/tavla_8exp resume
```

单独重算统计：

```bash
python scripts/summarize_tavla_8exp.py outputs/tavla_8exp --strict
```

统计写入 `metrics_summary.csv/json`，包括每组 40 条中的成功数、`XY < 3 mm` 数，以及同一仿真时刻满足 `XY < 3 mm` 时 `Z < 3/5/8/10 mm` 的数量。

两个端口输入完全相同，都是 10 帧 × 10 维力趋势：

- `8000`：real/sim 50:50 co-train affine force-trend；
- `8001`：sim affine force-trend。

## 4. Rollout 与 PI0

按 40 条真实轨迹逐条生成视觉 profile：

```bash
bash scripts/run_replay_rollouts_one_by_one.sh 0 39 outputs/paired_rollouts_40_dr
```

PI0 随机评估入口：

```bash
POLICY_HOST=114.214.164.36 conda run --no-capture-output -n env_isaaclab \
  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/pi0_randomized_eval.py \
  --policy pi0 --episodes 40 --episode-length-s 60 \
  --output-dir outputs/pi0_eval --headless
```

主要代码入口：

- `scripts/replay_real_joint_ppo.py`：关节轨迹回放和视觉 profile；
- `scripts/replay_real_ee_cartesian_ppo.py`：连续控制/物理力 rollout；
- `scripts/reinforcement_learning/rl_games/pi0_randomized_eval.py`：PI0/TAVLA 统一评估；
- `source/tacex_tasks/tacex_tasks/real2sim/sim_data_aligned_env.py`：aligned profile、在线力和控制连接；
- `source/tacex_tasks/tacex_tasks/real2sim/force_alignment.py`：共享因果力对齐；
- `scripts/run_causal_force_alignment_pipeline.sh`：力对齐数据流水线。

## 5. 提交到 GitHub

`outputs/`、`logs/`、`runs/`、`backup/`、环境依赖目录和缓存均已忽略。提交前执行：

```bash
python scripts/check_repository.py
git diff --check
git status --short
git add -A
git commit -m "Package portable TAVLA and PI0 evaluation workflow"
git push origin HEAD
```

必须使用 Git LFS 推送，否则新机器 clone 后会只得到大文件指针。`scripts/setup_environment.sh` 会在安装时再次执行 `git lfs pull` 并检查所有对象。
