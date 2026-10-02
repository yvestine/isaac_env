# TAVLA 仿真测试说明

## 当前数据契约

每个策略请求包含：

- `images.cam_high`：front RGB；
- `images.cam_left_wrist`：wrist RGB；
- `images.cam_right_wrist`：wrist 副本；
- `state`：7 维关节位置 + 1 维夹爪，形状 `[8]`；
- `effort`：10 帧历史，每帧 10 维力趋势，形状 `[10,10]`；
- `prompt`：`peg-in-hole`。

两个服务端模型接收同一输入：

- 8000：`pi0_lora_user_single_arm_force_trend_cotrain_affine_50_50`，real/sim 50:50 co-training；
- 8001：`pi0_lora_user_single_arm_force_trend_sim_affine`，sim-only affine training。

模型 checkpoint 在远程服务器，仿真仓库只包含输入 adapter，不复制权重。

## 视觉、力和控制的职责

- `sim-data/traj_N` 提供 40 个 in-distribution 初始关节、孔位和视觉随机化参数；
- `sim-data-aligned/traj_N/data.h5` 提供与训练一致的 aligned 力数据契约；
- 在线运行时从 rollout 相同的 PhysX wrist/link7 路径读取原始 wrench；
- `force_alignment.py`、affine adapter、force-trend encoder 只修改 observation；
- TAVLA 返回 8 维 absolute joint/gripper action 后，仍走原 TAVLA controller、插值和 PhysX 执行路径。

因此 force-trend 接入不应改变机械臂控制逻辑。当前专用任务为：

```text
TacEx-RealSim-PegInsert-SimDataAligned-v0
```

## 单条测试

```bash
POLICY_HOST=114.214.164.36 \
POLICY_PORT=8000 \
TAVLA_ACTION_START_INDEX=1 \
REPLAN_ACTIONS=5 \
EPISODE_LENGTH_S=60 \
bash scripts/run_sim_data_aligned_eval_all.sh \
  0 0 outputs/smoke_force_trend_8000_profile_0
```

默认是 overwrite；同一目录中的旧结果会被原位清理。设置 `EVAL_MODE=resume` 可跳过已经存在 `summary.json` 和 `episodes.csv` 的 profile。

## 320 条正式测试

```bash
POLICY_HOST=114.214.164.36 \
bash scripts/run_tavla_8exp_40.sh outputs/tavla_8exp overwrite
```

运行顺序固定为 port → action start → replan actions → profile，并且始终串行。每个 Isaac Sim 进程结束后才会启动下一个；中断时只清理当前进程组，不影响其他训练进程。

## 指标

每条 `episodes.csv` 保存：

- `best_xy_error_m`；
- `minimum_z_disp_m`；
- `minimum_z_disp_when_xy_lt_3mm_m`：仅在该步同时满足 XY < 3 mm 时更新；
- `success`、partial insertion、请求失败和 timeout。

`summarize_tavla_8exp.py` 使用第三个字段统计 `XY < 3 mm` 条件下的 Z 阈值，避免把不同时间点的独立 XY/Z 最小值组合成不存在的结果。旧输出没有该字段时会使用 fallback，并在 `conditional_z_exact_rows` 中标记精确记录条数。

## 输出布局

```text
outputs/tavla_8exp/
  port_8000_start_1_steps_5/profile_00/
  ...
  port_8001_start_5_steps_10/profile_39/
  metrics_summary.csv
  metrics_summary.json
```

每个 profile 目录包含 `summary.json`、`episodes.csv`、episode 诊断 CSV 和策略输入视频。`outputs/` 不进入 Git。
