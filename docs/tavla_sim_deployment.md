# TAVLA 真机微调策略在 Isaac Sim 中的部署记录

## 目的

本目录用于把 `real_data/` 中微调得到的单臂 TAVLA 策略部署到
`TacEx-RealSim-PegInsert` 仿真中。仿真使用与当前 PPO RealSim 任务相同的
原始 Franka 资产、机器人底座位置、reset 关节状态和 reward 计算；动作来源
替换为远程 TAVLA teacher，不加载 PPO actor。

## 数据契约

真机数据目录：

```text
/home/gujiawei/isaac_env/real_data/traj_*/data.h5
```

策略训练时使用的语义来自现有 `tavla_ee_wrench_finetuning_report.md`：

- 40 条单臂轨迹，控制/采样频率约 10 Hz；
- `state = [joint_pos(7), gripper(1)]`，关节单位为 rad；
- 夹爪为 `0=closed, 1=open`；
- 实测 H5 的 `gripper_pos`/action 数值约为 `0.084~0.087`，对应 `gripper_width_m≈0.0069 m`；部署保留这个归一化数值域，仿真用 `target_gripper * 0.04 m` 作为每个手指关节目标，因此策略输出约 `0.087` 会映射到约 `0.00348 m`/finger，而不是被误当成米制宽度。
- `effort = obs/state/ee_wrench_base`，顺序 `[Fx,Fy,Fz,Tx,Ty,Tz]`；
- 图像为 `front_camera` 和 `wrist_camera`，部署时 wrist 图像复制到左右腕相机键；
- action 为 8 维绝对关节/夹爪目标，action chunk 长度为 50；
- task prompt 固定为 `peg-in-hole`。

## 仿真部署链路

```text
TAVLA server 10.0.40.113:8000
  -> WebSocket/msgpack-numpy
  -> PI0RemotePolicyTAVLA
  -> TavlaResidualEnv teacher-only mode
  -> actions[1] 的 8-D absolute joint target
  -> articulation joint-position target + PhysX
  -> existing RealSim/Forge/Factory reward
```

teacher-only 模式的 RL action 为零且不会参与控制；它只借用同一环境的
reward 和 reset 逻辑来评估 TAVLA。`TAVLAResidual` 任务仍保留用于后续残差
PPO，但本次部署不使用残差训练。

## 当前原始 Franka 基线

- Robot asset：`assets/Factory/franka_mimic.usd`；
- articulation root：`fix_root_link=True`；
- RealSim 背景和机器人底座默认位姿：位置 `(0,0,0)`，四元数 `wxyz=(1,0,0,0)`；
- reset 关节来自 `FactoryEnvCfg.ctrl.reset_joints`，与现有 RealSim PPO reset 路径一致；
- physics dt 为 `1/120`，environment decimation 为 `4`，每 3 个 30-Hz 环境步请求一次 Server，
  对应约 10 Hz 的策略更新。

## 已有对齐结果

优先参考：

- `outputs/real_data_alignment/`：原始 Franka 的真机—仿真位姿标定；
- `outputs/replay_hierarchical_corrected/full_validation/`：40 条轨迹动力学回放的候选参数；
- `outputs/multi_replay/`：多轨迹 reward、关节和末端回放结果。

原始 Franka 标定报告中的关节映射为 identity（7 个 sign 均为 `+1`，offset 均为
`0 rad`）。历史 `outputs/*ee90*` 仅用于比较 90° 资产，不作为当前默认配置。

当前动力学对齐的已保存最佳候选为：

```text
lead_time_s = 0.17564227755665782
arm1_kp = 303.3830260094937
arm1_kd = 36.850833024805276
arm2_kp = 148.56203519189626
arm2_kd = 17.460777084541743
```

其中 `lead_time_s` 只用于离线真机轨迹 replay 的动力学标定；在线 TAVLA 部署使用同一组 arm1/arm2 隐式位置伺服增益，action chunk 通过 `hold_steps=3` 对应仿真 30 Hz 到策略 10 Hz。
这些参数先用于回放和 teacher-only 评估；不会改写 TAVLA checkpoint。

## 运行入口

语法和配置检查：

```bash
cd /home/gujiawei/isaac_env
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/tavla_eval.py --help
```

teacher-only 仿真评估：

```bash
cd /home/gujiawei/isaac_env
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/tavla_eval.py \
  --task TacEx-RealSim-PegInsert-TAVLA-Teacher-v0 \
  --tavla-host 10.0.40.113 \
  --tavla-port 8000 \
  --steps 600 \
  --episodes 1 \
  --output-dir outputs/tavla_teacher_eval \
  --headless \
  --enable_cameras
```

输出包括：

- `reward.csv`：每个仿真环境步的 reward；
- `reward_terms.csv`：现有 reward 各项；
- `tavla_actions.csv`：策略原始动作和裁剪后的目标；
- `tavla_joint_state.csv`：仿真 7 关节、夹爪状态和 teacher 误差；
- `report.json`：推理次数、失败次数、延迟、reward 汇总和 reset 配置；
- `episode_*/front/front.mp4`、`episode_*/wrist/wrist.mp4`：H.264、`yuv420p`、可直接在 VS Code 预览的相机视频；
- `episode_*/front/frame_*.png`、`episode_*/wrist/frame_*.png`：首帧、中间帧和末帧截图；
- `episode_*/tavla_executed_targets.csv`：实际下发给关节位置控制器的 8 维目标；
- `episode_*/tavla_inference_latency_s.csv`、`tavla_inference_timeouts.csv`、`tavla_action_nonfinite.csv`、`tavla_target_out_of_limits.csv`：闭环运行诊断。

## 低 reward 复核与当前结论

`reward=0.2075` 不是成功 reward。它来自 reward 中仍然保留的 keypoint 基础项：已有 `outputs/tavla_teacher_final_eval/reward_terms.csv` 显示 `curr_engaged=0`、`curr_success=0`、`insertion_progress≈0` 和 `pre_insert_progress≈0`，所以策略没有进入插入阶段；总 reward 不是被代码截断，而是任务几何状态没有满足插入门控。

已保存的 599 步纯 TAVLA 长测中，reward 前 10% 平均约 `0.17484`，后 10% 平均约 `0.15909`，线性斜率约 `-1.20e-4/step`，因此没有上升趋势。历史 privileged 诊断 `outputs/tavla_privileged_grasp_posxy_600/` 证明它有接近趋势但没有成功：XY 距离从 `35.36 mm` 最小到 `0.14 mm`，随后回到 `21.37 mm`；轴向间隙从 `91.67 mm` 最小到 `11.63 mm`，随后回到 `92.07 mm`；599 步中 engaged 565 步、success 0 步。也就是说它曾经接近孔位，但接触/姿态/控制保持没有完成插入。

默认评估现在保持纯 TAVLA：`privileged_xy_guidance=False`、`privileged_xyz_guidance=False`。只有显式传入 privileged 参数时才会启用仿真特权视角，且该结果必须单独标为 oracle 诊断，不能冒充真机策略。

当前最重要的排查结论是：WebSocket 契约、绝对关节 action、state/action 对齐、原始 `franka_mimic.usd`、固定底座和隐式动力学伺服均已接通；剩余低 reward 的主因是 TAVLA 输出在当前仿真接触几何下没有稳定完成末端姿态与插入深度，而不是 reward 统计链路本身。正在运行的 `outputs/replay_current_asset_traj9/` 原始真机轨迹回放用于进一步隔离资产/物理接触与 TAVLA 控制问题。

## 仿真相机视频

每次评估都会为每个 episode 保存两路相机视频和首/中/末帧截图，不生成额外 HTML：

- `outputs/tavla_teacher_eval/episode_*/front/front.mp4`
- `outputs/tavla_teacher_eval/episode_*/wrist/wrist.mp4`
- `outputs/tavla_teacher_eval/episode_*/{front,wrist}/frame_*.png`

MP4 使用 `libx264`、`yuv420p` 和 `+faststart`，可直接在 VS Code/Electron 中预览。

## 诊断运行方式

纯 TAVLA（推荐基线）：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/tavla_eval.py --headless --enable_cameras \
  --steps 600 --output-dir outputs/tavla_teacher_pure_600
```

开启 XY 特权对齐仅用于诊断：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/tavla_eval.py --headless --enable_cameras \
  --steps 600 --privileged-xy-guidance-weight 1.0 \
  --output-dir outputs/tavla_teacher_privileged_xy_600
```

每次评估都检查 `report.json`、`reward_terms.csv` 和 `privileged_state.csv`；其中 `report.json` 现在会写入 reward 前后 10% 均值、线性斜率和最佳 step。

## 回滚

本次修改涉及 TAVLA 默认配置、WebSocket client、相机实例化、底座配置参数、
teacher runtime、动力学对齐控制器和新增评估脚本/文档。`real_data/`、`outputs/`、
历史回放脚本和已有资产不删除。

回滚时优先只恢复本次列出的源码文件；不要删除 `real_data/` 或 `outputs/`：

```bash
git diff -- source/tacex_tasks/tacex_tasks/real2sim \
  scripts/reinforcement_learning/rl_games/tavla_eval.py \
  docs/tavla_sim_deployment.md
```

如果确认只需要撤销源码改动，可使用：

```bash
git restore --source=HEAD -- \
  source/tacex_tasks/tacex_tasks/real2sim/realsim_env.py \
  source/tacex_tasks/tacex_tasks/real2sim/realsim_env_cfg.py \
  source/tacex_tasks/tacex_tasks/real2sim/policy/configuration_pi0remote.py \
  source/tacex_tasks/tacex_tasks/real2sim/policy/modeling_pi0remote.py \
  source/tacex_tasks/tacex_tasks/real2sim/tavla_residual_env.py \
  source/tacex_tasks/tacex_tasks/real2sim/tavla_residual_env_cfg.py
```

新增的 `tavla_eval.py` 和本说明文档可以单独移除；不要使用宽范围的
`git clean`，因为当前工作区已有用户轨迹和对齐结果。


## WebSocket 输入对齐

评估脚本连接 ws://10.0.40.113:8000，复用一个连接完成整个 episode，并给握手和推理设置超时。
发送给 TAVLA 服务的字段为：

```text
images.cam_high        : front RGB, 640x480
images.cam_left_wrist  : wrist RGB, 640x480
images.cam_right_wrist : wrist RGB 的副本
state                  : [joint_pos(7), gripper(1)]
effort                 : (1, 6)，服务器自动补 batch 后为 (1, 1, 6)
prompt                 : peg-in-hole
```

这对应 real_data 中的单臂 state、base-frame wrench 和双相机训练接口；
力/力矩的物理量标定仍未纳入本轮，当前只保证 wrench 的字段、顺序和张量形状。

## 早期链路验证（不是成功结论）

早期 teacher-only 输出目录为 outputs/tavla_teacher_final_eval/；其 reward 结果已在上方低 reward 复核中重新解释。

- 600 步请求实际完成 599 个环境步、1 个 episode；
- TAVLA inference count 为 5，teacher failures 为 0，跨越多个 50-action chunk；
最后一次服务器推理延迟约 0.176 s；首次冷启动推理较慢，之后连接复用；
reward：mean 0.2074979，sum 124.2912，min 0.1563900，max 0.3068866；
- reward、action、joint-state CSV 均无 NaN/Inf；teacher 关节误差最大绝对值约
  0.3681 rad，平均绝对值约 0.0279 rad；
- report 确认 robot asset 为 assets/Factory/franka_mimic.usd，base 为
  (0,0,0), wxyz=(1,0,0,0)，reset 与现有 PPO 一致；
- 最终代码另做了 1 步 smoke，输出在 outputs/tavla_teacher_final_smoke/，
  确认隐式位置伺服模式和对齐增益已生效。

预先做的 60 步和旧显式 torque-PD 长测输出仍保留，便于比较；最终结论以
隐式动力学对齐版为准。当前 reward 评估证明链路和数值稳定，不等于已经完成
插入成功率优化；force/torque 对齐和 90 度夹爪资产问题仍是后续工作。

real_data/traj_0 的 60 帧回放诊断曾使用 endpoint 自动推断固定孔位，并在第一帧
env.step 长时间无输出后中止；该方法是循环验证，不能作为独立对齐证据。
独立的 40 轨迹结果仍以 outputs/replay_hierarchical_corrected/ 和
outputs/real_data_alignment/ 为准，诊断输出只保留了
outputs/tavla_alignment_replay_traj0/replay_config.json。

## 回滚补充

本次修改涉及 TAVLA 默认配置、WebSocket client、相机实例化、底座配置参数、
teacher runtime、动力学对齐控制器和新增评估脚本/文档。real_data、outputs、
历史回放脚本和已有资产不删除。

如需完整撤销本次部署源码，恢复以下 tracked 文件：

```text
source/tacex_tasks/tacex_tasks/real2sim/realsim_env.py
source/tacex_tasks/tacex_tasks/real2sim/realsim_env_cfg.py
source/tacex_tasks/tacex_tasks/real2sim/policy/configuration_pi0remote.py
source/tacex_tasks/tacex_tasks/real2sim/policy/modeling_pi0remote.py
source/tacex_tasks/tacex_tasks/real2sim/tavla_residual_env.py
source/tacex_tasks/tacex_tasks/real2sim/tavla_residual_env_cfg.py
```
新增的 scripts/reinforcement_learning/rl_games/tavla_eval.py 和本说明文档可单独移除；
不要使用宽范围 git clean，因为工作区已有用户轨迹和对齐结果。

## 修改与验证日志

### 2026-08-10 最终记录

1. 检查 10.0.40.113:8000 可连接并能返回 msgpack metadata。
2. 修复 TAVLA effort 输入为单历史帧 (1,6)，服务器端解析为 (1,1,6)；state、
   相机和 8 维 action 均通过服务端类型检查。
3. 远程 client 复用连接并增加握手/推理超时，避免 reset 时等待失控。
4. 使用原始 franka_mimic.usd、固定 root 和现有 PPO reset；teacher-only 时 RL
   action 置零，reward 仍来自原 RealSim/Factory。
5. 接入保存的 40 轨迹动力学对齐候选，并使用与历史回放相同的隐式位置伺服；
   600 步最终长测 inference failures=0，输出 CSV 数值有限。
