# 仿真端接入包（对应 8000 / 8001）

## 已检查到的两个 server

当前 TAVLA 机器上：

- **8000** 已加载 `pi0_lora_user_single_arm_force_trend_cotrain_affine_50_50`，checkpoint 为 `checkpoints/pi0_lora_user_single_arm_force_trend_cotrain_affine_50_50/cotrain_affine_trend_50_50_30k/29999`。
- **8001** 已加载 `pi0_lora_user_single_arm_force_trend_sim_affine`，checkpoint 为 `checkpoints/pi0_lora_user_single_arm_force_trend_sim_affine/sim_affine_trend_30k/29999`。

这两个模型都已经在 server 侧加载，不要把模型权重复制到仿真端。它们都使用 10 维力趋势输入和 10 帧历史；仿真端使用同一份处理结果，分别发给 8000、8001 即可比较两种模型的 action。

## 仿真端要做什么

`serve_policy.py` 本身只负责加载策略和处理推理请求。TAVLA 输入变换会读取请求里的 `effort`，但不会把原始 6 维 wrench 自动转换成力趋势。因此仿真端需要：

1. 每个 episode 开始时调用 `SimForceTrendPreprocessor.reset()`。
2. 每 0.1 秒把仿真 base frame 的原始 wrench `[Fx,Fy,Fz,Tx,Ty,Tz]` 交给 `update()`。
3. 将 `history()` 的结果作为请求的 `effort` 字段。形状必须是 `[10, 10]`，每行一个 10 维力趋势向量。
4. 请求同时带上当前 TAVLA client 所需的 `images` 和 `state`，分别调用 8000 和 8001 的 websocket policy client，再读取返回的 `actions`。

力单位须为 N，力矩单位为 N·m，坐标系须与训练数据相同。仿真频率高于 10 Hz 时，先按训练频率采样，再更新历史。

## 包内文件

- `src/openpi/force_features.py`：训练和运行时使用的力趋势特征定义。
- `src/openpi/shared/wrench_adapter.py`：读取和应用仿真到真机的幅值适配器。
- `assets/wrench_adapters/sim_aligned_to_real_affine.pt` 及 `.json`：适配器参数和元数据。
- `configs/tavla_sim_force_trend_affine.json`：力趋势特征配置。
- `scripts/sim_force_trend_preprocessor.py`：仿真端调用的封装，输出 10 维特征和 `[10,10]` 历史。

本仓库已经直接加载这些文件，不需要再次解压或复制到其他 TAVLA checkout。`SimDataAlignedEnv` 会按仓库相对路径导入力特征和 adapter。包不包含 checkpoint；checkpoint 仍在远程策略服务器。

## 校准窗口提醒

训练数据转换用 JSON 中每条轨迹标注的 `free_end_frames` 估计空载基线；现有在线 buffer 默认使用开头 20 帧。两者可能不同。正式比较 action 前，应让仿真侧的空载校准窗口与训练处理一致，否则输入特征会有差别。
