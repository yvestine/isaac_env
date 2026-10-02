# sim-data

这里保存 40 条真实轨迹在 Isaac Sim 中逐条回放后得到的视觉/reset profile。`traj_0` 到 `traj_39` 与 `real_data` 一一对应，当前总帧数为 7188。

每条目录包含：

- `data.h5`：状态、动作和回放数据；
- `front_camera.mp4`、`wrist_camera.mp4`：10 Hz 视觉；
- `replay_metadata.json`：孔位、帧数和固定 seed 的视觉域参数；
- 回放诊断 CSV。

TAVLA 在线测试使用这里的初始关节、孔位和视觉参数，但力训练通道从 `sim-data-aligned/traj_N/data.h5` 校验，并在仿真运行时从 PhysX 在线采集后通过同一套 causal force alignment 和 force-trend encoder。

所有路径均为仓库相对路径。生成入口：

```bash
conda run --no-capture-output -n env_isaaclab \
  python sim-data/build_sim_data.py
```

正式评估不会随机生成 OOD reset；profile `N` 始终使用 `traj_N` 保存的真实数据初始状态和视觉域参数。
