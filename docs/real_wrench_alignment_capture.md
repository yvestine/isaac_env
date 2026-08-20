# Real/simulation wrench alignment capture

This document defines the data needed before enabling `ft_use_corrected_wrench`.
It intentionally does not contain passwords, tokens, or network credentials.

## 1. Output layout

Capture one directory per episode or contact test:

```text
real_wrench_alignment/
  episode_000/
    robot_state_samples.jsonl
    wrench_pose.csv
    contact_tests.csv
    tool_load_config.json
```

Use the same clock for every file. `timestamp` must be monotonic and its unit
must be stated in the file metadata; seconds from the recorder start are
recommended. `episode_id` identifies the trajectory, while `sample_id`
identifies a row inside the trajectory.

## 2. RobotState snapshots

At the beginning, middle, and end of every collection episode, save one JSON
object to `robot_state_samples.jsonl` with these fields:

```json
{
  "timestamp": 0.0,
  "episode_id": "episode_000",
  "sample_id": "start",
  "EE_T_K": "[16 values, row-major 4x4]",
  "F_T_EE": "[16 values, row-major 4x4]",
  "O_T_EE": "[16 values, row-major 4x4]",
  "m_ee": 0.0,
  "F_x_Cee": "[3 values]",
  "I_ee": "[9 values, row-major 3x3]",
  "m_load": 0.0,
  "F_x_Cload": "[3 values]",
  "I_load": "[9 values, row-major 3x3]",
  "m_total": 0.0
}
```

The values above are placeholders for the real recorder output. Do not fill
unknown values with zero and do not infer `m_ee=0.95 kg` contents from the
current simulation config. Record the exact RobotState values returned by the
robot interface, plus the tool/fixture/pin configuration used at that time.

The three snapshots are used to check:

- whether `EE_T_K` is constant;
- whether `m_ee`, `m_load`, `m_total`, and inertias change;
- whether the tool, fixture, or pin was replaced during collection.

## 3. Synchronized wrench and pose CSV

Write `wrench_pose.csv` with one row per synchronized sample and this header:

```text
timestamp,episode_id,sample_id,O_F_ext_hat_K_0,O_F_ext_hat_K_1,O_F_ext_hat_K_2,O_F_ext_hat_K_3,O_F_ext_hat_K_4,O_F_ext_hat_K_5,K_F_ext_hat_K_0,K_F_ext_hat_K_1,K_F_ext_hat_K_2,K_F_ext_hat_K_3,K_F_ext_hat_K_4,K_F_ext_hat_K_5,ee_wrench_base_0,ee_wrench_base_1,ee_wrench_base_2,ee_wrench_base_3,ee_wrench_base_4,ee_wrench_base_5,ee_wrench_stiffness_0,ee_wrench_stiffness_1,ee_wrench_stiffness_2,ee_wrench_stiffness_3,ee_wrench_stiffness_4,ee_wrench_stiffness_5,ee_pose_x,ee_pose_y,ee_pose_z,ee_pose_qx,ee_pose_qy,ee_pose_qz,ee_pose_qw
```

All wrench columns use `[Fx,Fy,Fz,Tx,Ty,Tz]` and units
`[N,N,N,N*m,N*m,N*m]`. The pose order must be explicitly
`[x,y,z,qx,qy,qz,qw]`.

Do not collapse these fields before saving. The offline checker compares them
independently and reports missing metadata instead of guessing:

- `ee_wrench_base == O_F_ext_hat_K` must be checked from samples;
- `ee_wrench_stiffness == K_F_ext_hat_K` must be checked from samples;
- `ee_pose == O_T_EE` and its quaternion order must be stated;
- the torque reference point must be stated as K, EE, or another origin.

## 4. Known-direction contact tests

For each direction, record a short no-motion/light-contact segment in
`contact_tests.csv`. The operator must state the coordinate system of the
applied direction (base O, stiffness K, or tool frame), the approximate force,
and the contact point.

```text
test_id,timestamp_start,timestamp_end,direction_name,force_vector,force_frame,approx_force_N,contact_point_frame,contact_point_m,O_F_ext_hat_K_0,O_F_ext_hat_K_1,O_F_ext_hat_K_2,O_F_ext_hat_K_3,O_F_ext_hat_K_4,O_F_ext_hat_K_5,K_F_ext_hat_K_0,K_F_ext_hat_K_1,K_F_ext_hat_K_2,K_F_ext_hat_K_3,K_F_ext_hat_K_4,K_F_ext_hat_K_5,notes
```

Recommended tests are `+X`, `-X`, `+Y`, `-Y`, `+Z`, and `-Z`. Keep them
small and repeatable. The purpose is to decide, from measured data rather than
PhysX assumptions, whether the simulation incoming wrench needs an overall
minus sign, an axis permutation, or per-axis sign changes.

Do not enable corrected TAVLA input until the tests establish:

1. incoming force sign;
2. force-axis mapping;
3. torque-axis mapping;
4. base/stiffness frame rotation;
5. torque reference point.

## 5. Tool/load manifest

Save `tool_load_config.json` for every configuration used during collection:

```json
{
  "timestamp": 0.0,
  "episode_id": "episode_000",
  "tool_mass_kg": null,
  "fixture_mass_kg": null,
  "pin_mass_kg": null,
  "end_object_mass_kg": null,
  "m_ee_kg": null,
  "m_load_kg": null,
  "m_total_kg": null,
  "center_of_mass_position_m": null,
  "center_of_mass_frame": null,
  "installation_direction": null,
  "tool_origin_relative_to_EE_m": null,
  "tool_origin_relative_to_K_m": null,
  "tool_origin_relative_to_flange_m": null,
  "components_in_m_ee": [],
  "components_in_m_load": [],
  "configuration_changed_during_collection": null,
  "same_as_current_deployment": null,
  "notes": ""
}
```

In particular, explicitly answer what `m_ee=0.95 kg` contains. `m_load=0`
means only that no external load was configured in that RobotState; it does
not prove that the physical tool/fixture/pin mass is zero.

## 6. Simulation contract after calibration

The simulation writes these fields and keeps the first three for comparison:

| field | frame | torque reference | status |
|---|---|---|---|
| `wrench_raw` | configured incoming frame | configured incoming reference | PhysX source value |
| `wrench_anchor` | force-sensor anchor frame | force-sensor joint anchor | geometric diagnostic |
| `wrench_base` | robot base | robot-base origin | real training-contract candidate |
| `wrench_corrected` | configured robot-base frame | configured reference (`base_origin` by default) | calibration candidate |

The translation used by the simulation is:

```text
tau_B = tau_A + cross(p_A - p_B, F)
```

The configured 3x3 calibration matrices use column-vector convention
`output = M @ input`, followed by the configured six-component sign vector.
All six components are ordered `[Fx,Fy,Fz,Tx,Ty,Tz]`.

The current IsaacLab 2.x defaults are `parent_body + parent_origin`, and the
base-layer torque reference is the robot-base origin. Corrected use remains
disabled. After the simulator contact/sign test is reviewed,
set the frame/reference/sign/matrix fields in `RealSimEnvCfg`, then set both
`ft_corrected_ready=True` and `ft_use_corrected_wrench=True`. The runtime will
reject the latter if the former is still false.
