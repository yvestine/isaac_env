"""Isaac Sim 5.1 torch-transform compatibility for IsaacLab 6.

The TacEx task code uses the former ``isaacsim.core.utils.torch`` names and
its ``(quaternion, position)`` return order for frame operations. IsaacLab 6
keeps the math implementation in ``isaaclab.utils.math`` and uses XYZW
quaternions, so this adapter preserves the old call signatures while using
the current convention internally.
"""

from __future__ import annotations

from pathlib import Path

import isaacsim

_ISAACSIM_EXTS = Path(isaacsim.__file__).resolve().parent / "exts"
for _extension_name in ("isaacsim.core.prims", "isaacsim.core.utils"):
    _extension_pkg = _ISAACSIM_EXTS / _extension_name / "isaacsim"
    if _extension_pkg.is_dir() and str(_extension_pkg) not in isaacsim.__path__:
        isaacsim.__path__.append(str(_extension_pkg))

# TacEx/RealSim uses the original Isaac quaternion order: WXYZ.
from isaacsim.core.utils import torch as _torch_utils  # noqa: E402

quat_apply = _torch_utils.quat_apply
quat_conjugate = _torch_utils.quat_conjugate
quat_from_angle_axis = _torch_utils.quat_from_angle_axis
quat_from_euler_xyz = _torch_utils.quat_from_euler_xyz
quat_mul = _torch_utils.quat_mul
get_euler_xyz = _torch_utils.get_euler_xyz
tf_inverse = _torch_utils.tf_inverse
tf_combine = _torch_utils.tf_combine
xyzw2wxyz = _torch_utils.xyzw2wxyz
wxyz2xyzw = _torch_utils.wxyz2xyzw
