# UR5e model provenance and project extension

The upstream files in this directory were copied from Google DeepMind's
`mujoco_menagerie` repository at commit
`da76818e269b82289eba39808e2fb91d679d6994`.

Upstream model: `universal_robots_ur5e`

Upstream license: BSD-3-Clause, preserved in `LICENSE`.

Project additions:

- `ur5e_eye_on_hand.xml` starts from the upstream `ur5e.xml` and adds the
  configured U-channel Cage gripper, TCP endpoint site and an eye-on-hand
  perspective camera to `wrist_3_link`.  The derived Base is rotated into the
  project's positive-x workspace convention; the untouched upstream file keeps
  its original frame.
- `scene_eye_on_hand.xml` loads that derived robot, adds lighting, floor and a
  renderable mocap target used by the trajectory/perception loop.

The Menagerie README describes this as a simplified UR5e description derived
from the public URDF.  It is a curated dynamics model, not a calibrated digital
twin of a particular physical arm, gripper or camera.  Real deployment still
requires system identification and hand-eye/intrinsics calibration.
