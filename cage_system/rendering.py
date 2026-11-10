"""Shared MuJoCo scene drawing helpers."""
import numpy as np

def _add_capsule(scene, mujoco, start, end, radius, rgba):
    """Append one non-physical audit capsule to an already populated MjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.asarray([radius, radius, radius], dtype=float),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    scene.ngeom += 1
