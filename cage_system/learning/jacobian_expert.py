"""Training-only damped-Jacobian expert for local UR5e Cage tracking."""

from dataclasses import dataclass

import numpy as np

from cage_system.learning.ppo2 import relative_pose_error6


def _clip_norm(vector, maximum):
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= float(maximum) or norm <= 1e-12:
        return value
    return value * (float(maximum) / norm)


@dataclass(frozen=True)
class JacobianExpertResult:
    normalized_action: np.ndarray
    joint_velocity_radps: np.ndarray
    pose_error6: np.ndarray
    minimum_singular_value: float
    damping: float
    saturated: bool


class DampedJacobianExpert:
    """Map the same current Cage error seen by Actor into one 50 ms action.

    The expert never reads a future object point.  It uses the current Cage
    target, current MuJoCo TCP transform, q and qdot only.  Its result already
    follows the PPO normalized per-visual-event action convention.
    """

    def __init__(self, environment, config):
        self.environment = environment
        self.config = config

    def action(self, cage_target_pose6=None):
        env = self.environment
        target = (
            env.current_target
            if cage_target_pose6 is None
            else np.asarray(cage_target_pose6, dtype=float)
        )
        tcp_pose = env.tcp_pose6()
        error = relative_pose_error6(target, tcp_pose)
        linear = _clip_norm(
            float(self.config["position_gain_s_inv"]) * error[:3],
            float(self.config["maximum_tcp_linear_speed_mps"]),
        )
        angular = _clip_norm(
            float(self.config["orientation_gain_s_inv"]) * error[3:],
            float(self.config["maximum_tcp_angular_speed_radps"]),
        )
        desired_twist = np.concatenate([linear, angular])

        jacobian_position = np.zeros((3, env.model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, env.model.nv), dtype=float)
        env.mujoco.mj_jacSite(
            env.model,
            env.data,
            jacobian_position,
            jacobian_rotation,
            env.tcp_site_id,
        )
        jacobian = np.vstack(
            [
                jacobian_position[:, env.dof_addresses],
                jacobian_rotation[:, env.dof_addresses],
            ]
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        minimum_singular = float(np.min(singular_values))
        threshold = float(self.config["singularity_threshold"])
        damping = float(self.config["base_damping"])
        if minimum_singular < threshold:
            fraction = (threshold - minimum_singular) / max(threshold, 1e-9)
            damping += float(self.config["singularity_damping_gain"]) * fraction
        regularized = jacobian @ jacobian.T + damping ** 2 * np.eye(6)
        try:
            task_inverse = jacobian.T @ np.linalg.solve(
                regularized, np.eye(6)
            )
        except np.linalg.LinAlgError:
            task_inverse = np.linalg.pinv(jacobian, rcond=1e-4)
        joint_velocity = task_inverse @ desired_twist

        # The Actor observes qdot, so its teacher must not assign the same
        # command to an arm already moving toward the target and one moving
        # away from it. This feedback turns the kinematic solution into a
        # locally damped one-step command without reading a future state.
        current_joint_velocity = env.data.qvel[env.dof_addresses].copy()
        joint_velocity = joint_velocity - float(
            self.config["joint_velocity_feedback_gain"]
        ) * current_joint_velocity

        joints = env.data.qpos[env.qpos_addresses].copy()
        center = 0.5 * (env.lower + env.upper)
        half_range = np.maximum(0.5 * (env.upper - env.lower), 1e-6)
        centering_velocity = -float(self.config["joint_centering_gain"]) * (
            joints - center
        ) / half_range
        nullspace = np.eye(6) - task_inverse @ jacobian
        joint_velocity = joint_velocity + nullspace @ centering_velocity

        configured_velocity = np.asarray(
            env.robot["joint_velocity_limit_radps"], dtype=float
        )
        velocity_limit = configured_velocity * float(
            self.config["joint_velocity_limit_fraction"]
        )
        clipped_velocity = np.clip(joint_velocity, -velocity_limit, velocity_limit)
        safe_delta = np.minimum(
            env.delta_max,
            configured_velocity * env.decision_dt,
        )
        requested_delta = clipped_velocity * env.decision_dt
        normalized = np.clip(
            requested_delta / np.maximum(safe_delta, 1e-9), -1.0, 1.0
        )
        saturated = bool(
            np.any(np.abs(joint_velocity - clipped_velocity) > 1e-9)
            or np.any(np.abs(requested_delta) > safe_delta + 1e-9)
        )
        return JacobianExpertResult(
            normalized.astype(np.float32),
            clipped_velocity.astype(np.float32),
            error.astype(np.float32),
            minimum_singular,
            damping,
            saturated,
        )
