"""Time-parameterized Cartesian tracking with constrained differential IK.

This is the executable counterpart of
``Jacobian_约束微分逆运动学_算法流程.pdf``.  The least-squares QP is
solved directly under joint velocity, predicted joint-position, and joint
acceleration bounds; it never computes an unconstrained pseudoinverse and then
clips the result.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from cage_system.geometry import rotation_as_matrix, rotation_from_matrix
from cage_system.learning.ppo2 import relative_pose_error6


def _clip_norm(vector, maximum):
    value = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(value))
    if length <= float(maximum) or length <= 1e-12:
        return value
    return value * (float(maximum) / length)


@dataclass(frozen=True)
class DifferentialIKCommand:
    joint_velocity_radps: np.ndarray
    reference_pose6: np.ndarray
    reference_twist6: np.ndarray
    commanded_twist6: np.ndarray
    task_error6: np.ndarray
    lower_velocity_bound_radps: np.ndarray
    upper_velocity_bound_radps: np.ndarray
    acceleration_relaxed_joints: tuple
    qp_success: bool
    qp_cost: float
    solve_time_ms: float


class TimeConstrainedDifferentialIK:
    """Closed-loop 100 Hz Cartesian reference tracker for the UR5e TCP."""

    def __init__(self, ur5e_model, config: Dict):
        self.ur5e = ur5e_model
        self.config = config
        self.control_dt = float(ur5e_model.control_dt)
        self.position_only = bool(config.get("position_only", False))
        self.start_pose6 = None
        self.goal_pose6 = None
        self.duration_s = None
        self.translation_velocity = None
        self.spatial_rotation_vector = None
        self.previous_joint_velocity = np.zeros(6, dtype=float)

    def reset(
        self,
        start_pose6,
        goal_pose6,
        duration_s: float,
        previous_joint_velocity=None,
    ) -> None:
        start = np.asarray(start_pose6, dtype=float)
        goal = np.asarray(goal_pose6, dtype=float)
        duration = float(duration_s)
        if start.shape != (6,) or goal.shape != (6,):
            raise ValueError("Differential IK requires start and goal TCP pose6")
        if duration <= 0.0:
            raise ValueError("The prescribed Cartesian motion time must be positive")
        start_rotation = rotation_as_matrix(Rotation.from_rotvec(start[3:]))
        goal_rotation = rotation_as_matrix(Rotation.from_rotvec(goal[3:]))
        relative_rotation = goal_rotation @ start_rotation.T
        self.start_pose6 = start.copy()
        self.goal_pose6 = goal.copy()
        self.duration_s = duration
        self.translation_velocity = (goal[:3] - start[:3]) / duration
        self.spatial_rotation_vector = rotation_from_matrix(
            relative_rotation
        ).as_rotvec()
        self.previous_joint_velocity = (
            np.zeros(6, dtype=float)
            if previous_joint_velocity is None
            else np.asarray(previous_joint_velocity, dtype=float).copy()
        )
        if self.previous_joint_velocity.shape != (6,):
            raise ValueError("Previous joint velocity must contain six values")

    def reference(self, elapsed_s: float):
        if self.start_pose6 is None:
            raise RuntimeError("Call reset before requesting a reference")
        alpha = np.clip(float(elapsed_s) / self.duration_s, 0.0, 1.0)
        start_rotation = rotation_as_matrix(
            Rotation.from_rotvec(self.start_pose6[3:])
        )
        reference_rotation = (
            rotation_as_matrix(
                Rotation.from_rotvec(alpha * self.spatial_rotation_vector)
            )
            @ start_rotation
        )
        reference_pose = np.concatenate(
            [
                self.start_pose6[:3]
                + alpha * (self.goal_pose6[:3] - self.start_pose6[:3]),
                rotation_from_matrix(reference_rotation).as_rotvec(),
            ]
        )
        moving = float(elapsed_s) < self.duration_s
        reference_twist = np.concatenate(
            [
                self.translation_velocity if moving else np.zeros(3),
                (
                    self.spatial_rotation_vector / self.duration_s
                    if moving and not self.position_only
                    else np.zeros(3)
                ),
            ]
        )
        return reference_pose, reference_twist

    def _velocity_bounds(self, joints):
        q = np.asarray(joints, dtype=float)
        dt = self.control_dt
        velocity_limit = self.ur5e.velocity_limits * float(
            self.config["joint_velocity_limit_fraction"]
        )
        margin = float(self.config["joint_position_margin_rad"])
        safe_lower = self.ur5e.lower + margin
        safe_upper = self.ur5e.upper - margin
        position_lower = (safe_lower - q) / dt
        position_upper = (safe_upper - q) / dt
        hard_lower = np.maximum(-velocity_limit, position_lower)
        hard_upper = np.minimum(velocity_limit, position_upper)
        if np.any(hard_lower > hard_upper):
            raise RuntimeError(
                "Current q is too far outside the configured safe joint range"
            )

        acceleration_limit = np.asarray(
            self.config["joint_acceleration_limit_radps2"], dtype=float
        )
        acceleration_lower = self.previous_joint_velocity - acceleration_limit * dt
        acceleration_upper = self.previous_joint_velocity + acceleration_limit * dt
        lower = np.maximum(hard_lower, acceleration_lower)
        upper = np.minimum(hard_upper, acceleration_upper)
        relaxed = []
        for index in range(6):
            if lower[index] > upper[index]:
                # Position and velocity are safety constraints. Acceleration is
                # relaxed only when it conflicts with one of those hard bounds.
                lower[index] = hard_lower[index]
                upper[index] = hard_upper[index]
                relaxed.append(index)
            if upper[index] - lower[index] < 1e-10:
                midpoint = 0.5 * (lower[index] + upper[index])
                lower[index] = midpoint - 5e-11
                upper[index] = midpoint + 5e-11
        return lower, upper, tuple(relaxed)

    def command(self, data, elapsed_s: float) -> DifferentialIKCommand:
        reference_pose, reference_twist = self.reference(elapsed_s)
        current_pose = self.ur5e.tcp_pose6(data)
        tracking_error = relative_pose_error6(reference_pose, current_pose)
        if self.position_only:
            tracking_error[3:] = 0.0
        feedback_twist = np.concatenate(
            [
                float(self.config["position_gain_s_inv"]) * tracking_error[:3],
                (
                    float(self.config["orientation_gain_s_inv"])
                    * tracking_error[3:]
                    if not self.position_only
                    else np.zeros(3)
                ),
            ]
        )
        commanded_twist = reference_twist + feedback_twist
        commanded_twist[:3] = _clip_norm(
            commanded_twist[:3],
            float(self.config["maximum_tcp_linear_speed_mps"]),
        )
        commanded_twist[3:] = _clip_norm(
            commanded_twist[3:],
            float(self.config["maximum_tcp_angular_speed_radps"]),
        )

        jacobian6 = self.ur5e.tcp_jacobian(data)
        if self.position_only:
            jacobian = jacobian6[:3]
            task_velocity = commanded_twist[:3]
            task_weight = np.full(3, float(self.config["position_task_weight"]))
        else:
            jacobian = jacobian6
            task_velocity = commanded_twist
            task_weight = np.concatenate(
                [
                    np.full(3, float(self.config["position_task_weight"])),
                    np.full(3, float(self.config["orientation_task_weight"])),
                ]
            )
        weighted_jacobian = task_weight[:, None] * jacobian
        weighted_velocity = task_weight * task_velocity
        regularization = float(self.config["regularization"])
        augmented_matrix = np.vstack(
            [weighted_jacobian, np.sqrt(regularization) * np.eye(6)]
        )
        augmented_target = np.concatenate([weighted_velocity, np.zeros(6)])
        lower, upper, relaxed = self._velocity_bounds(
            self.ur5e.joints(data)
        )

        started = perf_counter()
        result = lsq_linear(
            augmented_matrix,
            augmented_target,
            bounds=(lower, upper),
            method="trf",
            lsmr_tol="auto",
            max_iter=int(self.config["qp_max_iterations"]),
        )
        solve_time_ms = (perf_counter() - started) * 1000.0
        joint_velocity = np.clip(result.x, lower, upper)
        self.previous_joint_velocity = joint_velocity.copy()
        return DifferentialIKCommand(
            joint_velocity_radps=joint_velocity.astype(np.float64),
            reference_pose6=reference_pose.astype(np.float64),
            reference_twist6=reference_twist.astype(np.float64),
            commanded_twist6=commanded_twist.astype(np.float64),
            task_error6=tracking_error.astype(np.float64),
            lower_velocity_bound_radps=lower.astype(np.float64),
            upper_velocity_bound_radps=upper.astype(np.float64),
            acceleration_relaxed_joints=relaxed,
            qp_success=bool(result.success),
            qp_cost=float(result.cost),
            solve_time_ms=float(solve_time_ms),
        )


def simulate_time_constrained_ik(
    ur5e_model,
    controller_config: Dict,
    start_joints,
    goal_pose6,
    duration_s: float,
):
    """Run the constrained controller and return an auditable JSON-safe report."""
    data = ur5e_model.new_data()
    ur5e_model.reset_data(data, start_joints)
    start_pose = ur5e_model.tcp_pose6(data)
    goal = np.asarray(goal_pose6, dtype=float)
    controller = TimeConstrainedDifferentialIK(ur5e_model, controller_config)
    controller.reset(start_pose, goal, duration_s)
    command_q = ur5e_model.joints(data)
    dt = ur5e_model.control_dt
    maximum_time = float(duration_s) + float(controller_config["settling_time_s"])
    step_count = int(np.ceil(maximum_time / dt))
    tcp_positions = [start_pose[:3].copy()]
    trace = []
    contact_count = 0
    time_constraint_snapshot: Optional[Dict] = None
    relaxed_count = 0
    solve_times = []

    for step_index in range(step_count):
        elapsed = step_index * dt
        command = controller.command(data, elapsed)
        command_q = np.clip(
            command_q + command.joint_velocity_radps * dt,
            ur5e_model.lower,
            ur5e_model.upper,
        )
        contact = ur5e_model.step_position_target(data, command_q)
        contact_count += int(contact)
        actual_pose = ur5e_model.tcp_pose6(data)
        actual_velocity = ur5e_model.tcp_jacobian(data) @ ur5e_model.joint_velocities(data)
        goal_error = relative_pose_error6(goal, actual_pose)
        tcp_positions.append(actual_pose[:3].copy())
        relaxed_count += len(command.acceleration_relaxed_joints)
        solve_times.append(command.solve_time_ms)
        record = {
            "time_s": float((step_index + 1) * dt),
            "position_error_m": float(np.linalg.norm(goal_error[:3])),
            "orientation_error_deg": float(np.rad2deg(np.linalg.norm(goal_error[3:]))),
            "tcp_linear_speed_mps": float(np.linalg.norm(actual_velocity[:3])),
            "tcp_angular_speed_radps": float(np.linalg.norm(actual_velocity[3:])),
            "qp_success": bool(command.qp_success),
            "contact": bool(contact),
        }
        trace.append(record)
        if time_constraint_snapshot is None and record["time_s"] >= float(duration_s) - 1e-9:
            time_constraint_snapshot = dict(record)

    if time_constraint_snapshot is None:
        time_constraint_snapshot = dict(trace[-1])
    position_threshold = float(controller_config["success_position_m"])
    orientation_threshold = float(controller_config["success_orientation_deg"])
    speed_threshold = float(controller_config["success_tcp_speed_mps"])
    angular_speed_threshold = float(
        controller_config["success_tcp_angular_speed_radps"]
    )

    def is_success(record):
        orientation_ok = bool(controller_config.get("position_only", False)) or (
            record["orientation_error_deg"] <= orientation_threshold
            and record["tcp_angular_speed_radps"] <= angular_speed_threshold
        )
        return bool(
            record["position_error_m"] <= position_threshold
            and record["tcp_linear_speed_mps"] <= speed_threshold
            and orientation_ok
        )

    positions = np.asarray(tcp_positions, dtype=float)
    five_step_net = (
        np.linalg.norm(positions[5:] - positions[:-5], axis=1)
        if len(positions) > 5
        else np.empty(0)
    )
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    five_step_path = (
        np.convolve(segment_lengths, np.ones(5), mode="valid")
        if len(segment_lengths) >= 5
        else np.empty(0)
    )
    ten_step_net = (
        np.linalg.norm(positions[10:] - positions[:-10], axis=1)
        if len(positions) > 10
        else np.empty(0)
    )
    ten_step_path = (
        np.convolve(segment_lengths, np.ones(10), mode="valid")
        if len(segment_lengths) >= 10
        else np.empty(0)
    )

    def window_mean(values):
        return float(np.mean(values)) if len(values) else float("nan")

    def window_p95(values):
        return (
            float(np.percentile(values, 95.0))
            if len(values)
            else float("nan")
        )
    return {
        "controller": "time_parameterized_constrained_differential_ik_v1",
        "ur5e_contract": ur5e_model.contract_dict(),
        "start_joints_rad": np.asarray(start_joints, dtype=float).tolist(),
        "start_tcp_pose6": start_pose.tolist(),
        "goal_tcp_pose6": goal.tolist(),
        "prescribed_duration_s": float(duration_s),
        "success_at_prescribed_time": is_success(time_constraint_snapshot),
        "prescribed_time_state": time_constraint_snapshot,
        "success_after_settling": is_success(trace[-1]),
        "final_state": trace[-1],
        "contact_control_steps": int(contact_count),
        "acceleration_relaxation_count": int(relaxed_count),
        "qp_solve_time_mean_ms": float(np.mean(solve_times)),
        "qp_solve_time_p95_ms": float(np.percentile(solve_times, 95.0)),
        "tcp_50ms_net_displacement_mean_m": window_mean(five_step_net),
        "tcp_50ms_net_displacement_p95_m": window_p95(five_step_net),
        "tcp_50ms_path_length_mean_m": window_mean(five_step_path),
        "tcp_100ms_net_displacement_mean_m": window_mean(ten_step_net),
        "tcp_100ms_net_displacement_p95_m": window_p95(ten_step_net),
        "tcp_100ms_path_length_mean_m": window_mean(ten_step_path),
        "trace": trace,
    }
