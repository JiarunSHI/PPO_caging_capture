from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from cage_system.geometry import rotation_distance
from cage_system.types import CandidatePose


@dataclass
class BackendResult:
    reachable: bool
    collision_free: bool
    joints: Optional[np.ndarray]
    reachability_margin: float
    collision_distance: float
    reason: str = ""
    solver_source: str = ""
    attempted_seed_count: int = 1


class FeasibilityBackend(Protocol):
    name: str

    def solve(self, transform_base_tcp: np.ndarray, seed_joints: np.ndarray) -> BackendResult:
        ...


class WorkspaceFeasibilityBackend:
    """Geometry-only smoke-test backend. It is forbidden in production mode."""

    name = "workspace"

    def __init__(self, robot_config):
        self.minimum = np.asarray(robot_config["workspace_min_m"], dtype=float)
        self.maximum = np.asarray(robot_config["workspace_max_m"], dtype=float)

    def solve(self, transform_base_tcp: np.ndarray, seed_joints: np.ndarray) -> BackendResult:
        position = transform_base_tcp[:3, 3]
        margins = np.concatenate((position - self.minimum, self.maximum - position))
        reachable = bool(np.min(margins) >= 0.0)
        return BackendResult(
            reachable=reachable,
            collision_free=reachable,
            joints=np.asarray(seed_joints, dtype=float).copy() if reachable else None,
            reachability_margin=float(np.min(margins)),
            collision_distance=float(np.min(margins)),
            reason="" if reachable else "outside_workspace",
        )


class CandidateFeasibilityChecker:
    def __init__(
        self,
        robot_config,
        candidate_config,
        timing_config,
        backend: FeasibilityBackend,
    ):
        self.robot = robot_config
        self.candidate = candidate_config
        self.backend = backend
        self.lower = np.asarray(robot_config["joint_lower_rad"], dtype=float)
        self.upper = np.asarray(robot_config["joint_upper_rad"], dtype=float)
        self.velocity_limits = np.asarray(robot_config["joint_velocity_limit_radps"], dtype=float)
        self.event_delta_limits = np.asarray(
            robot_config["normalized_delta_max_rad_per_vision_update"], dtype=float
        )
        self.visual_period = 1.0 / float(timing_config["detection_rate_hz"])

    def check(
        self,
        candidate: CandidatePose,
        current_joints: np.ndarray,
        current_joint_velocity: np.ndarray,
        available_dt: float,
        previous_best_transform: Optional[np.ndarray] = None,
    ) -> CandidatePose:
        reasons = []
        pose_jump = 0.0
        orientation_jump = 0.0
        temporal_continuity_cost = 0.0
        if previous_best_transform is not None:
            pose_jump = float(
                np.linalg.norm(candidate.tcp_transform_base[:3, 3] - previous_best_transform[:3, 3])
            )
            orientation_jump = rotation_distance(
                previous_best_transform[:3, :3], candidate.tcp_transform_base[:3, :3]
            )
            maximum_pose_jump = float(self.candidate["maximum_pose_jump_m"])
            maximum_orientation_jump = np.deg2rad(
                float(self.candidate["maximum_orientation_jump_deg"])
            )
            temporal_continuity_cost = 0.5 * (
                pose_jump / maximum_pose_jump
                + orientation_jump / maximum_orientation_jump
            )
            # These values are audit-only. PPO is responsible for realizing a
            # sequence of hard-feasible Cage targets; temporal tracking cost
            # must not alter the offline EST label optimum.

        # Cartesian workspace is the only inexpensive hard prefilter. The
        # analytic/multi-start backend then decides actual IK and collision.
        tcp_position = candidate.tcp_transform_base[:3, 3]
        workspace_minimum = np.asarray(self.robot["workspace_min_m"], dtype=float)
        workspace_maximum = np.asarray(self.robot["workspace_max_m"], dtype=float)
        workspace_margin = float(
            np.min(
                np.concatenate(
                    (
                        tcp_position - workspace_minimum,
                        workspace_maximum - tcp_position,
                    )
                )
            )
        )
        if workspace_margin < 0.0:
            reasons.append("outside_workspace")

        if reasons:
            candidate.rejection_reasons.extend(reasons)
            candidate.feasible = False
            candidate.metrics.update(
                {
                    "reachability_margin": workspace_margin,
                    "collision_distance": -np.inf,
                    "joint_limit_margin": -np.inf,
                    "joint_delta": np.inf,
                    "available_visual_events": 0,
                    "pose_jump": pose_jump,
                    "orientation_jump": orientation_jump,
                    "temporal_continuity_cost": temporal_continuity_cost,
                    "feasibility_stage": "cartesian_prefilter",
                }
            )
            return candidate

        result = self.backend.solve(candidate.tcp_transform_base, current_joints)
        reachable = bool(result.reachable and result.joints is not None)
        if not reachable:
            reasons.append(result.reason or "ik_unreachable")
        else:
            if not result.collision_free:
                reasons.append(result.reason or "collision")
            elif result.collision_distance < float(
                self.robot["minimum_collision_distance_m"]
            ):
                reasons.append("collision_clearance_too_small")

        joint_margin = -np.inf
        joint_delta = np.inf
        available_events = 0
        if result.joints is not None:
            joints = np.asarray(result.joints, dtype=float)
            if np.any(joints < self.lower) or np.any(joints > self.upper):
                reasons.append("joint_position_limit")
            available_events = max(
                1, int(round(float(available_dt) / self.visual_period))
            )
            joint_margin = float(
                np.min(np.minimum(joints - self.lower, self.upper - joints))
            )
            joint_delta = float(np.linalg.norm(joints - current_joints))
            candidate.ik_joints = joints

        candidate.rejection_reasons.extend(reasons)
        candidate.feasible = not reasons
        candidate.metrics.update(
            {
                "reachability_margin": float(result.reachability_margin),
                "collision_distance": float(result.collision_distance),
                "joint_limit_margin": joint_margin,
                "joint_delta": joint_delta,
                "available_visual_events": available_events if result.joints is not None else 0,
                "pose_jump": pose_jump,
                "orientation_jump": orientation_jump,
                "temporal_continuity_cost": temporal_continuity_cost,
                "ik_solver_source": result.solver_source,
                "ik_attempted_seed_count": int(result.attempted_seed_count),
                "feasibility_stage": "full_ik_collision",
            }
        )
        return candidate
