"""JAiLeR-style UR5e reaching with time-multiplexed logical environments."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from cage_system.geometry import rotation_as_matrix, rotation_distance
from cage_system.learning.ppo2 import build_ppo_state, relative_pose_error6


def _wrapped_joint_distance(first, second):
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    wrapped = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(wrapped))


def _random_joint_candidate(rng, nominal, offset, lower, upper):
    return np.clip(nominal + rng.uniform(-offset, offset), lower, upper)


def generate_initial_joint_pool(ur5e, config: Dict, count: int, seed: int):
    """Generate diverse, collision-free q6 states against the selected MJCF."""
    settings = config["initial_pose_pool"]
    count = int(count)
    if count <= 0:
        raise ValueError("Initial joint pool count must be positive")
    rng = np.random.RandomState(int(seed))
    nominal = np.asarray(ur5e.robot["nominal_joint_state_rad"], dtype=float)
    offset = np.asarray(settings["joint_offset_max_rad"], dtype=float)
    margin = float(settings["joint_limit_margin_rad"])
    lower = ur5e.lower + margin
    upper = ur5e.upper - margin
    if nominal.shape != (6,) or offset.shape != (6,) or np.any(offset <= 0.0):
        raise ValueError("Initial pool needs nominal q6 and six positive offsets")
    if np.any(lower >= upper):
        raise ValueError("Initial pool margin consumes the joint range")
    nominal = np.clip(nominal, lower, upper)
    workspace_min = np.asarray(ur5e.robot["workspace_min_m"], dtype=float)
    workspace_max = np.asarray(ur5e.robot["workspace_max_m"], dtype=float)
    minimum_distance = float(settings["minimum_pairwise_joint_distance_rad"])
    maximum_attempts = int(settings["maximum_attempts"])
    data = ur5e.new_data()
    accepted = []
    tcp_poses = []

    for attempt in range(maximum_attempts):
        candidate = (
            nominal.copy()
            if attempt == 0
            else _random_joint_candidate(
                rng, nominal, offset, lower, upper
            )
        )
        if accepted and min(
            _wrapped_joint_distance(candidate, previous) for previous in accepted
        ) < minimum_distance:
            continue
        ur5e.reset_data(data, candidate)
        tcp_pose = ur5e.tcp_pose6(data)
        if ur5e.collision_count(data):
            continue
        if np.any(tcp_pose[:3] < workspace_min) or np.any(
            tcp_pose[:3] > workspace_max
        ):
            continue
        accepted.append(candidate.copy())
        tcp_poses.append(tcp_pose.copy())
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise RuntimeError(
            "Generated only {}/{} distinct collision-free UR5e initial poses "
            "after {} attempts; reduce the pairwise-distance threshold or "
            "increase maximum_attempts".format(
                len(accepted), count, maximum_attempts
            )
        )
    return {
        "joint_positions_rad": np.asarray(accepted, dtype=np.float64),
        "tcp_pose6": np.asarray(tcp_poses, dtype=np.float64),
        "joint_names": np.asarray(ur5e.joint_names),
        "tcp_site_name": np.asarray(ur5e.tcp_site_name),
        "model_path": np.asarray(ur5e.model_path),
        "seed": np.asarray(int(seed)),
    }


def save_initial_joint_pool(path: str, arrays: Dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(destination), **arrays)


def load_initial_joint_pool(path: str, ur5e, required_count: int):
    with np.load(path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joint_positions_rad"], dtype=float)
        joint_names = tuple(str(value) for value in archive["joint_names"].tolist())
        tcp_site_name = str(np.asarray(archive["tcp_site_name"]).item())
    if joints.ndim != 2 or joints.shape[1] != 6 or len(joints) < int(required_count):
        raise ValueError(
            "Initial pool must contain at least {} q6 rows".format(required_count)
        )
    if joint_names != ur5e.joint_names or tcp_site_name != ur5e.tcp_site_name:
        raise ValueError(
            "Initial pool joint order/TCP does not match the selected UR5e MJCF"
        )
    validation_data = ur5e.new_data()
    for index, joints_row in enumerate(joints[: int(required_count)]):
        ur5e.reset_data(validation_data, joints_row)
        if ur5e.collision_count(validation_data):
            raise ValueError(
                "Initial pool row {} is in contact in the selected MJCF".format(
                    index
                )
            )
    return joints[: int(required_count)].copy()


@dataclass
class JailerLogicalSlot:
    slot_id: int
    data: object
    command_q: np.ndarray
    previous_desired_velocity: np.ndarray
    target_pose6: np.ndarray
    tcp_history: deque
    episode_step_count: int = 0
    reset_count: int = 0
    target_sampling_fallback_count: int = 0
    initial_position_error_m: float = 0.0
    initial_orientation_error_deg: float = 0.0
    previous_position_error_m: float = 0.0
    previous_orientation_error_rad: float = 0.0


class JailerLogicalVectorEnv:
    """One shared UR5e model with N independent MuJoCo ``MjData`` slots.

    The scene never contains N robot bodies.  ``MjModel`` stores one exact
    UR5e morphology while each slot stores an independent dynamic state.
    """

    def __init__(self, ur5e, config: Dict, initial_joint_pool, seed: int):
        self.ur5e = ur5e
        self.config = config
        self.slot_count = int(config["logical_slots"])
        pool = np.asarray(initial_joint_pool, dtype=float)
        if pool.shape != (self.slot_count, 6):
            raise ValueError(
                "Logical slots require an exact [{},6] initial pool".format(
                    self.slot_count
                )
            )
        self.initial_joint_pool = pool.copy()
        self.rng = np.random.RandomState(int(seed))
        self.scratch_data = ur5e.new_data()
        self.safe_margin = float(config["joint_position_margin_rad"])
        self.safe_lower = ur5e.lower + self.safe_margin
        self.safe_upper = ur5e.upper - self.safe_margin
        self.action_velocity_limit = ur5e.velocity_limits * float(
            config["action_joint_velocity_limit_fraction"]
        )
        self.joint_acceleration_limit = np.asarray(
            config["joint_acceleration_limit_radps2"], dtype=float
        )
        if self.joint_acceleration_limit.shape != (6,) or np.any(
            self.joint_acceleration_limit <= 0.0
        ):
            raise ValueError("JAiLeR needs six positive acceleration limits")
        self.slots: List[JailerLogicalSlot] = []
        for slot_id, joints in enumerate(pool):
            data = ur5e.new_data()
            ur5e.reset_data(data, joints)
            tcp_pose = ur5e.tcp_pose6(data)
            # Eleven FK samples span ten 100 Hz control intervals.  This one
            # buffer supports both the five-interval MOVE50 and ten-interval
            # MOVE100 audits without estimating motion from commanded joints.
            history = deque(maxlen=11)
            history.append(tcp_pose[:3].copy())
            self.slots.append(
                JailerLogicalSlot(
                    slot_id=slot_id,
                    data=data,
                    command_q=joints.copy(),
                    previous_desired_velocity=np.zeros(6, dtype=float),
                    target_pose6=tcp_pose.copy(),
                    tcp_history=history,
                )
            )
        self.last_slot_order = np.arange(self.slot_count, dtype=int)

    @staticmethod
    def stage_task(stage):
        task = str(stage["task"])
        if task not in {"position", "pose"}:
            raise ValueError("Curriculum stage task must be position or pose")
        return task

    def _sample_target(self, slot: JailerLogicalSlot, stage: Dict):
        task = self.stage_task(stage)
        position_bounds = np.asarray(stage["position_error_range_m"], dtype=float)
        orientation_bounds = np.deg2rad(
            np.asarray(stage["orientation_error_range_deg"], dtype=float)
        )
        joint_offset = np.asarray(stage["target_joint_offset_max_rad"], dtype=float)
        maximum_attempts = int(self.config["target_sampling_max_attempts"])
        current_q = self.ur5e.joints(slot.data)
        current_pose = self.ur5e.tcp_pose6(slot.data)
        workspace_min = np.asarray(self.ur5e.robot["workspace_min_m"], dtype=float)
        workspace_max = np.asarray(self.ur5e.robot["workspace_max_m"], dtype=float)
        for _ in range(maximum_attempts):
            target_q = np.clip(
                current_q + self.rng.uniform(-joint_offset, joint_offset),
                self.safe_lower,
                self.safe_upper,
            )
            self.ur5e.reset_data(self.scratch_data, target_q)
            if self.ur5e.collision_count(self.scratch_data):
                continue
            target_pose = self.ur5e.tcp_pose6(self.scratch_data)
            position_error = float(
                np.linalg.norm(target_pose[:3] - current_pose[:3])
            )
            current_rotation = rotation_as_matrix(
                Rotation.from_rotvec(current_pose[3:])
            )
            target_rotation = rotation_as_matrix(
                Rotation.from_rotvec(target_pose[3:])
            )
            orientation_error = rotation_distance(
                current_rotation, target_rotation
            )
            if not position_bounds[0] <= position_error <= position_bounds[1]:
                continue
            if task == "pose" and not (
                orientation_bounds[0]
                <= orientation_error
                <= orientation_bounds[1]
            ):
                continue
            if np.any(target_pose[:3] < workspace_min) or np.any(
                target_pose[:3] > workspace_max
            ):
                continue
            return target_pose
        raise RuntimeError(
            "Could not sample a reachable target for curriculum stage '{}' "
            "from logical slot {}".format(stage["name"], slot.slot_id)
        )

    def _reset_slot_state(self, slot: JailerLogicalSlot, joints) -> None:
        """Reset one logical robot without carrying dynamics across tasks."""
        joints = np.asarray(joints, dtype=float)
        self.ur5e.reset_data(slot.data, joints)
        slot.command_q = joints.copy()
        slot.previous_desired_velocity = np.zeros(6, dtype=float)
        slot.target_pose6 = self.ur5e.tcp_pose6(slot.data)
        slot.tcp_history.clear()
        slot.tcp_history.append(slot.target_pose6[:3].copy())
        slot.episode_step_count = 0
        slot.previous_position_error_m = 0.0
        slot.previous_orientation_error_rad = 0.0

    def _start_slot_task(
        self,
        slot: JailerLogicalSlot,
        stage: Dict,
        preferred_pool_index=None,
    ) -> None:
        """Start a task from the pool, retrying another valid pose if needed."""
        pool_count = len(self.initial_joint_pool)
        shuffled = self.rng.permutation(pool_count).tolist()
        if preferred_pool_index is not None:
            preferred = int(preferred_pool_index)
            shuffled = [preferred] + [
                index for index in shuffled if index != preferred
            ]
        retry_limit = min(
            pool_count,
            int(self.config.get("target_sampling_pool_retries", pool_count)),
        )
        last_error = None
        for fallback_index, pool_index in enumerate(shuffled[:retry_limit]):
            self._reset_slot_state(slot, self.initial_joint_pool[pool_index])
            try:
                slot.target_pose6 = self._sample_target(slot, stage)
            except RuntimeError as error:
                last_error = error
                continue
            initial_error = relative_pose_error6(
                slot.target_pose6,
                self.ur5e.tcp_pose6(slot.data),
            )
            slot.initial_position_error_m = float(
                np.linalg.norm(initial_error[:3])
            )
            slot.initial_orientation_error_deg = float(
                np.rad2deg(np.linalg.norm(initial_error[3:]))
            )
            slot.previous_position_error_m = float(
                np.linalg.norm(initial_error[:3])
            )
            slot.previous_orientation_error_rad = float(
                np.linalg.norm(initial_error[3:])
            )
            slot.reset_count += 1
            slot.target_sampling_fallback_count += int(fallback_index)
            return
        raise RuntimeError(
            "Could not initialize curriculum stage '{}' for logical slot {} "
            "from {} collision-free pool poses: {}".format(
                stage["name"], slot.slot_id, retry_limit, last_error
            )
        )

    def begin_rollout(self, stage: Dict):
        """Start every rollout from a shuffled one-to-one initial-pose pool."""
        self.last_slot_order = self.rng.permutation(self.slot_count)
        pool_order = self.rng.permutation(self.slot_count)
        for slot_index, pool_index in zip(self.last_slot_order, pool_order):
            slot = self.slots[int(slot_index)]
            self._start_slot_task(
                slot,
                stage,
                preferred_pool_index=int(pool_index),
            )
        return self.last_slot_order.copy()

    def _state(self, slot: JailerLogicalSlot, stage: Dict):
        tcp_pose = self.ur5e.tcp_pose6(slot.data)
        state = build_ppo_state(
            slot.target_pose6,
            tcp_pose,
            self.ur5e.joints(slot.data),
            self.ur5e.joint_velocities(slot.data),
        )
        orientation_mode = str(
            stage.get("orientation_observation_mode", "observed")
        )
        if orientation_mode == "masked_zero":
            state[3:6] = 0.0
        elif orientation_mode != "observed":
            raise ValueError(
                "Unsupported orientation observation mode '{}'".format(
                    orientation_mode
                )
            )
        return state

    def observations(self, stage: Dict):
        return np.stack([self._state(slot, stage) for slot in self.slots])

    def _position_quality(self, stage, position_error):
        """Return composite, fine and coarse position qualities in [0, 1]."""
        error = max(0.0, float(position_error))
        mode = str(
            self.config.get(
                "reward_position_quality_mode", "legacy_single_exponential_v1"
            )
        )
        if mode == "legacy_single_exponential_v1":
            quality = np.exp(
                -float(self.config["reward_position_error_gain"]) * error ** 2
            )
            return float(quality), float(quality), float(quality)
        if mode != "stage_normalized_fine_exp_coarse_inverse_quadratic_v2":
            raise ValueError("Unsupported JAiLeR position-quality mode")

        fine_scale = float(stage["success_position_m"])
        coarse_scale = float(stage["position_error_range_m"][1])
        if fine_scale <= 0.0 or coarse_scale <= 0.0:
            raise ValueError("Position-quality scales must be positive")
        fine_quality = np.exp(
            -np.log(2.0) * (error / fine_scale) ** 2
        )
        coarse_quality = 1.0 / (1.0 + (error / coarse_scale) ** 2)
        fine_weight = float(self.config["reward_position_fine_quality_weight"])
        coarse_weight = 1.0 - fine_weight
        quality = fine_weight * fine_quality + coarse_weight * coarse_quality
        return (
            float(quality),
            float(fine_quality),
            float(coarse_quality),
        )

    def _pose_quality(
        self,
        slot,
        stage,
        current_pose,
        position_error,
        position_quality=None,
    ):
        if position_quality is None:
            position_quality = self._position_quality(
                stage, position_error
            )[0]
        target_rotation = rotation_as_matrix(
            Rotation.from_rotvec(slot.target_pose6[3:])
        )
        current_rotation = rotation_as_matrix(
            Rotation.from_rotvec(current_pose[3:])
        )
        orientation_error = rotation_distance(target_rotation, current_rotation)
        lever = float(self.config["orientation_marker_lever_m"])
        # Position already has its own quality term.  Anchor both virtual TCP
        # axis triplets at the same origin so this component measures rotation
        # only; including the two TCP translations here would reward position
        # twice and leave the pose residual with almost no orientation signal.
        target_markers = lever * target_rotation.T
        current_markers = lever * current_rotation.T
        marker_mse = float(
            np.mean(np.sum((target_markers - current_markers) ** 2, axis=1))
        )
        # The exponential marker quality used in the first pose prototype
        # becomes almost exactly zero after a large wrist excursion, leaving
        # PPO with little signal to recover.  This inverse-quadratic quality is
        # equally bounded in [0,1] near the goal but retains a useful gradient
        # for large orientation errors.
        orientation_quality = 1.0 / (
            1.0
            + float(self.config["reward_orientation_marker_gain"])
            * marker_mse
        )
        position_weight, orientation_weight = self._reward_weights(
            stage, position_error, orientation_error
        )
        quality = (
            position_weight * position_quality
            + orientation_weight * orientation_quality
        )
        return (
            float(quality),
            float(orientation_error),
            float(orientation_quality),
            marker_mse,
        )

    def _reward_weights(self, stage, position_error, orientation_error):
        # Position curricula are deliberately pure position learning.  The
        # observed pass exposes rotation inputs for normalization only.
        if self.stage_task(stage) == "position":
            return 1.0, 0.0
        if (
            str(self.config.get("reward_weight_mode", "fixed_stage_v1"))
            in {
                "dynamic_normalized_pose_error_v1",
                "dynamic_normalized_pose_error_achieved_floor_v2",
            }
            and self.stage_task(stage) == "pose"
        ):
            mode = str(self.config["reward_weight_mode"])
            minimum = float(self.config["dynamic_reward_min_weight"])
            power = float(self.config["dynamic_reward_urgency_power"])
            position_urgency = (
                float(position_error) / float(stage["success_position_m"])
            ) ** power
            orientation_urgency = (
                float(np.rad2deg(orientation_error))
                / float(stage["success_orientation_deg"])
            ) ** power
            position_achieved = bool(position_urgency <= 1.0)
            orientation_achieved = bool(orientation_urgency <= 1.0)
            if (
                mode == "dynamic_normalized_pose_error_achieved_floor_v2"
                and position_achieved != orientation_achieved
            ):
                orientation_weight = (
                    1.0 - minimum if position_achieved else minimum
                )
            elif position_achieved and orientation_achieved:
                orientation_weight = 0.5
            else:
                total_urgency = position_urgency + orientation_urgency
                orientation_weight = (
                    0.5
                    if total_urgency <= 1e-12
                    else orientation_urgency / total_urgency
                )
            orientation_weight = float(
                np.clip(orientation_weight, minimum, 1.0 - minimum)
            )
        else:
            orientation_weight = float(
                stage.get(
                    "reward_orientation_quality_weight",
                    0.0
                    if self.stage_task(stage) == "position"
                    else 1.0
                    - float(self.config["reward_position_quality_weight"]),
                )
            )
        position_weight = 1.0 - orientation_weight
        return float(position_weight), float(orientation_weight)

    @staticmethod
    def _component_progress(progress, achieved, mode):
        value = float(progress)
        if mode == "reward_both_directions_v1":
            return value
        if mode == "no_positive_after_component_success_v2":
            return min(value, 0.0) if bool(achieved) else value
        raise ValueError("Unsupported component-progress success mode")

    def _recover_slot(self, slot: JailerLogicalSlot, stage: Dict):
        pool_index = int(self.rng.randint(0, len(self.initial_joint_pool)))
        self._start_slot_task(
            slot,
            stage,
            preferred_pool_index=pool_index,
        )

    def step(
        self,
        normalized_actions,
        stage: Dict,
        horizon_boundary=False,
        active_mask=None,
        recover_terminated=True,
    ):
        actions = np.asarray(normalized_actions, dtype=float)
        if actions.shape != (self.slot_count, 6):
            raise ValueError(
                "Logical vector step expects [{},6] actions".format(
                    self.slot_count
                )
            )
        rewards = np.zeros(self.slot_count, dtype=np.float32)
        dones = np.zeros(self.slot_count, dtype=np.float32)
        infos = []
        active = (
            np.ones(self.slot_count, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (self.slot_count,):
            raise ValueError("Logical active mask must have shape [slot_count]")
        dt = self.ur5e.control_dt
        task_step_limit = int(stage["rollout_steps_per_slot"])
        if task_step_limit <= 0:
            raise ValueError("Stage rollout_steps_per_slot must be positive")
        for slot_index in self.last_slot_order:
            slot = self.slots[int(slot_index)]
            if not bool(active[slot.slot_id]):
                continue
            slot.episode_step_count += 1
            action = np.clip(actions[slot.slot_id], -1.0, 1.0)
            action_mode = str(
                self.config.get("ppo", {}).get(
                    "action_composition_mode", ""
                )
            )
            internally_scaled_action = action_mode in {
                "frozen_position_plus_error_scaled_pose_residual_v2",
                "joint_pose_actor_dynamic_speed_with_position_reference_v1",
                "joint_pose_actor_dynamic_speed_ppo_only_v2",
            }
            # Under the v2 policy contract the stage value caps only the pose
            # residual before composition.  The frozen position controller
            # must retain the scale=1 physical mapping on which it was trained.
            action_velocity_scale = (
                1.0
                if internally_scaled_action
                else float(stage["action_velocity_scale"])
            )
            desired_velocity = (
                action * self.action_velocity_limit * action_velocity_scale
            )
            desired_acceleration = (
                desired_velocity - slot.previous_desired_velocity
            ) / dt
            slot.previous_desired_velocity = desired_velocity.copy()
            slot.command_q = np.clip(
                slot.command_q + desired_velocity * dt,
                self.safe_lower,
                self.safe_upper,
            )
            contact = self.ur5e.step_position_target(slot.data, slot.command_q)
            current_pose = self.ur5e.tcp_pose6(slot.data)
            actual_tcp_twist = self.ur5e.tcp_jacobian(
                slot.data
            ) @ self.ur5e.joint_velocities(slot.data)
            pose_error = relative_pose_error6(slot.target_pose6, current_pose)
            position_error = float(np.linalg.norm(pose_error[:3]))
            (
                position_quality,
                position_fine_quality,
                position_coarse_quality,
            ) = self._position_quality(stage, position_error)
            quality, orientation_error, orientation_quality, marker_mse = (
                self._pose_quality(
                    slot,
                    stage,
                    current_pose,
                    position_error,
                    position_quality=position_quality,
                )
            )
            position_reward_weight, orientation_reward_weight = (
                self._reward_weights(
                    stage, position_error, orientation_error
                )
            )
            progress_mode = str(
                self.config.get(
                    "reward_position_progress_mode",
                    "normalized_error_reduction_v1",
                )
            )
            if progress_mode == "normalized_error_reduction_v1":
                position_progress_raw = (
                    slot.previous_position_error_m - position_error
                ) / float(self.config["reward_position_progress_reference_m"])
            elif progress_mode == "position_quality_delta_v2":
                previous_position_quality = self._position_quality(
                    stage, slot.previous_position_error_m
                )[0]
                position_progress_raw = (
                    position_quality - previous_position_quality
                ) / float(
                    self.config[
                        "reward_position_progress_quality_reference"
                    ]
                )
            else:
                raise ValueError("Unsupported position-progress mode")
            position_progress_raw = float(
                np.clip(position_progress_raw, -1.0, 1.0)
            )
            slot.previous_position_error_m = float(position_error)
            orientation_progress_reference = np.deg2rad(
                float(self.config["reward_orientation_progress_reference_deg"])
            )
            orientation_progress_raw = float(
                np.clip(
                    (
                        slot.previous_orientation_error_rad
                        - orientation_error
                    )
                    / orientation_progress_reference,
                    -1.0,
                    1.0,
                )
            )
            slot.previous_orientation_error_rad = float(orientation_error)
            component_progress_mode = str(
                self.config.get(
                    "reward_component_progress_after_success_mode",
                    "reward_both_directions_v1",
                )
            )
            position_achieved = bool(
                position_error <= float(stage["success_position_m"])
            )
            orientation_achieved = bool(
                self.stage_task(stage) == "pose"
                and orientation_error
                <= np.deg2rad(float(stage["success_orientation_deg"]))
            )
            # Once one component is inside its success band, getting even
            # closer earns no progress bonus.  Retreat remains negative,
            # while the always-active quality term protects achieved accuracy
            # as the other component is optimized.
            position_progress = self._component_progress(
                position_progress_raw,
                position_achieved,
                component_progress_mode,
            )
            orientation_progress = self._component_progress(
                orientation_progress_raw,
                orientation_achieved,
                component_progress_mode,
            )
            normalized_acceleration = desired_acceleration / self.joint_acceleration_limit
            acceleration_cost = float(np.mean(normalized_acceleration ** 2))
            step_quality_baseline = float(
                self.config["reward_step_quality_baseline"]
            )
            # Optional V14+ scaling keeps the useful shape of ``quality - 1``
            # while reducing the per-control-step time cost.  Omitting the
            # key is exactly the historical scale=1 reward contract.
            step_quality_cost_scale = float(
                self.config.get("reward_step_quality_cost_scale", 1.0)
            )
            step_quality_cost = (
                step_quality_cost_scale * step_quality_baseline
            )
            quality_reward = step_quality_cost_scale * quality
            acceleration_penalty = float(
                self.config["reward_joint_acceleration_weight"]
            ) * acceleration_cost
            orientation_progress_reward = (
                float(self.config["reward_orientation_progress_weight"])
                * orientation_reward_weight
                * orientation_progress
            )
            reward = (
                quality_reward
                - step_quality_cost
                - acceleration_penalty
                + orientation_progress_reward
            )
            near_target_progress_weight = 0.0
            position_progress_reward = 0.0
            if (
                self.stage_task(stage) == "position"
                or bool(self.config.get("joint_reward_progress_global", False))
            ):
                # Position curricula need an observable direction-of-progress
                # signal over their complete distance range.  In pose stages
                # the same term remains locally gated so it cannot dominate
                # orientation refinement far from the joint pose target.
                near_target_progress_weight = 1.0
            else:
                near_position = max(
                    0.0,
                    1.0
                    - position_error
                    / float(self.config["reward_near_target_position_m"]),
                )
                near_orientation = max(
                    0.0,
                    1.0
                    - orientation_error
                    / np.deg2rad(
                        float(
                            self.config[
                                "reward_near_target_orientation_deg"
                            ]
                        )
                    ),
                )
                near_target_progress_weight = float(
                    near_position * near_orientation
                )
            position_progress_reward = (
                float(self.config["reward_position_progress_weight"])
                * position_reward_weight
                * near_target_progress_weight
                * position_progress
            )
            reward += position_progress_reward
            workspace_min = np.asarray(
                self.ur5e.robot["workspace_min_m"], dtype=float
            )
            workspace_max = np.asarray(
                self.ur5e.robot["workspace_max_m"], dtype=float
            )
            workspace_violation = bool(
                np.any(current_pose[:3] < workspace_min)
                or np.any(current_pose[:3] > workspace_max)
            )
            finite = bool(
                np.all(np.isfinite(current_pose))
                and np.all(np.isfinite(self.ur5e.joints(slot.data)))
            )
            termination_semantics_mode = str(
                self.config.get("termination_semantics_mode", "legacy_v1")
            )
            compensated_failure_mode = bool(
                termination_semantics_mode
                == "time_limit_bootstrap_failure_compensation_v1"
            )
            if termination_semantics_mode not in {
                "legacy_v1",
                "time_limit_bootstrap_failure_compensation_v1",
            }:
                raise ValueError(
                    "Unsupported JAiLeR termination-semantics mode"
                )

            failure_terminal = bool(
                contact or workspace_violation or not finite
            )
            failure_fixed_penalty = 0.0
            failure_remaining_steps = 0
            failure_remaining_baseline_per_step = 0.0
            failure_discounted_remaining_compensation = 0.0
            constraint_penalty = 0.0
            if compensated_failure_mode and failure_terminal:
                # Contact and workspace failure can be true together.  The
                # fixed terminal penalty is nevertheless charged exactly once.
                failure_fixed_penalty = float(
                    self.config["collision_penalty"]
                )
                failure_remaining_steps = max(
                    0, task_step_limit - int(slot.episode_step_count)
                )
                failure_remaining_baseline_per_step = float(
                    step_quality_cost
                )
                gamma = float(self.config.get("ppo", {}).get("gamma", 0.99))
                if failure_remaining_steps > 0:
                    if abs(gamma - 1.0) <= 1e-12:
                        discounted_steps = float(failure_remaining_steps)
                    else:
                        discounted_steps = float(
                            gamma
                            * (1.0 - gamma ** failure_remaining_steps)
                            / (1.0 - gamma)
                        )
                    failure_discounted_remaining_compensation = float(
                        failure_remaining_baseline_per_step
                        * discounted_steps
                    )
                constraint_penalty = float(
                    failure_fixed_penalty
                    + failure_discounted_remaining_compensation
                )
            elif not compensated_failure_mode:
                # Preserve the historical behavior byte-for-byte when the new
                # termination contract is absent.
                if contact:
                    constraint_penalty += float(
                        self.config["collision_penalty"]
                    )
                if workspace_violation:
                    constraint_penalty += float(
                        self.config["collision_penalty"]
                    )
            if compensated_failure_mode and not np.isfinite(reward):
                # A non-finite physical transition must still yield a finite,
                # learnable terminal target rather than poisoning the batch.
                reward = -float(step_quality_cost)
            reward -= constraint_penalty

            slot.tcp_history.append(current_pose[:3].copy())
            net_50ms = np.nan
            path_50ms = np.nan
            net_100ms = np.nan
            path_100ms = np.nan
            history = np.asarray(slot.tcp_history, dtype=float)
            if len(history) >= 6:
                history_50ms = history[-6:]
                net_50ms = float(
                    np.linalg.norm(history_50ms[-1] - history_50ms[0])
                )
                path_50ms = float(
                    np.sum(
                        np.linalg.norm(
                            np.diff(history_50ms, axis=0), axis=1
                        )
                    )
                )
            if len(history) >= 11:
                history_100ms = history[-11:]
                net_100ms = float(
                    np.linalg.norm(history_100ms[-1] - history_100ms[0])
                )
                path_100ms = float(
                    np.sum(
                        np.linalg.norm(
                            np.diff(history_100ms, axis=0), axis=1
                        )
                    )
                )
            position_ok = position_error <= float(stage["success_position_m"])
            speed_ok = float(np.linalg.norm(actual_tcp_twist[:3])) <= float(
                stage["success_tcp_speed_mps"]
            )
            orientation_ok = True
            if self.stage_task(stage) == "pose":
                orientation_ok = orientation_error <= np.deg2rad(
                    float(stage["success_orientation_deg"])
                )
                speed_ok = speed_ok and float(
                    np.linalg.norm(actual_tcp_twist[3:])
                ) <= float(stage["success_tcp_angular_speed_radps"])
            task_success = bool(
                finite
                and not contact
                and not workspace_violation
                and position_ok
                and orientation_ok
                and speed_ok
            )
            if task_success:
                success_reward = float(self.config["success_bonus"])
                reward += success_reward
            else:
                success_reward = 0.0
            task_timeout = bool(
                slot.episode_step_count >= task_step_limit
                and not task_success
                and not contact
                and not workspace_violation
                and finite
            )
            task_terminated = bool(
                task_success
                or contact
                or workspace_violation
                or not finite
                or task_timeout
            )
            # Time-limit expiry is a sequence boundary, not a physical MDP
            # terminal.  GAE must stop before the next randomized task while
            # its TD target still bootstraps from V(next_state).  Legacy runs
            # retain the old zero-bootstrap timeout behavior.
            bootstrap_terminal = bool(
                task_terminated
                and not (
                    compensated_failure_mode and task_timeout
                )
            )
            # A rollout boundary is only a data-collection truncation when the
            # current task has not received its full stage-local time budget.
            # It stops GAE recursion, but it is not a failed task and the
            # Critic must bootstrap from the observed next state.
            rollout_truncated = bool(
                horizon_boundary and not task_terminated
            )
            sequence_boundary = bool(task_terminated or rollout_truncated)
            if task_success:
                terminal_reason = "success"
            elif contact:
                terminal_reason = "contact"
            elif workspace_violation:
                terminal_reason = "workspace_violation"
            elif not finite:
                terminal_reason = "non_finite_state"
            elif task_timeout:
                terminal_reason = "task_timeout"
            elif rollout_truncated:
                terminal_reason = "rollout_truncated"
            else:
                terminal_reason = "ongoing"
            rewards[slot.slot_id] = float(reward)
            dones[slot.slot_id] = float(sequence_boundary)
            infos.append(
                {
                    "slot_id": int(slot.slot_id),
                    "position_error_m": position_error,
                    "position_quality": position_quality,
                    "position_fine_quality": position_fine_quality,
                    "position_coarse_quality": position_coarse_quality,
                    "orientation_error_deg": float(np.rad2deg(orientation_error)),
                    "orientation_quality": orientation_quality,
                    "position_reward_weight": position_reward_weight,
                    "orientation_reward_weight": orientation_reward_weight,
                    "orientation_marker_mse_m2": marker_mse,
                    "position_progress": position_progress,
                    "position_progress_raw": position_progress_raw,
                    "position_progress_reward": position_progress_reward,
                    "orientation_progress": orientation_progress,
                    "orientation_progress_raw": orientation_progress_raw,
                    "orientation_progress_reward": orientation_progress_reward,
                    "near_target_progress_weight": near_target_progress_weight,
                    "quality_reward": quality_reward,
                    "raw_quality": quality,
                    "step_quality_cost_scale": step_quality_cost_scale,
                    "step_quality_cost": step_quality_cost,
                    "joint_acceleration_cost": acceleration_cost,
                    "joint_acceleration_penalty": acceleration_penalty,
                    "success_reward": success_reward,
                    "constraint_penalty": constraint_penalty,
                    "termination_semantics_mode": termination_semantics_mode,
                    "failure_terminal": failure_terminal,
                    "failure_fixed_penalty": failure_fixed_penalty,
                    "failure_remaining_steps": failure_remaining_steps,
                    "failure_remaining_baseline_per_step": (
                        failure_remaining_baseline_per_step
                    ),
                    "failure_discounted_remaining_compensation": (
                        failure_discounted_remaining_compensation
                    ),
                    "failure_total_penalty": constraint_penalty,
                    "action_velocity_scale": action_velocity_scale,
                    "tcp_linear_speed_mps": float(
                        np.linalg.norm(actual_tcp_twist[:3])
                    ),
                    "tcp_angular_speed_radps": float(
                        np.linalg.norm(actual_tcp_twist[3:])
                    ),
                    "tcp_50ms_net_displacement_m": net_50ms,
                    "tcp_50ms_path_length_m": path_50ms,
                    "tcp_100ms_net_displacement_m": net_100ms,
                    "tcp_100ms_path_length_m": path_100ms,
                    "contact": bool(contact),
                    "workspace_violation": workspace_violation,
                    "horizon_boundary": bool(horizon_boundary),
                    "rollout_truncated": rollout_truncated,
                    "task_timeout": task_timeout,
                    "task_success": task_success,
                    "episode_done": task_terminated,
                    "bootstrap_terminal": bootstrap_terminal,
                    "sequence_boundary": sequence_boundary,
                    "episode_step_count": int(slot.episode_step_count),
                    "task_sequence": int(slot.reset_count),
                    "initial_position_error_m": float(
                        slot.initial_position_error_m
                    ),
                    "initial_orientation_error_deg": float(
                        slot.initial_orientation_error_deg
                    ),
                    "previous_orientation_error_rad": float(
                        slot.previous_orientation_error_rad
                    ),
                    "previous_position_error_m": float(
                        slot.previous_position_error_m
                    ),
                    "terminal_reason": terminal_reason,
                    "reward": float(reward),
                }
            )
            if (
                task_terminated
                and not horizon_boundary
                and bool(recover_terminated)
            ):
                self._recover_slot(slot, stage)
        return self.observations(stage), rewards, dones, infos

    def state_dict(self):
        return {
            "initial_joint_pool": self.initial_joint_pool.copy(),
            "rng_state": self.rng.get_state(),
            "slots": [
                {
                    "qpos": self.ur5e.joints(slot.data),
                    "qvel": self.ur5e.joint_velocities(slot.data),
                    "command_q": slot.command_q.copy(),
                    "previous_desired_velocity": slot.previous_desired_velocity.copy(),
                    "target_pose6": slot.target_pose6.copy(),
                    "episode_step_count": int(slot.episode_step_count),
                    "reset_count": int(slot.reset_count),
                    "target_sampling_fallback_count": int(
                        slot.target_sampling_fallback_count
                    ),
                    "initial_position_error_m": float(
                        slot.initial_position_error_m
                    ),
                    "initial_orientation_error_deg": float(
                        slot.initial_orientation_error_deg
                    ),
                    "previous_position_error_m": float(
                        slot.previous_position_error_m
                    ),
                    "previous_orientation_error_rad": float(
                        slot.previous_orientation_error_rad
                    ),
                }
                for slot in self.slots
            ],
        }

    def load_state_dict(self, state):
        pool = np.asarray(state["initial_joint_pool"], dtype=float)
        if pool.shape != self.initial_joint_pool.shape or not np.allclose(
            pool, self.initial_joint_pool
        ):
            raise ValueError("Resume checkpoint uses a different initial joint pool")
        self.rng.set_state(state["rng_state"])
        slot_states = state["slots"]
        if len(slot_states) != self.slot_count:
            raise ValueError("Resume checkpoint logical-slot count changed")
        for slot, saved in zip(self.slots, slot_states):
            self.ur5e.reset_data(slot.data, saved["qpos"], saved["qvel"])
            slot.command_q = np.asarray(saved["command_q"], dtype=float).copy()
            slot.previous_desired_velocity = np.asarray(
                saved["previous_desired_velocity"], dtype=float
            ).copy()
            slot.target_pose6 = np.asarray(saved["target_pose6"], dtype=float).copy()
            slot.episode_step_count = int(saved.get("episode_step_count", 0))
            slot.reset_count = int(saved["reset_count"])
            slot.target_sampling_fallback_count = int(
                saved.get("target_sampling_fallback_count", 0)
            )
            saved_initial_position = saved.get("initial_position_error_m")
            saved_initial_orientation = saved.get(
                "initial_orientation_error_deg"
            )
            if (
                saved_initial_position is None
                or saved_initial_orientation is None
            ):
                initial_error = relative_pose_error6(
                    slot.target_pose6,
                    self.ur5e.tcp_pose6(slot.data),
                )
                slot.initial_position_error_m = float(
                    np.linalg.norm(initial_error[:3])
                )
                slot.initial_orientation_error_deg = float(
                    np.rad2deg(np.linalg.norm(initial_error[3:]))
                )
                slot.previous_position_error_m = float(
                    np.linalg.norm(initial_error[:3])
                )
                slot.previous_orientation_error_rad = float(
                    np.linalg.norm(initial_error[3:])
                )
            else:
                slot.initial_position_error_m = float(saved_initial_position)
                slot.initial_orientation_error_deg = float(
                    saved_initial_orientation
                )
                current_error = relative_pose_error6(
                    slot.target_pose6,
                    self.ur5e.tcp_pose6(slot.data),
                )
                slot.previous_position_error_m = float(
                    saved.get(
                        "previous_position_error_m",
                        np.linalg.norm(current_error[:3]),
                    )
                )
                slot.previous_orientation_error_rad = float(
                    saved.get(
                        "previous_orientation_error_rad",
                        np.deg2rad(slot.initial_orientation_error_deg),
                    )
                )
            slot.tcp_history.clear()
            slot.tcp_history.append(self.ur5e.tcp_pose6(slot.data)[:3].copy())
