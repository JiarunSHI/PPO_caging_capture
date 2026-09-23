from collections import deque
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from cage_system.geometry import (
    pose6_to_transform,
    rotation_as_matrix,
    rotation_distance,
    rotation_from_matrix,
)
from cage_system.geometry import make_transform, normalize, orthonormal_frame, plane_basis
from cage_system.cage.candidates import transform_from_config
from cage_system.learning.online_interface import HistoricalCageInference
from cage_system.learning.ppo2 import build_ppo_state
from cage_system.learning.jacobian_expert import DampedJacobianExpert
from cage_system.learning.privileged_recovery import (
    project_predictions_to_known_trajectory,
    recovery_history_matrix,
    truth_history_matrix,
)
from cage_system.perception.eye_on_hand import MujocoEyeOnHandCamera
from cage_system.sim.mujoco_feasibility import MujocoFeasibilityBackend

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GymBase = gym.Env
except ImportError:
    gym = None
    spaces = None
    _GymBase = object


class MujocoUR5eCageEnv(_GymBase):
    """Small Gymnasium-compatible UR5e joint-increment environment.

    The MJCF must expose the six standard UR joint names and a TCP site named
    by ``tcp_site_name``. Target sequences are actual outputs of the frozen
    Cage network, not ground-truth labels.
    """

    def __init__(self, model_path: str, target_sequences: List, config, tcp_site_name="attachment_site"):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("MuJoCo is optional; install the simulation extra first") from error
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.config = config
        self.robot = config["robot"]
        self.ppo = config["ppo"]
        self.timing = config["timing"]
        self.target_sequences = target_sequences
        if not self.target_sequences:
            raise ValueError("At least one frozen Cage target sequence is required")
        self.sequence_index = -1
        self.sequence = None
        self.elapsed = 0.0
        # One environment/PPO action is triggered by each completed visual update.
        # MuJoCo still integrates its low-level actuator dynamics at the MJCF
        # timestep, but there is no independent high-rate PPO decision loop.
        self.decision_dt = 1.0 / float(self.timing["detection_rate_hz"])
        self.low_level_steps = max(
            1, int(round(self.decision_dt / self.model.opt.timestep))
        )
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.robot["joint_names"]]
        )
        if np.any(self.joint_ids < 0):
            raise ValueError("MJCF is missing one or more standard UR5e joints")
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids]
        self.tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name
        )
        if self.tcp_site_id < 0:
            raise ValueError("MJCF is missing TCP site '{}'".format(tcp_site_name))
        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        actuator_ids = []
        for joint_id in self.joint_ids:
            matches = np.flatnonzero(actuator_joint_ids == joint_id)
            if not len(matches):
                raise ValueError("Every UR5e joint needs a position actuator")
            actuator_ids.append(matches[0])
        self.actuator_ids = np.asarray(actuator_ids)
        self.lower = np.asarray(self.robot["joint_lower_rad"])
        self.upper = np.asarray(self.robot["joint_upper_rad"])
        self.delta_max = np.asarray(
            self.robot["normalized_delta_max_rad_per_vision_update"]
        )
        self.current_target = np.zeros(6)
        self.jacobian_expert = DampedJacobianExpert(
            self, config.get("expert_training", {}).get("jacobian", {})
        ) if config.get("expert_training", {}).get("jacobian") else None
        if spaces is not None:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(18,), dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        if gym is not None:
            super().reset(seed=seed)
        elif seed is not None:
            np.random.seed(seed)
        self.mujoco.mj_resetData(self.model, self.data)
        self.sequence_index = (self.sequence_index + 1) % len(self.target_sequences)
        self.sequence = self.target_sequences[self.sequence_index]
        self.sequence.reset()
        self.elapsed = 0.0
        home = np.asarray(self.robot["nominal_joint_state_rad"], dtype=float)
        self.data.qpos[self.qpos_addresses] = home
        self.data.ctrl[self.actuator_ids] = home
        self.mujoco.mj_forward(self.model, self.data)
        self.current_target = self.sequence(self.elapsed)
        return self._observation(), {}

    @property
    def action_ready(self):
        return True

    def _apply_robot_bias_compensation(self):
        if bool(self.robot["compensate_bias_forces_in_position_control"]):
            # qfrc_bias contains gravity, Coriolis and centrifugal terms. A
            # real UR/ros_control loop compensates these below the policy.
            self.data.qfrc_applied[self.dof_addresses] = (
                self.data.qfrc_bias[self.dof_addresses]
            )
        else:
            self.data.qfrc_applied[self.dof_addresses] = 0.0

    def _execute_position_command(self, command):
        """Integrate the low-level position servo with UR-style bias compensation."""
        self.data.ctrl[self.actuator_ids] = np.asarray(command, dtype=float)
        contact_collision = False
        for _ in range(self.low_level_steps):
            self._apply_robot_bias_compensation()
            self.mujoco.mj_step(self.model, self.data)
            contact_collision = contact_collision or bool(self.data.ncon > 0)
        return contact_collision

    def step(self, normalized_action):
        # This is the Cage pose contained in s_t and therefore the target that
        # conditioned a_t.  Hold it fixed for the complete transition reward;
        # the sequence may advance only when constructing s_{t+1}.
        reward_target = self.current_target.copy()
        action_state_relative_error = self._observation()[:6].copy()
        action = np.clip(np.asarray(normalized_action, dtype=float), -1.0, 1.0)
        joints = self.data.qpos[self.qpos_addresses].copy()
        velocity_bound = np.asarray(self.robot["joint_velocity_limit_radps"]) * self.decision_dt
        safe_delta = np.minimum(self.delta_max, velocity_bound)
        command = np.clip(joints + safe_delta * action, self.lower, self.upper)
        contact_collision = self._execute_position_command(command)
        reward, reward_info = self._reward(
            command - joints,
            reward_target,
            action_state_relative_error,
        )
        if contact_collision:
            reward -= float(self.ppo["collision_penalty"])
        self.elapsed += self.decision_dt
        self.current_target = self.sequence(self.elapsed)
        observation = self._observation()
        reward_info.update(
            {
                "contact_collision": contact_collision,
                "reward_cage_pose6": reward_target.copy(),
                "next_cage_pose6": self.current_target.copy(),
                "action_state_relative_error6": action_state_relative_error,
                "next_state_relative_error6": observation[:6].copy(),
                "reward_target_contract": "same_cage_pose_as_action_state",
            }
        )
        terminated = bool(
            (self.ppo.get("terminate_on_contact", True) and contact_collision)
            or not np.all(np.isfinite(observation))
        )
        truncated = self.elapsed >= self.sequence.duration
        return observation, reward, terminated, truncated, reward_info

    def _observation(self):
        joints = self.data.qpos[self.qpos_addresses].copy()
        velocities = self.data.qvel[self.dof_addresses].copy()
        return build_ppo_state(
            self.current_target, self.tcp_pose6(), joints, velocities
        )

    def tcp_pose6(self):
        return np.concatenate(
            [
                self.data.site_xpos[self.tcp_site_id].copy(),
                rotation_from_matrix(
                    self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
                ).as_rotvec(),
            ]
        )

    def expert_action(self):
        if self.jacobian_expert is None:
            raise RuntimeError("expert_training.jacobian is not configured")
        return self.jacobian_expert.action(self.current_target)

    @staticmethod
    def _random_unit_vector(rng):
        for _ in range(32):
            value = np.asarray(rng.normal(size=3), dtype=float)
            norm = float(np.linalg.norm(value))
            if norm > 1e-9:
                return value / norm
        raise RuntimeError("Could not sample a non-zero random direction")

    def reset_random_local_expert(self, rng, demonstration_config):
        """Create a trajectory-free random local reaching problem for BC."""
        settings = demonstration_config
        nominal = np.asarray(self.robot["nominal_joint_state_rad"], dtype=float)
        offset = np.asarray(settings["joint_offset_max_rad"], dtype=float)
        margin = float(settings["joint_limit_margin_rad"])
        lower = self.lower + margin
        upper = self.upper - margin
        workspace_min = np.asarray(self.robot["workspace_min_m"], dtype=float)
        workspace_max = np.asarray(self.robot["workspace_max_m"], dtype=float)
        position_range = np.asarray(settings["position_error_range_m"], dtype=float)
        orientation_range = np.deg2rad(
            np.asarray(settings["orientation_error_range_deg"], dtype=float)
        )
        attempts = int(settings["maximum_reset_attempts"])

        for attempt in range(1, attempts + 1):
            self.mujoco.mj_resetData(self.model, self.data)
            # The rendered object is irrelevant to this trajectory-free stage.
            # Moving the mocap body away removes contacts and visual dependence.
            if hasattr(self, "target_mocap_id"):
                self.data.mocap_pos[self.target_mocap_id] = np.array(
                    [0.0, 0.0, -5.0], dtype=float
                )
            joints = nominal + rng.uniform(-offset, offset)
            joints = np.clip(joints, lower, upper)
            velocity_std = float(settings["joint_velocity_std_radps"])
            velocities = rng.normal(0.0, velocity_std, size=6)
            velocity_limits = np.asarray(
                self.robot["joint_velocity_limit_radps"], dtype=float
            )
            velocities = np.clip(velocities, -velocity_limits, velocity_limits)
            self.data.qpos[self.qpos_addresses] = joints
            self.data.qvel[self.dof_addresses] = velocities
            self.data.ctrl[self.actuator_ids] = joints
            self.mujoco.mj_forward(self.model, self.data)
            if self.data.ncon > 0:
                continue

            tcp_position = self.data.site_xpos[self.tcp_site_id].copy()
            position_error = self._random_unit_vector(rng) * rng.uniform(
                position_range[0], position_range[1]
            )
            target_position = tcp_position + position_error
            if np.any(target_position < workspace_min) or np.any(
                target_position > workspace_max
            ):
                continue

            rotation_error = self._random_unit_vector(rng) * rng.uniform(
                orientation_range[0], orientation_range[1]
            )
            tcp_rotation = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
            target_rotation = (
                Rotation.from_rotvec(rotation_error).as_matrix() @ tcp_rotation
            )
            self.current_target = np.concatenate(
                [
                    target_position,
                    rotation_from_matrix(target_rotation).as_rotvec(),
                ]
            )
            self.elapsed = 0.0
            return self._observation(), {
                "source": "random_local_static_cage",
                "reset_attempts": int(attempt),
                "initial_position_error_m": float(np.linalg.norm(position_error)),
                "initial_orientation_error_rad": float(np.linalg.norm(rotation_error)),
            }
        raise RuntimeError(
            "Could not sample a collision-free random local expert state after "
            "{} attempts".format(attempts)
        )

    def step_fixed_expert_target(self, normalized_action, demonstration_config):
        """Advance one 50 ms expert step without a trajectory or vision update."""
        reward_target = self.current_target.copy()
        action_state_relative_error = self._observation()[:6].copy()
        action = np.clip(np.asarray(normalized_action, dtype=float), -1.0, 1.0)
        joints = self.data.qpos[self.qpos_addresses].copy()
        velocity_bound = (
            np.asarray(self.robot["joint_velocity_limit_radps"], dtype=float)
            * self.decision_dt
        )
        safe_delta = np.minimum(self.delta_max, velocity_bound)
        command = np.clip(joints + safe_delta * action, self.lower, self.upper)
        contact_collision = self._execute_position_command(command)
        self.elapsed += self.decision_dt
        observation = self._observation()
        reward, reward_info = self._reward(
            command - joints,
            reward_target,
            action_state_relative_error,
        )
        position_success = reward_info["position_error_m"] <= float(
            demonstration_config["success_position_m"]
        )
        orientation_success = reward_info["orientation_error_rad"] <= np.deg2rad(
            float(demonstration_config["success_orientation_deg"])
        )
        reward_info.update(
            {
                "contact_collision": bool(contact_collision),
                "local_target_success": bool(position_success and orientation_success),
                "reward_cage_pose6": reward_target.copy(),
                "action_state_relative_error6": action_state_relative_error,
                "next_state_relative_error6": observation[:6].copy(),
                "executed_normalized_action": action.astype(np.float32),
                "target_source": "random_local_static_cage",
            }
        )
        terminated = bool(contact_collision or not np.all(np.isfinite(observation)))
        return observation, reward, terminated, False, reward_info

    def _reward(
        self,
        joint_delta,
        cage_target_pose6,
        before_relative_error6=None,
    ):
        target = np.asarray(cage_target_pose6, dtype=float)
        if target.shape != (6,):
            raise ValueError("Reward Cage target must be one Base-frame pose6")
        tcp_position = self.data.site_xpos[self.tcp_site_id].copy()
        tcp_rotation = self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
        target_rotation = rotation_as_matrix(Rotation.from_rotvec(target[3:]))
        position_error = float(np.linalg.norm(tcp_position - target[:3]))
        orientation_error = rotation_distance(target_rotation, tcp_rotation)
        joints = self.data.qpos[self.qpos_addresses]
        velocities = self.data.qvel[self.dof_addresses]
        center = 0.5 * (self.lower + self.upper)
        half_range = np.maximum(0.5 * (self.upper - self.lower), 1e-6)
        joint_limit_cost = float(
            np.mean(((joints - center) / half_range) ** 4)
        )
        velocity_limits = np.maximum(
            np.asarray(self.robot["joint_velocity_limit_radps"], dtype=float),
            1e-6,
        )
        velocity_cost = float(np.mean((velocities / velocity_limits) ** 2))
        velocity_bound = velocity_limits * self.decision_dt
        safe_delta = np.maximum(np.minimum(self.delta_max, velocity_bound), 1e-6)
        action_cost = float(
            np.mean((np.asarray(joint_delta, dtype=float) / safe_delta) ** 2)
        )
        joint_cost = float(
            joint_limit_cost + 0.05 * np.mean(velocities ** 2)
            + 0.1 * np.mean(np.asarray(joint_delta) ** 2)
        )
        reward_mode = str(self.ppo.get("reward_mode", "legacy_squared"))
        reward_details = {}
        if reward_mode == "legacy_squared":
            reward = (
                -float(self.ppo["reward_position_weight"]) * position_error ** 2
                -float(self.ppo["reward_orientation_weight"]) * orientation_error ** 2
                -float(self.ppo["reward_joint_weight"]) * joint_cost
            )
        elif reward_mode in {
            "normalized_cage_tracking_v1",
            "normalized_cage_tracking_v2",
        }:
            position_reference = float(
                self.ppo["reward_position_reference_m"]
            )
            orientation_reference = np.deg2rad(
                float(self.ppo["reward_orientation_reference_deg"])
            )
            if reward_mode == "normalized_cage_tracking_v1":
                quality_mapping = "exponential_squared"

                def quality(error, reference):
                    return float(np.exp(-((error / reference) ** 2)))

            else:
                quality_mapping = "inverse_quadratic"

                def quality(error, reference):
                    ratio = error / reference
                    return float(1.0 / (1.0 + ratio ** 2))

            position_quality = quality(position_error, position_reference)
            orientation_quality = quality(
                orientation_error, orientation_reference
            )
            position_weight = float(
                self.ppo["reward_position_quality_weight"]
            )
            orientation_weight = float(
                self.ppo["reward_orientation_quality_weight"]
            )
            quality_after = (
                position_weight * position_quality
                + orientation_weight * orientation_quality
            )
            before = np.asarray(
                before_relative_error6
                if before_relative_error6 is not None
                else np.concatenate(
                    [
                        target[:3] - tcp_position,
                        np.zeros(3, dtype=float),
                    ]
                ),
                dtype=float,
            )
            if before.shape != (6,):
                raise ValueError("Reward before-error must contain six values")
            before_position = float(np.linalg.norm(before[:3]))
            before_orientation = float(np.linalg.norm(before[3:]))
            before_position_quality = quality(
                before_position, position_reference
            )
            before_orientation_quality = quality(
                before_orientation, orientation_reference
            )
            quality_before = (
                position_weight * before_position_quality
                + orientation_weight * before_orientation_quality
            )
            success = bool(
                position_error <= float(self.ppo["success_position_m"])
                and orientation_error
                <= np.deg2rad(float(self.ppo["success_orientation_deg"]))
            )
            tracking_reward = (
                quality_after - 1.0
                if reward_mode == "normalized_cage_tracking_v1"
                else quality_after
            )
            progress_reward = float(
                self.ppo["reward_progress_weight"]
            ) * (quality_after - quality_before)
            success_reward = (
                float(self.ppo["reward_success_bonus"]) if success else 0.0
            )
            safety_cost = (
                float(self.ppo["reward_joint_limit_weight"])
                * joint_limit_cost
                + float(self.ppo["reward_velocity_weight"]) * velocity_cost
                + float(self.ppo["reward_action_weight"]) * action_cost
            )
            reward = tracking_reward + progress_reward + success_reward - safety_cost
            reward_details = {
                "tracking_quality_before": float(quality_before),
                "tracking_quality_after": float(quality_after),
                "tracking_quality_mapping": quality_mapping,
                "tracking_reward": float(tracking_reward),
                "progress_reward": float(progress_reward),
                "success_reward": float(success_reward),
                "pose_success": bool(success),
                "safety_cost": float(safety_cost),
            }
        else:
            raise ValueError("Unsupported PPO reward_mode '{}'".format(reward_mode))
        return reward, {
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "joint_cost": joint_cost,
            "joint_limit_cost": joint_limit_cost,
            "joint_velocity_cost": velocity_cost,
            "joint_action_cost": action_cost,
            "reward_mode": reward_mode,
            **reward_details,
        }


class _StaticTargetSequence:
    duration = 1.0

    def reset(self):
        return None

    def __call__(self, _elapsed):
        return np.zeros(6, dtype=float)


class MujocoVisionUR5eCageEnv(MujocoUR5eCageEnv):
    """Vision-in-the-loop UR5e environment with a rendered eye-on-hand RGB-D camera.

    The mocap target follows the continuous stored trajectory.  At the detection
    rate the renderer produces RGB and depth, the perception module recovers a
    Base-frame center point, and the strict five-frame causal interface updates
    the frozen Cage target. PPO observes TCP-to-Cage error6 + q6 + qdot6.
    """

    def __init__(
        self,
        model_path: str,
        trajectories: List,
        frozen_cage_network,
        config,
        tcp_site_name="attachment_site",
        detector_device="auto",
    ):
        if not trajectories:
            raise ValueError("Vision-in-the-loop training requires raw target trajectories")
        super().__init__(
            model_path, [_StaticTargetSequence()], config, tcp_site_name=tcp_site_name
        )
        self.trajectories = list(trajectories)
        self.frozen_cage_network = frozen_cage_network
        self.trajectory_index = -1
        self.object_trajectory = None
        self.object_start_time = 0.0
        self.object_time = 0.0
        self.detection_period = self.decision_dt
        self.last_vision_detection = None
        self.last_valid_detection_time = None
        self.missed_detection_count = 0
        self.rejected_cage_jump_count = 0
        self.last_cage_update_time = None
        self.cage_action_ready = False
        self.cage_hold_reason = "insufficient_five_frame_history"
        self.training_reset = config["ppo_training"]
        self.consecutive_missed_detections = 0
        self.recent_detected_positions = deque(maxlen=2)
        self.missing_kalman_positions = []
        self.privileged_recovery_count = 0
        self._privileged_initial_cache = {}
        vision = config["vision"]
        target_body_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_BODY,
            str(vision["target_mocap_body"]),
        )
        if target_body_id < 0 or self.model.body_mocapid[target_body_id] < 0:
            raise ValueError("MJCF target body must exist and use mocap=true")
        self.target_mocap_id = int(self.model.body_mocapid[target_body_id])
        self.camera = MujocoEyeOnHandCamera(
            self.model,
            vision,
            seed=int(config["system"]["seed"]),
            detector_device=detector_device,
        )
        self.reset_ik = MujocoFeasibilityBackend(
            model_path, tcp_site_name, config["robot"]
        )
        self.transform_tcp_endpoint = transform_from_config(
            config["gripper"]["T_tcp_endpoint"]
        )

    def close(self):
        self.camera.close()

    def _trajectory_position(self, timestamp: float):
        trajectory = self.object_trajectory
        timestamp = float(np.clip(timestamp, trajectory.timestamps[0], trajectory.timestamps[-1]))
        return np.array(
            [
                np.interp(timestamp, trajectory.timestamps, trajectory.true_positions[:, axis])
                for axis in range(3)
            ],
            dtype=float,
        )

    def _set_target_position(self, position):
        self.data.mocap_pos[self.target_mocap_id] = np.asarray(position, dtype=float)

    def _privileged_initial_target(self, trajectory):
        track_id = int(trajectory.track_id)
        if track_id in self._privileged_initial_cache:
            return self._privileged_initial_cache[track_id]
        window_number = int(
            self.training_reset["privileged_reset_window_number"]
        )
        start_index = window_number - 1
        history = truth_history_matrix(
            trajectory.timestamps,
            trajectory.true_positions,
            start_index,
        )
        target = self.frozen_cage_network.predict(history)
        nominal = np.asarray(self.robot["nominal_joint_state_rad"], dtype=float)
        result = self.reset_ik.solve(pose6_to_transform(target), nominal)
        value = (
            target.copy(), result.joints.copy()
        ) if (
            result.reachable
            and result.collision_free
            and result.joints is not None
        ) else None
        self._privileged_initial_cache[track_id] = value
        return value

    def _find_track_and_initial_joints(self):
        for _ in range(len(self.trajectories)):
            self.trajectory_index = (self.trajectory_index + 1) % len(self.trajectories)
            trajectory = self.trajectories[self.trajectory_index]
            first_policy_window = int(
                self.training_reset["first_policy_window_number"]
            )
            if len(trajectory.timestamps) < first_policy_window + 4:
                continue
            initial = self._privileged_initial_target(trajectory)
            if initial is not None:
                target, joints = initial
                return trajectory, joints, target
        raise RuntimeError(
            "No trajectory has a collision-free privileged third-window Cage IK pose"
        )

    def _new_inference(self):
        return HistoricalCageInference(
            self.frozen_cage_network,
            self.config["trajectory"],
            self.config["kalman"],
            self.config["timing"],
            self.config["history_quality"],
        )

    def _clear_visual_history(self):
        self.inference = self._new_inference()
        self.consecutive_missed_detections = 0
        self.recent_detected_positions.clear()
        self.missing_kalman_positions = []
        self.last_valid_detection_time = None
        self.last_vision_detection = None
        self.cage_action_ready = False
        self.cage_hold_reason = "insufficient_five_frame_history"

    def _warm_visual_history(self, start_index, end_index):
        prediction = None
        consumed = 0
        for sample_index in range(int(start_index), int(end_index) + 1):
            timestamp = float(self.object_trajectory.timestamps[sample_index])
            self.object_time = timestamp
            self._set_target_position(
                self.object_trajectory.true_positions[sample_index]
            )
            self.mujoco.mj_forward(self.model, self.data)
            prediction = self._capture_and_update(timestamp, sample_index)
            consumed += 1
        return prediction, consumed

    def _warm_until_first_policy_state(self):
        """Render W4 first, then wait causally if that window is invalid.

        P3--P7 remains the nominal fourth completed five-frame window.  A
        noisy rendered history can still fail the same curvature/turn gate
        used online.  In that case PPO must not act; rendering advances one
        observation at a time until the first valid window, without reading a
        future frame into the current Cage input.
        """
        first_policy_window = int(
            self.training_reset["first_policy_window_number"]
        )
        nominal_index = first_policy_window + 3
        prediction, consumed = self._warm_visual_history(0, nominal_index)
        if prediction is not None and self.action_ready:
            return prediction, consumed, nominal_index

        maximum_extra = int(
            self.training_reset.get("initial_policy_valid_search_frames", 12)
        )
        last_index = min(
            len(self.object_trajectory.timestamps) - 1,
            nominal_index + maximum_extra,
        )
        for sample_index in range(nominal_index + 1, last_index + 1):
            prediction, additional = self._warm_visual_history(
                sample_index, sample_index
            )
            consumed += additional
            if prediction is not None and self.action_ready:
                return prediction, consumed, sample_index
        return None, consumed, last_index

    def _capture_and_update(self, timestamp: float, source_index: int):
        detection = self.camera.capture(self.data, timestamp, source_index)
        self.last_vision_detection = detection
        prediction = self.inference.update_then_predict(detection.observation)
        self.cage_action_ready = False
        self.cage_hold_reason = self.inference.last_quality.reason
        if detection.detected:
            self.last_valid_detection_time = float(timestamp)
            self.consecutive_missed_detections = 0
            self.missing_kalman_positions = []
            self.recent_detected_positions.append(
                detection.observation.position.copy()
            )
        else:
            self.missed_detection_count += 1
            self.consecutive_missed_detections += 1
            if self.inference.history:
                self.missing_kalman_positions.append(
                    self.inference.history[-1].position.copy()
                )
                self.missing_kalman_positions = self.missing_kalman_positions[-3:]
        if prediction is not None:
            # A valid visual history always publishes its new Cage target.
            # Physical tracking speed is enforced by PPO's per-event joint
            # increment and UR5e velocity limits, not by dropping valid targets.
            self.current_target = prediction.copy()
            self.last_cage_update_time = float(timestamp)
            self.cage_action_ready = True
            self.cage_hold_reason = ""
        return prediction

    @property
    def action_ready(self):
        fresh = (
            self.last_valid_detection_time is not None
            and self.object_time - self.last_valid_detection_time
            <= float(self.timing["max_observation_age_s"])
        )
        return bool(
            self.cage_action_ready
            and fresh
            and self.consecutive_missed_detections
            < int(self.training_reset["visual_escape_consecutive_misses"])
        )

    def reset(self, seed=None, options=None):
        if gym is not None:
            gym.Env.reset(self, seed=seed)
        elif seed is not None:
            np.random.seed(seed)
        self.mujoco.mj_resetData(self.model, self.data)
        trajectory, initial_joints, privileged_target = (
            self._find_track_and_initial_joints()
        )
        self.object_trajectory = trajectory
        self.data.qpos[self.qpos_addresses] = initial_joints
        self.data.qvel[self.dof_addresses] = 0.0
        self.data.ctrl[self.actuator_ids] = initial_joints
        self.mujoco.mj_forward(self.model, self.data)
        self.inference = self._new_inference()
        self.missed_detection_count = 0
        self.rejected_cage_jump_count = 0
        self.privileged_recovery_count = 0
        self.last_cage_update_time = None
        self.consecutive_missed_detections = 0
        self.recent_detected_positions.clear()
        self.missing_kalman_positions = []
        self.last_valid_detection_time = None
        self.last_vision_detection = None
        self.cage_action_ready = False
        self.cage_hold_reason = "insufficient_five_frame_history"
        first_policy_window = int(
            self.training_reset["first_policy_window_number"]
        )
        prediction, consumed, first_policy_index = (
            self._warm_until_first_policy_state()
        )
        if prediction is None or not self.action_ready:
            raise RuntimeError(
                "No valid visual PPO start state was found from the fourth "
                "completed five-frame window: {}".format(self.cage_hold_reason)
            )
        self.object_start_time = float(trajectory.timestamps[0])
        self.elapsed = self.object_time - self.object_start_time
        self.sequence = _StaticTargetSequence()
        self.sequence.duration = float(trajectory.timestamps[-1] - self.object_start_time)
        return self._observation(), {
            "trajectory_id": int(trajectory.track_id),
            "warmup_frames": int(consumed),
            "privileged_reset_window_number": int(
                self.training_reset["privileged_reset_window_number"]
            ),
            "first_policy_window_number": first_policy_window,
            "first_policy_sample_index": int(first_policy_index),
            "first_policy_delayed_frames": int(
                first_policy_index - (first_policy_window + 3)
            ),
            "privileged_reset_cage_pose6": privileged_target.copy(),
            "vision_detected": bool(self.last_vision_detection.detected),
            "vision_mask_pixels": int(self.last_vision_detection.mask_pixels),
        }

    def _privileged_recover_from_escape(self):
        if (
            len(self.recent_detected_positions) != 2
            or len(self.missing_kalman_positions) != 3
        ):
            return False, {"reason": "recovery_history_not_two_detected_plus_three_predicted"}
        try:
            projection = project_predictions_to_known_trajectory(
                self.missing_kalman_positions,
                self.object_trajectory.true_positions,
                self.recent_detected_positions[-1],
                int(self.training_reset["recovery_match_lookahead_samples"]),
                float(self.training_reset["recovery_match_max_distance_m"]),
            )
            matched = self.object_trajectory.true_positions[projection.indices]
            recovery_input = recovery_history_matrix(
                np.asarray(self.recent_detected_positions),
                matched,
                self.detection_period,
            )
            recovery_target = self.frozen_cage_network.predict(recovery_input)
            current_joints = self.data.qpos[self.qpos_addresses].copy()
            result = self.reset_ik.solve(
                pose6_to_transform(recovery_target), current_joints
            )
            if not (
                result.reachable
                and result.collision_free
                and result.joints is not None
            ):
                return False, {"reason": "recovery_cage_has_no_collision_free_ik"}
            self.data.qpos[self.qpos_addresses] = result.joints
            self.data.qvel[self.dof_addresses] = 0.0
            self.data.ctrl[self.actuator_ids] = result.joints
            self.mujoco.mj_forward(self.model, self.data)
            anchor_index = int(projection.indices[-1])
            warmup_start = max(0, anchor_index - 7)
            self._clear_visual_history()
            visual_target, consumed = self._warm_visual_history(
                warmup_start, anchor_index
            )
            resume_index = anchor_index
            if visual_target is None or not self.action_ready:
                maximum_extra = int(
                    self.training_reset.get(
                        "initial_policy_valid_search_frames", 12
                    )
                )
                last_index = min(
                    len(self.object_trajectory.timestamps) - 1,
                    anchor_index + maximum_extra,
                )
                for sample_index in range(anchor_index + 1, last_index + 1):
                    visual_target, additional = self._warm_visual_history(
                        sample_index, sample_index
                    )
                    consumed += additional
                    resume_index = sample_index
                    if visual_target is not None and self.action_ready:
                        break
            if visual_target is None or not self.action_ready:
                return False, {
                    "reason": "recovery_visual_warmup_failed:{}".format(
                        self.cage_hold_reason
                    )
                }

            # The known track is allowed to choose a training-only breakpoint,
            # but the first state after that boundary must again be local fine
            # tracking.  Synchronize the robot to the Cage pose predicted from
            # the newly rendered visual history, outside every PPO transition.
            synchronized = self.reset_ik.solve(
                pose6_to_transform(visual_target), result.joints
            )
            if not (
                synchronized.reachable
                and synchronized.collision_free
                and synchronized.joints is not None
            ):
                return False, {
                    "reason": "recovery_visual_cage_has_no_collision_free_ik"
                }
            self.data.qpos[self.qpos_addresses] = synchronized.joints
            self.data.qvel[self.dof_addresses] = 0.0
            self.data.ctrl[self.actuator_ids] = synchronized.joints
            self.mujoco.mj_forward(self.model, self.data)
            self.object_time = float(
                self.object_trajectory.timestamps[resume_index]
            )
            self.elapsed = self.object_time - self.object_start_time
            self.privileged_recovery_count += 1
            return True, {
                "reason": "",
                "matched_indices": projection.indices.copy(),
                "match_distances_m": projection.distances_m.copy(),
                "anchor_index": anchor_index,
                "resume_index": int(resume_index),
                "visual_warmup_frames": int(consumed),
                "recovery_cage_pose6": recovery_target.copy(),
                "resume_visual_cage_pose6": visual_target.copy(),
            }
        except (ValueError, IndexError, np.linalg.LinAlgError) as error:
            return False, {"reason": str(error)}

    def step(self, normalized_action):
        # Keep the exact Cage pose that produced s_t.  The RGB-D frame at the
        # end of this interval is allowed to produce C_{t+1}, but C_{t+1}
        # cannot retroactively score the action selected for C_t.
        reward_target = self.current_target.copy()
        action_state_relative_error = self._observation()[:6].copy()
        action = np.clip(np.asarray(normalized_action, dtype=float), -1.0, 1.0)
        history_invalid_before_action = not self.action_ready
        stale_before_action = not self.action_ready
        if history_invalid_before_action:
            action = np.zeros(6, dtype=float)
        executed_action = action.copy()
        joints = self.data.qpos[self.qpos_addresses].copy()
        velocity_bound = np.asarray(self.robot["joint_velocity_limit_radps"]) * self.decision_dt
        safe_delta = np.minimum(self.delta_max, velocity_bound)
        command = np.clip(joints + safe_delta * action, self.lower, self.upper)
        self.data.ctrl[self.actuator_ids] = command
        contact_collision = False
        for substep in range(self.low_level_steps):
            substep_elapsed = self.elapsed + (substep + 1) * self.model.opt.timestep
            substep_object_time = self.object_start_time + substep_elapsed
            self._set_target_position(self._trajectory_position(substep_object_time))
            self._apply_robot_bias_compensation()
            self.mujoco.mj_step(self.model, self.data)
            contact_collision = contact_collision or bool(self.data.ncon > 0)
        self.elapsed += self.decision_dt
        self.object_time = self.object_start_time + self.elapsed
        self._set_target_position(self._trajectory_position(self.object_time))
        self.mujoco.mj_forward(self.model, self.data)
        reward, reward_info = self._reward(
            command - joints,
            reward_target,
            action_state_relative_error,
        )
        if contact_collision:
            reward -= float(self.ppo["collision_penalty"])
        # The new RGB-D frame closes this decision interval.  The returned
        # 18-D observation is consumed immediately for exactly one next action.
        self._capture_and_update(self.object_time, -1)
        vision_updated = True
        escape_threshold = int(
            self.training_reset["visual_escape_consecutive_misses"]
        )
        visual_escape = self.consecutive_missed_detections >= escape_threshold
        escape_detection = self.last_vision_detection
        true_visual_escape = bool(
            visual_escape
            and escape_detection is not None
            and not escape_detection.target_visible_truth
        )
        detector_only_escape = bool(visual_escape and not true_visual_escape)
        if true_visual_escape:
            reward -= float(self.ppo.get("visual_escape_penalty", 0.0))
        recovery_succeeded = False
        recovery_info = {"reason": ""}
        rollout_boundary = False
        if visual_escape:
            recovery_succeeded, recovery_info = (
                self._privileged_recover_from_escape()
            )
            rollout_boundary = True
        observation = self._observation()
        stale = (
            self.last_valid_detection_time is None
            or self.object_time - self.last_valid_detection_time
            > float(self.timing["max_observation_age_s"])
        )
        reward_info.update(
            {
                "contact_collision": contact_collision,
                "vision_updated": vision_updated,
                "vision_detected": bool(
                    self.last_vision_detection is not None
                    and self.last_vision_detection.detected
                ),
                "vision_mask_pixels": int(
                    0 if self.last_vision_detection is None else self.last_vision_detection.mask_pixels
                ),
                "vision_stale": stale,
                "action_suppressed_for_stale_vision": stale_before_action,
                "action_suppressed_for_history_quality": history_invalid_before_action,
                "executed_normalized_action": executed_action.copy(),
                "visual_escape": bool(visual_escape),
                "true_visual_escape": bool(true_visual_escape),
                "detector_only_escape": bool(detector_only_escape),
                "frame_dropout_injected": bool(
                    escape_detection is not None
                    and escape_detection.dropout_injected
                ),
                "target_visible_truth": bool(
                    escape_detection is not None
                    and escape_detection.target_visible_truth
                ),
                "privileged_recovery_succeeded": bool(recovery_succeeded),
                "privileged_recovery_reason": recovery_info.get("reason", ""),
                "privileged_recovery_count": int(self.privileged_recovery_count),
                "rollout_boundary": bool(rollout_boundary),
                "history_quality_valid": bool(self.inference.last_quality.valid),
                "history_quality_reason": self.cage_hold_reason,
                "history_maximum_interval_s": self.inference.last_quality.maximum_interval_s,
                "history_curvature_m_inv": self.inference.last_quality.curvature_m_inv,
                "history_five_frame_turn_deg": self.inference.last_quality.five_frame_turn_deg,
                "missed_detection_count": self.missed_detection_count,
                "rejected_cage_jump_count": self.rejected_cage_jump_count,
                "reward_cage_pose6": reward_target.copy(),
                "next_cage_pose6": self.current_target.copy(),
                "action_state_relative_error6": action_state_relative_error,
                "next_state_relative_error6": observation[:6].copy(),
                "reward_target_contract": "same_cage_pose_as_action_state",
            }
        )
        terminated = bool(
            (self.ppo.get("terminate_on_contact", True) and contact_collision)
            or (visual_escape and not recovery_succeeded)
            or not np.all(np.isfinite(observation))
        )
        truncated = self.object_time >= self.object_trajectory.timestamps[-1]
        return observation, reward, terminated, truncated, reward_info
