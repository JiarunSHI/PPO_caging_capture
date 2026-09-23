import json
from collections import deque
from pathlib import Path

import numpy as np

from cage_system.audit_io import open_audit_text
from cage_system.cage.candidates import transform_from_config
from cage_system.geometry import (
    make_transform,
    normalize,
    orthonormal_frame,
    plane_basis,
    pose6_to_transform,
)
from cage_system.io import validate_trajectory_detection_period
from cage_system.learning.dataset import CageDatasetArrays, make_cage_input
from cage_system.learning.history_quality import evaluate_history_quality
from cage_system.perception.eye_on_hand import MujocoEyeOnHandCamera
from cage_system.sim.mujoco_feasibility import MujocoFeasibilityBackend
from cage_system.trajectory.online import OnlineTrajectoryEstimator


class VisionCageDatasetBuilder:
    """Replace synthetic history inputs with rendered causal RGB-D observations.

    Labels remain the EST-selected current TCP poses.  During rendering, the
    robot is teacher-forced to the most recent *strictly previous* feasible
    label, so the camera pose never uses the label currently being predicted.
    """

    def __init__(self, config, model_path: str, detector_device="auto"):
        import mujoco

        self.mujoco = mujoco
        self.config = config
        self.nominal_detection_period_s = 1.0 / float(
            config["timing"]["detection_rate_hz"]
        )
        self.model_path = str(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.camera = MujocoEyeOnHandCamera(
            self.model,
            config["vision"],
            seed=int(config["system"]["seed"]),
            detector_device=detector_device,
        )
        self.ik = MujocoFeasibilityBackend(
            self.model_path,
            config["labeling"]["mujoco_tcp_site"],
            config["robot"],
        )
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in config["robot"]["joint_names"]
            ]
        )
        if np.any(self.joint_ids < 0):
            raise ValueError("Vision MJCF is missing configured UR5e joints")
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        self.actuator_ids = np.asarray(
            [np.flatnonzero(actuator_joint_ids == joint_id)[0] for joint_id in self.joint_ids]
        )
        target_body = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            config["vision"]["target_mocap_body"],
        )
        self.target_mocap_id = int(self.model.body_mocapid[target_body])
        self.transform_tcp_endpoint = transform_from_config(
            config["gripper"]["T_tcp_endpoint"]
        )
        self.last_observation_pose_score = None

    def close(self):
        self.camera.close()

    @staticmethod
    def _key(timestamp):
        return round(float(timestamp), 9)

    @staticmethod
    def _true_position(trajectory, timestamp):
        return np.array(
            [
                np.interp(timestamp, trajectory.timestamps, trajectory.true_positions[:, axis])
                for axis in range(3)
            ],
            dtype=float,
        )

    def _initial_joints(self, trajectory):
        return self._observation_joints(
            trajectory,
            0,
            np.asarray(self.config["robot"]["nominal_joint_state_rad"], dtype=float),
        )

    def _observation_joints(self, trajectory, sample_index, seed_joints, label=None):
        """Choose one reachable fixed pose with the best trajectory coverage."""

        self.last_observation_pose_score = None
        nominal = np.asarray(self.config["robot"]["nominal_joint_state_rad"], dtype=float)
        seed_joints = np.asarray(seed_joints, dtype=float)
        joint_candidates = []
        if label is not None:
            result = self.ik.solve(pose6_to_transform(label), seed_joints)
            if result.reachable and result.collision_free and result.joints is not None:
                joint_candidates.append(result.joints.copy())

        sample_index = int(np.clip(sample_index, 0, len(trajectory.true_positions) - 1))
        remaining_points = np.asarray(
            trajectory.true_positions[sample_index:], dtype=float
        )
        focus_position = np.mean(remaining_points, axis=0)
        before = max(0, sample_index - 2)
        after = min(len(trajectory.true_positions) - 1, sample_index + 2)
        tangent = normalize(
            trajectory.true_positions[after] - trajectory.true_positions[before]
        )
        u_axis, v_axis = plane_basis(trajectory.plane_normal, tangent)
        radius_candidates = self.config["vision"].get(
            "observation_pose_radius_candidates_m",
            [float(self.config["candidate"]["radius_min_m"])],
        )
        for seed in (seed_joints, nominal):
            for radius_value in radius_candidates:
                radius = float(radius_value)
                for angle_deg in (0, -20, 20, -40, 40, -60, 60, -70, 70):
                    angle = np.deg2rad(float(angle_deg))
                    direction = normalize(
                        np.cos(angle) * u_axis + np.sin(angle) * v_axis
                    )
                    endpoint = focus_position - radius * direction
                    rotation = orthonormal_frame(
                        direction, trajectory.plane_normal
                    )
                    transform_base_endpoint = make_transform(endpoint, rotation)
                    transform_base_tcp = transform_base_endpoint @ np.linalg.inv(
                        self.transform_tcp_endpoint
                    )
                    result = self.ik.solve(transform_base_tcp, seed)
                    if (
                        result.reachable
                        and result.collision_free
                        and result.joints is not None
                    ):
                        joint_candidates.append(result.joints.copy())
        if not joint_candidates:
            return None

        original_qpos = self.data.qpos.copy()
        original_ctrl = self.data.ctrl.copy()
        best_joints = None
        best_score = None
        try:
            for joints in joint_candidates:
                self.data.qpos[self.qpos_addresses] = joints
                self.data.ctrl[self.actuator_ids] = joints
                self.mujoco.mj_forward(self.model, self.data)
                score = self._trajectory_view_score(
                    trajectory, sample_index
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_joints = joints.copy()
        finally:
            self.data.qpos[:] = original_qpos
            self.data.ctrl[:] = original_ctrl
            self.mujoco.mj_forward(self.model, self.data)
        self.last_observation_pose_score = best_score
        if (
            bool(self.config["vision"].get("require_full_trajectory_visibility", False))
            and (best_score is None or best_score[1] < 1.0 - 1e-12)
        ):
            return None
        return best_joints

    def _trajectory_view_score(self, trajectory, sample_index, lookahead_frames=None):
        """Rank a camera pose by current and near-future truth-point margins."""

        configured_lookahead = (
            int(self.config["vision"].get("observation_pose_lookahead_frames", 0))
            if lookahead_frames is None
            else int(lookahead_frames)
        )
        start = int(np.clip(sample_index, 0, len(trajectory.true_positions) - 1))
        stop = (
            len(trajectory.true_positions)
            if configured_lookahead == 0
            else start + configured_lookahead
        )
        points = np.asarray(
            trajectory.true_positions[start:stop], dtype=float
        )
        camera_position = self.data.cam_xpos[self.camera.camera_id].copy()
        camera_rotation = self.data.cam_xmat[self.camera.camera_id].reshape(3, 3).copy()
        points_camera = (points - camera_position) @ camera_rotation
        depth = -points_camera[:, 2]
        safe_depth = np.maximum(depth, 1e-9)
        columns = self.camera.intrinsics.cx + self.camera.intrinsics.fx * (
            points_camera[:, 0] / safe_depth
        )
        rows = self.camera.intrinsics.cy - self.camera.intrinsics.fy * (
            points_camera[:, 1] / safe_depth
        )
        target_pixel_radius = self.camera.intrinsics.fx * float(
            self.config["vision"]["target_radius_m"]
        ) / safe_depth
        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= float(self.config["vision"]["maximum_depth_m"]))
            & (columns - target_pixel_radius >= 0.0)
            & (columns + target_pixel_radius < float(self.camera.width))
            & (rows - target_pixel_radius >= 0.0)
            & (rows + target_pixel_radius < float(self.camera.height))
        )
        edge_margin = np.minimum.reduce(
            (
                (columns - target_pixel_radius)
                / max(float(self.camera.width - 1), 1.0),
                (float(self.camera.width - 1) - columns - target_pixel_radius)
                / max(float(self.camera.width - 1), 1.0),
                (rows - target_pixel_radius)
                / max(float(self.camera.height - 1), 1.0),
                (float(self.camera.height - 1) - rows - target_pixel_radius)
                / max(float(self.camera.height - 1), 1.0),
            )
        )
        visible_margins = edge_margin[valid]
        minimum_margin = (
            float(np.min(visible_margins)) if len(visible_margins) else -1.0
        )
        mean_margin = (
            float(np.mean(visible_margins)) if len(visible_margins) else -1.0
        )
        return (
            int(bool(valid[0])),
            float(np.mean(valid)),
            minimum_margin,
            mean_margin,
        )

    def _new_online_estimator(self):
        return OnlineTrajectoryEstimator(
            self.config["trajectory"], self.config["kalman"], self.config["timing"]
        )

    def build(self, trajectories, geometric_labels: CageDatasetArrays, audit_path=None):
        validate_trajectory_detection_period(
            trajectories, self.nominal_detection_period_s
        )
        geometric_labels.validate().require_detection_period(
            self.nominal_detection_period_s
        )
        output_inputs, output_labels, output_tracks, output_times = [], [], [], []
        audit_handle = open_audit_text(audit_path, "w") if audit_path else None
        source_keys = set()
        source_track_ids = set()
        for trajectory in trajectories:
            track_id = int(trajectory.track_id)
            if track_id in source_track_ids:
                raise ValueError("Vision trajectories contain a duplicate track_id")
            source_track_ids.add(track_id)
            if len(trajectory.timestamps) != len(trajectory.true_positions):
                raise ValueError("Trajectory timestamps and true positions are misaligned")
            track_keys = [
                (track_id, self._key(timestamp))
                for timestamp in trajectory.timestamps
            ]
            if len(set(track_keys)) != len(track_keys):
                raise ValueError("A vision trajectory contains duplicate timestamps")
            source_keys.update(track_keys)
        label_lookup = {}
        for index in range(len(geometric_labels.inputs)):
            key = (
                int(geometric_labels.track_ids[index]),
                self._key(geometric_labels.label_timestamps[index]),
            )
            if key in label_lookup:
                raise ValueError("Geometric labels contain a duplicate track/timestamp key")
            if key not in source_keys:
                raise ValueError(
                    "A geometric label does not match its source trajectory timestamp"
                )
            label_lookup[key] = geometric_labels.labels[index]
        try:
            for trajectory in trajectories:
                self.mujoco.mj_resetData(self.model, self.data)
                current_joints = self._initial_joints(trajectory)
                if current_joints is None:
                    reason = (
                        "no_fixed_camera_pose_covers_full_trajectory"
                        if self.last_observation_pose_score is not None
                        else "no_collision_free_initial_camera_ik"
                    )
                    self._write(
                        audit_handle,
                        {
                            "track_id": int(trajectory.track_id),
                            "valid": False,
                            "reason": reason,
                            "best_observation_pose_score": (
                                self.last_observation_pose_score
                            ),
                        },
                    )
                    continue
                self.data.qpos[self.qpos_addresses] = current_joints
                self.data.ctrl[self.actuator_ids] = current_joints
                online = self._new_online_estimator()
                history = deque(maxlen=5)
                history_source_indices = deque(maxlen=5)
                rebuilding_history = True
                consecutive_missed_frames = 0
                fixed_view_failed = False
                track_inputs, track_labels, track_times = [], [], []
                view_loss_threshold = int(
                    self.config["vision"].get("view_loss_consecutive_misses", 3)
                )
                # Replay the clean trajectory clock. Synthetic geometric
                # observation noise/dropout is deliberately not reused here;
                # all failures must come from the rendered RGB-D frontend or
                # its causal Kalman innovation gate.
                for observation_index, source_timestamp in enumerate(trajectory.timestamps):
                    timestamp = float(source_timestamp)
                    # The rendered target and the geometric-label lookup share
                    # the exact same archive index and timestamp.
                    true_position = np.asarray(
                        trajectory.true_positions[observation_index], dtype=float
                    ).copy()
                    self.data.mocap_pos[self.target_mocap_id] = true_position
                    self.mujoco.mj_forward(self.model, self.data)
                    detection = self.camera.capture(self.data, timestamp, observation_index)
                    history_before_current = list(history)
                    history_indices_before_current = list(history_source_indices)
                    current_frame = None
                    accepted_measurements_before_current = (
                        online.accepted_measurement_count
                    )
                    innovation_gate_bypassed_for_warmup = bool(
                        detection.observation.valid
                        and accepted_measurements_before_current
                        < online.innovation_gate_warmup_measurements
                    )
                    if detection.observation.valid or online.filter.timestamp is not None:
                        current_frame = online.update(detection.observation)
                    accepted_measurements_after_update = (
                        online.accepted_measurement_count
                    )
                    innovation_mahalanobis = (
                        online.filter.last_innovation_mahalanobis
                    )
                    innovation_gate_bypassed_actual = bool(
                        online.filter.last_innovation_gate_bypassed
                    )
                    label = label_lookup.get((int(trajectory.track_id), self._key(timestamp)))
                    history_quality = evaluate_history_quality(
                        history_before_current,
                        timestamp,
                        self.config["timing"],
                        self.config["history_quality"],
                        enforce_maximum_interval=False,
                    )
                    measurement_accepted = bool(
                        current_frame is not None and current_frame.measurement_accepted
                    )
                    missed_before_current = consecutive_missed_frames
                    if detection.observation.valid:
                        consecutive_missed_frames = 0
                    else:
                        consecutive_missed_frames += 1
                    view_loss_declared = bool(
                        not detection.observation.valid
                        and consecutive_missed_frames >= view_loss_threshold
                    )
                    history_indices_match = bool(
                        history_indices_before_current
                        == list(range(observation_index - 5, observation_index))
                    )
                    label_written = False
                    if (
                        label is not None
                        and history_quality.valid
                        and measurement_accepted
                        and history_indices_match
                    ):
                        track_inputs.append(make_cage_input(history_before_current))
                        track_labels.append(np.asarray(label, dtype=float).copy())
                        track_times.append(timestamp)
                        label_written = True
                    center_error = (
                        float(np.linalg.norm(detection.observation.position - true_position))
                        if detection.detected
                        else None
                    )
                    ik_applied_after_capture = False
                    privileged_recovery_applied = False
                    history_reset = False
                    robot_pose_held_after_failure = False
                    camera_action = "hold"
                    failure_reason = ""
                    if not measurement_accepted:
                        failure_reason = (
                            "kalman_innovation_rejected"
                            if detection.observation.valid
                            else "visual_detection_missed"
                        )
                        history.clear()
                        history_source_indices.clear()
                        history_reset = True
                        rebuilding_history = True
                        # The Kalman prediction/covariance is still advanced at
                        # this timestamp, but predict-only/rejected states never
                        # enter the network history or its curvature segment.
                        online.clear_causal_curve_history()
                        if view_loss_declared:
                            # The camera is fixed for the entire trajectory.
                            # Truth is used only to distinguish an actual FOV
                            # failure from detector/dropout failure; it never
                            # enters a Cage input feature.
                            if not detection.target_visible_truth:
                                fixed_view_failed = True
                            else:
                                consecutive_missed_frames = 0
                                online = self._new_online_estimator()
                            robot_pose_held_after_failure = True
                        else:
                            # A single/two-frame miss or innovation rejection
                            # must never cause a robot/camera pose change.
                            robot_pose_held_after_failure = True
                    # During ordinary point-by-point rendering the robot and
                    # eye-on-hand camera stay fixed. Geometric Cage labels are
                    # supervision targets only; they never drive this camera.
                    if measurement_accepted:
                        history.append(current_frame)
                        history_source_indices.append(observation_index)
                        if len(history) == history.maxlen:
                            rebuilding_history = False
                    self._write(
                        audit_handle,
                        {
                            "track_id": int(trajectory.track_id),
                            "timestamp": timestamp,
                            "detected": bool(detection.detected),
                            "dropout_injected": bool(detection.dropout_injected),
                            "target_visible_truth": bool(detection.target_visible_truth),
                            "mask_pixels": int(detection.mask_pixels),
                            "confidence": float(detection.confidence),
                            "center_error_m": center_error,
                            "history_count_before_current": len(history_before_current),
                            "history_source_indices_before_current": (
                                history_indices_before_current
                            ),
                            "history_source_indices_match": history_indices_match,
                            "history_quality_valid": history_quality.valid,
                            "history_quality_reason": history_quality.reason,
                            "history_missed_detections": history_quality.missed_detections,
                            "history_maximum_interval_s": history_quality.maximum_interval_s,
                            "history_curvature_m_inv": history_quality.curvature_m_inv,
                            "history_five_frame_turn_deg": history_quality.five_frame_turn_deg,
                            "history_five_frame_plane_ratio": (
                                history_quality.five_frame_plane_ratio
                            ),
                            "current_measurement_accepted": measurement_accepted,
                            "kalman_accepted_measurements_before_current": (
                                accepted_measurements_before_current
                            ),
                            "kalman_accepted_measurements_after_update": (
                                accepted_measurements_after_update
                            ),
                            "innovation_gate_warmup_measurements": (
                                online.innovation_gate_warmup_measurements
                            ),
                            "innovation_gate_bypassed_for_warmup": (
                                innovation_gate_bypassed_for_warmup
                            ),
                            "innovation_gate_bypassed_actual": (
                                innovation_gate_bypassed_actual
                            ),
                            "innovation_mahalanobis": innovation_mahalanobis,
                            "current_failure_reason": failure_reason,
                            "consecutive_missed_frames_before_current": missed_before_current,
                            "consecutive_missed_frames_after_current": consecutive_missed_frames,
                            "view_loss_declared": view_loss_declared,
                            "view_loss_threshold_frames": view_loss_threshold,
                            "history_reset_after_current": history_reset,
                            "rebuilding_five_accepted_frames": rebuilding_history,
                            "geometric_label_available": label is not None,
                            "vision_label_written": label_written,
                            "previous_label_ik_applied_after_capture": ik_applied_after_capture,
                            "privileged_observation_recovery_applied": privileged_recovery_applied,
                            "robot_pose_held_after_short_failure": robot_pose_held_after_failure,
                            "camera_action_after_capture": camera_action,
                            "trajectory_sample_index": observation_index,
                            "trajectory_timestamp_matched": bool(
                                abs(timestamp - float(trajectory.timestamps[observation_index]))
                                <= 1e-12
                            ),
                        },
                    )
                    if fixed_view_failed:
                        self._write(
                            audit_handle,
                            {
                                "track_id": int(trajectory.track_id),
                                "timestamp": timestamp,
                                "trajectory_sample_index": observation_index,
                                "valid": False,
                                "reason": "fixed_camera_lost_true_target_visibility",
                                "discarded_track_visual_labels": len(track_inputs),
                            },
                        )
                        break
                if not fixed_view_failed:
                    output_inputs.extend(track_inputs)
                    output_labels.extend(track_labels)
                    output_tracks.extend(
                        [int(trajectory.track_id)] * len(track_inputs)
                    )
                    output_times.extend(track_times)
        finally:
            if audit_handle is not None:
                audit_handle.close()
        if not output_inputs:
            return CageDatasetArrays(
                np.empty((0, 5, 4)),
                np.empty((0, 6)),
                np.empty((0,), dtype=int),
                np.empty((0,)),
                self.nominal_detection_period_s,
                "mujoco_eye_on_hand_rgbd",
            )
        return CageDatasetArrays(
            np.asarray(output_inputs),
            np.asarray(output_labels),
            np.asarray(output_tracks),
            np.asarray(output_times),
            self.nominal_detection_period_s,
            "mujoco_eye_on_hand_rgbd",
        ).validate()

    @staticmethod
    def _write(handle, record):
        if handle is not None:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
