"""Dynamic eye-on-hand Cage reconstruction and JAiLeR tracking evaluation.

This module deliberately keeps the offline EST supervisor out of inference.
The current Cage target is reconstructed from causal RGB-D observations with
one fixed normalized location on the signed minor arc.
"""

import copy
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from cage_system.cage.candidates import (
    CandidateGenerator,
    SpeedRadiusMapper,
    transform_from_config,
)
from cage_system.cage.plane import LocalMotionPlaneEstimator
from cage_system.control.constrained_differential_ik import (
    TimeConstrainedDifferentialIK,
)
from cage_system.geometry import make_transform, normalize, orthonormal_frame, transform_to_pose6
from cage_system.jailer_visualization import _add_pose_axes, _torch_load
from cage_system.io import save_generated_trajectories
from cage_system.learning.jailer_policy import JailerPPO2Trainer
from cage_system.learning.ppo2 import build_ppo_state, relative_pose_error6
from cage_system.perception.eye_on_hand import MujocoEyeOnHandCamera
from cage_system.perception.vision_dataset import VisionCageDatasetBuilder
from cage_system.sim.ur5e_control import UR5eControlModel
from cage_system.system_visualization import _add_capsule
from cage_system.trajectory.generator import PlanarTrajectoryGenerator
from cage_system.trajectory.online import OnlineTrajectoryEstimator
from cage_system.trajectory.spline import SharedCubicSpline3D


def _stage_from_checkpoint(checkpoint, stage_name):
    definition = checkpoint.get("curriculum_definition")
    if not definition:
        raise ValueError("JAiLeR checkpoint has no curriculum definition")
    stages = {str(item["name"]): copy.deepcopy(item) for item in definition["stages"]}
    if stage_name is None:
        index = int(checkpoint.get("curriculum", {}).get("stage_index", len(stages) - 1))
        ordered = list(definition["stages"])
        index = int(np.clip(index, 0, len(ordered) - 1))
        stage = copy.deepcopy(ordered[index])
    else:
        if stage_name not in stages:
            raise ValueError("Checkpoint has no stage '{}'".format(stage_name))
        stage = stages[stage_name]
    if str(stage["task"]) != "pose":
        raise ValueError("Dynamic Cage inference requires a pose-policy stage")
    return stage


class CausalFixedAlphaCageReconstructor:
    """Reconstruct one Cage pose from causal Kalman and local-spline state."""

    def __init__(self, config, normalized_arc_position=0.4375):
        self.config = config
        self.alpha = float(normalized_arc_position)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Normalized Cage arc position must be in [0,1]")
        self.online = OnlineTrajectoryEstimator(
            config["trajectory"], config["kalman"], config["timing"]
        )
        self.plane = LocalMotionPlaneEstimator(config["plane"])
        self.radius = SpeedRadiusMapper(config["candidate"])
        self.candidates = CandidateGenerator(
            config["candidate"],
            transform_from_config(config["gripper"]["T_tcp_endpoint"]),
        )
        self.points = deque()

    def update(self, observation):
        if not observation.valid and self.online.filter.timestamp is None:
            return None, None
        frame = self.online.update(observation)
        self.points.append((float(frame.timestamp), frame.position.copy()))
        # Before five valid RGB-D measurements have jointly initialized X/P,
        # frames are visual bootstrap data only and cannot produce a Cage.
        if self.online.filter.timestamp is None:
            return frame, None
        window = float(self.config["trajectory"]["online_window_s"])
        while self.points and frame.timestamp - self.points[0][0] > window:
            self.points.popleft()
        if len(self.points) < 5:
            return frame, None
        times = np.asarray([item[0] for item in self.points], dtype=float)
        positions = np.asarray([item[1] for item in self.points], dtype=float)
        ages = float(frame.timestamp) - times
        weights = 1.0 + float(
            self.config["trajectory"]["recent_weight_gain"]
        ) * np.exp(-ages / max(0.35 * window, 1e-3))
        try:
            spline = SharedCubicSpline3D(
                self.config["trajectory"]["spline_control_spacing_s"],
                self.config["trajectory"]["spline_regularization"],
            ).fit(times, positions, weights)
        except (ValueError, np.linalg.LinAlgError):
            return frame, None

        current_time = float(frame.timestamp)
        center = self.online.filter.state[:3].copy()
        velocity = self.online.filter.state[3:6].copy()
        acceleration = self.online.filter.state[6:9].copy()
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-6:
            tangent = velocity / speed
        else:
            tangent = normalize(spline.evaluate(current_time, 1), frame.tangent)
        spline_velocity = spline.evaluate(current_time, 1)
        spline_acceleration = spline.evaluate(current_time, 2)
        spline_speed = float(np.linalg.norm(spline_velocity))
        curvature = (
            0.0
            if spline_speed < 1e-6
            else float(
                np.linalg.norm(np.cross(spline_velocity, spline_acceleration))
                / spline_speed ** 3
            )
        )
        plane_points = spline.sample(times[0], current_time, max(12, 3 * len(times)))
        plane = self.plane.estimate(
            plane_points, center, tangent, curvature, acceleration
        )
        radius, radius_audit = self.radius.map(speed, current_time)
        horizon = min(3.0, max(1.0, 1.15 * radius / max(speed, 0.04)))
        endpoint_offset = center - spline.evaluate(current_time)

        def spline_query(timestamp):
            return spline.evaluate(float(timestamp)) + endpoint_offset

        fallback = False
        try:
            arc = self.candidates.construct_arc(
                center,
                tangent,
                plane.normal,
                radius,
                current_time,
                current_time + horizon,
                spline_query,
            )
        except ValueError:
            # A local cubic can bend back before reaching the speed radius.
            # Fall back to the current causal tangent; this is a degenerate
            # minor arc, not a future ground-truth point or an EST call.
            fallback = True

            def tangent_query(timestamp):
                return center + max(speed, 0.04) * (float(timestamp) - current_time) * tangent

            arc = self.candidates.construct_arc(
                center,
                tangent,
                plane.normal,
                radius,
                current_time,
                current_time + 1.1 * radius / max(speed, 0.04),
                tangent_query,
            )
        angle = float(arc.angle_start + self.alpha * (arc.angle_end - arc.angle_start))
        pose = self.candidates.build_poses(arc, np.asarray([angle]))[0]
        return frame, {
            "pose6": transform_to_pose6(pose.tcp_transform_base),
            "center": center,
            "speed_mps": speed,
            "velocity_base_mps": velocity,
            "acceleration_base_mps2": acceleration,
            "curvature_m_inv": curvature,
            "plane_normal_base": plane.normal.copy(),
            "radius_m": float(radius),
            "arc_start_rad": float(arc.angle_start),
            "arc_end_rad": float(arc.angle_end),
            "arc_angle_rad": angle,
            "normalized_arc_position": self.alpha,
            "future_curve_fallback": fallback,
            "radius_audit": radius_audit,
        }


def _near_object_initial_joints(
    planner,
    trajectory,
    standoff_candidates,
    minimum_visibility,
    visibility_lookahead_frames,
    trend_offset_candidates_deg=(0.0, -10.0, 10.0, -20.0, 20.0),
):
    """Place the wrist ahead of the simulated object's initial motion trend.

    This is privileged simulation setup only.  The resulting tangent/normal are
    never passed to the online Kalman/spline reconstructor or either controller.
    """

    nominal = np.asarray(planner.config["robot"]["nominal_joint_state_rad"], dtype=float)
    planner.data.qpos[planner.qpos_addresses] = nominal
    planner.data.ctrl[planner.actuator_ids] = nominal
    planner.mujoco.mj_forward(planner.model, planner.data)
    tcp_site_id = planner.mujoco.mj_name2id(
        planner.model,
        planner.mujoco.mjtObj.mjOBJ_SITE,
        str(planner.config["labeling"]["mujoco_tcp_site"]),
    )
    if tcp_site_id < 0:
        raise ValueError("Configured TCP site is absent from the observation MJCF")

    initial_position = np.asarray(trajectory.true_positions[0], dtype=float)
    trend_index = min(max(int(visibility_lookahead_frames) - 1, 1), len(trajectory.true_positions) - 1)
    initial_tangent = normalize(
        trajectory.true_positions[trend_index] - initial_position,
        np.array([1.0, 0.0, 0.0]),
    )
    initial_normal = normalize(
        np.asarray(trajectory.plane_normal, dtype=float),
        np.array([0.0, 0.0, 1.0]),
    )
    reference_normal = normalize(
        np.asarray(
            planner.config["plane"].get("base_reference_direction", [0.0, 0.0, 1.0]),
            dtype=float,
        )
    )
    if np.dot(initial_normal, reference_normal) < 0.0:
        initial_normal *= -1.0
    lateral = normalize(
        np.cross(initial_normal, initial_tangent),
        np.array([0.0, 1.0, 0.0]),
    )
    directions = []
    for offset_deg in trend_offset_candidates_deg:
        offset = np.deg2rad(offset_deg)
        ahead_axis = normalize(
            np.cos(offset) * initial_tangent + np.sin(offset) * lateral,
            initial_tangent,
        )
        # TCP/Cage +x must point from the forward pose back to the object.
        directions.append(-ahead_axis)

    best_joints = None
    best_view_score = None
    best_rank = None
    for standoff in [float(value) for value in standoff_candidates]:
        standoff_best_joints = None
        standoff_best_score = None
        standoff_best_rank = None
        for direction in directions:
            for z_reference in (initial_normal, reference_normal):
                try:
                    z_axis = z_reference - np.dot(z_reference, direction) * direction
                    rotation = orthonormal_frame(direction, z_axis)
                except (ValueError, FloatingPointError):
                    continue
                endpoint = initial_position - standoff * direction
                transform_base_endpoint = make_transform(endpoint, rotation)
                transform_base_tcp = transform_base_endpoint @ np.linalg.inv(
                    planner.transform_tcp_endpoint
                )
                result = planner.ik.solve(transform_base_tcp, nominal)
                if not result.reachable or not result.collision_free or result.joints is None:
                    continue
                planner.data.qpos[planner.qpos_addresses] = result.joints
                planner.data.ctrl[planner.actuator_ids] = result.joints
                planner.mujoco.mj_forward(planner.model, planner.data)
                score = planner._trajectory_view_score(
                    trajectory, 0, lookahead_frames=visibility_lookahead_frames
                )
                rank = (
                    int(bool(score[0])),
                    float(score[1]),
                    float(score[2]),
                    float(score[3]),
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_view_score = score
                    best_joints = result.joints.copy()
                if standoff_best_rank is None or rank > standoff_best_rank:
                    standoff_best_rank = rank
                    standoff_best_score = score
                    standoff_best_joints = result.joints.copy()
        # Prefer the first (closest) standoff that sees the complete causal
        # startup window; do not search farther merely for a larger image margin.
        if standoff_best_joints is not None:
            assert standoff_best_score is not None
            if (
                bool(standoff_best_score[0])
                and float(standoff_best_score[1]) >= float(minimum_visibility)
            ):
                return standoff_best_joints, standoff_best_score, "accepted"

    if best_joints is None or best_view_score is None:
        return None, None, "initial_near_pose_ik_or_collision"
    if (
        not bool(best_view_score[0])
        or float(best_view_score[1]) < float(minimum_visibility)
    ):
        return None, best_view_score, "initial_causal_window_not_visible"
    return best_joints, best_view_score, "accepted"


def _open_curve_metrics(trajectory):
    """Return accumulated tangent turn and endpoint/path-length ratio."""

    segments = np.diff(np.asarray(trajectory.true_positions, dtype=float), axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-9
    directions = segments[valid] / lengths[valid, None]
    if len(directions) < 2:
        return float("inf"), 0.0
    dots = np.clip(np.sum(directions[1:] * directions[:-1], axis=1), -1.0, 1.0)
    total_turn_deg = float(np.degrees(np.sum(np.arccos(dots))))
    path_length = float(np.sum(lengths))
    endpoint_distance = float(
        np.linalg.norm(trajectory.true_positions[-1] - trajectory.true_positions[0])
    )
    return total_turn_deg, endpoint_distance / max(path_length, 1e-12)


def _trajectory_workspace_and_speed_metrics(trajectory):
    """Summarize the complete privileged evaluation path for setup filtering only."""

    positions = np.asarray(trajectory.true_positions, dtype=float)
    timestamps = np.asarray(trajectory.timestamps, dtype=float)
    base_distances = np.linalg.norm(positions, axis=1)
    base_xy_distances = np.linalg.norm(positions[:, :2], axis=1)
    delta_t = np.diff(timestamps)
    delta_position = np.diff(positions, axis=0)
    valid = delta_t > 1e-9
    speeds = np.linalg.norm(delta_position[valid], axis=1) / delta_t[valid]
    return {
        "minimum_base_distance_m": float(np.min(base_distances)),
        "maximum_base_distance_m": float(np.max(base_distances)),
        "minimum_base_xy_distance_m": float(np.min(base_xy_distances)),
        "maximum_base_xy_distance_m": float(np.max(base_xy_distances)),
        "minimum_height_m": float(np.min(positions[:, 2])),
        "maximum_height_m": float(np.max(positions[:, 2])),
        "minimum_speed_mps": float(np.min(speeds)) if len(speeds) else 0.0,
        "mean_speed_mps": float(np.mean(speeds)) if len(speeds) else 0.0,
        "maximum_speed_mps": float(np.max(speeds)) if len(speeds) else 0.0,
    }


def _load_policy(config, checkpoint_path, model_path, tcp_site_name, stage_name, device):
    checkpoint = _torch_load(checkpoint_path)
    stage = _stage_from_checkpoint(checkpoint, stage_name)
    settings = copy.deepcopy(config["jailer"])
    settings["curriculum"] = copy.deepcopy(checkpoint["curriculum_definition"])
    ur5e = UR5eControlModel(
        model_path,
        config["robot"],
        tcp_site_name=tcp_site_name,
        control_rate_hz=float(settings["control_rate_hz"]),
    )
    saved_contract = checkpoint.get("ur5e_contract")
    if saved_contract is not None and saved_contract != ur5e.contract_dict():
        raise ValueError("JAiLeR checkpoint does not match the requested UR5e/TCP MJCF")
    trainer = JailerPPO2Trainer(settings["ppo"], device=device)
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.state_normalizer.load_state_dict(checkpoint["state_normalizer"])
    trainer.set_curriculum_task("pose")
    trainer.model.eval()
    return checkpoint, stage, settings, ur5e, trainer


def _target_mocap_id(ur5e, vision_config):
    body_id = ur5e.mujoco.mj_name2id(
        ur5e.model,
        ur5e.mujoco.mjtObj.mjOBJ_BODY,
        str(vision_config["target_mocap_body"]),
    )
    if body_id < 0 or int(ur5e.model.body_mocapid[body_id]) < 0:
        raise ValueError("Configured target body is not a MuJoCo mocap body")
    return int(ur5e.model.body_mocapid[body_id])


def evaluate_dynamic_cage_tracking(
    config,
    checkpoint_path,
    model_path,
    tcp_site_name,
    episodes,
    seed,
    output_path,
    trajectory_output=None,
    stage_name=None,
    video_output=None,
    video_fps=25.0,
    show_window=False,
    realtime_display=False,
    show_gripper=True,
    device="auto",
):
    """Evaluate random, fully visible tracks and optionally render episode zero."""
    try:
        import cv2
        import mujoco
    except ImportError as error:
        raise RuntimeError("MuJoCo and OpenCV are required for dynamic evaluation") from error

    inference = config["dynamic_cage_inference"]
    if realtime_display and not show_window:
        raise ValueError("--realtime-display requires --show-window")
    checkpoint, stage, settings, ur5e, trainer = _load_policy(
        config, checkpoint_path, model_path, tcp_site_name, stage_name, device
    )
    camera = MujocoEyeOnHandCamera(
        ur5e.model,
        config["vision"],
        seed=int(seed),
        detector_device=device,
    )
    # Simulator truth is used only to place the target and reject an initial
    # setup that cannot see the five-frame warmup.  It is never passed to the
    # Kalman/spline Cage reconstruction, Jacobian controller, or PPO state.
    observation_planner = VisionCageDatasetBuilder(config, model_path, detector_device=device)
    generator = PlanarTrajectoryGenerator(config["trajectory"], seed=int(seed))
    target_mocap_id = _target_mocap_id(ur5e, config["vision"])
    control_dt = float(ur5e.control_dt)
    visual_dt = 1.0 / float(config["timing"]["detection_rate_hz"])
    control_steps_per_visual = int(round(visual_dt / control_dt))
    if not np.isclose(control_steps_per_visual * control_dt, visual_dt):
        raise ValueError("The 10 Hz visual period must contain an integer number of controls")
    action_limit = ur5e.velocity_limits * float(
        settings["action_joint_velocity_limit_fraction"]
    )
    margin = float(settings["joint_position_margin_rad"])
    safe_lower, safe_upper = ur5e.lower + margin, ur5e.upper - margin
    alpha = float(inference.get("normalized_arc_position", 0.4375))
    warmup_frames = int(inference.get("visual_hold_warmup_frames", 5))
    miss_limit = int(inference.get("consecutive_miss_failure", 3))
    switch_position = float(inference.get("ppo_switch_position_m", 0.015))
    switch_orientation = float(inference.get("ppo_switch_orientation_deg", 45.0))
    initial_standoffs = inference.get(
        "initial_standoff_candidates_m", [0.12, 0.14, 0.16, 0.18]
    )
    initial_trend_offsets = inference.get(
        "initial_trend_offset_candidates_deg", [0.0, -10.0, 10.0, -20.0, 20.0]
    )
    minimum_initial_visibility = float(
        inference.get("minimum_initial_trajectory_visibility_fraction", 1.0)
    )
    initial_visibility_lookahead = int(
        inference.get("initial_visibility_lookahead_frames", warmup_frames + 1)
    )
    maximum_total_turn_deg = float(inference.get("maximum_total_turn_deg", 150.0))
    minimum_endpoint_ratio = float(
        inference.get("minimum_endpoint_displacement_ratio", 0.45)
    )
    minimum_base_distance = float(inference.get("minimum_base_distance_m", 0.0))
    maximum_base_distance = float(inference.get("maximum_base_distance_m", float("inf")))
    minimum_base_xy_distance = float(
        inference.get("minimum_base_xy_distance_m", 0.0)
    )
    minimum_height, maximum_height = map(
        float, inference.get("base_height_bounds_m", [-float("inf"), float("inf")])
    )
    max_attempts = int(inference.get("maximum_trajectory_attempts_per_episode", 30))
    ik_config = copy.deepcopy(config["constrained_differential_ik"])
    ik_config["position_only"] = False
    summaries = []
    selected_trajectories = []
    next_track_id = 0
    writer = None
    overview = None
    render_options = mujoco.MjvOption()
    # MuJoCo hides geom groups 3--5 by default.  The physical Cage fingers
    # and the translucent internal region live in groups 3 and 4, so make
    # every model geometry visible in the live view and recorded video.
    render_options.geomgroup[:] = 1
    render_options.geomgroup[3] = int(bool(show_gripper))
    render_options.geomgroup[4] = int(bool(show_gripper))
    video_stride = None
    window_created = False

    def open_video():
        nonlocal writer, overview, video_stride
        if (video_output is None and not show_window) or overview is not None:
            return
        if video_output is not None:
            video_path = Path(video_output)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                float(video_fps), (960, 720)
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open the requested MP4")
        video_stride = int(round(float(settings["control_rate_hz"]) / float(video_fps)))
        if video_stride <= 0:
            raise ValueError("video-fps is too high")
        overview = mujoco.Renderer(ur5e.model, height=720, width=960, max_geom=10000)

    def render_frame(data, trajectory, cage_target, phase, elapsed_s, metrics):
        nonlocal window_created
        active_overview = overview
        if active_overview is None:
            return
        active_overview.update_scene(
            data, camera="overview", scene_option=render_options
        )
        for index in range(1, len(trajectory.true_positions)):
            _add_capsule(
                active_overview.scene,
                mujoco,
                trajectory.true_positions[index - 1],
                trajectory.true_positions[index],
                0.002,
                (0.15, 0.85, 0.95, 0.65),
            )
        if cage_target is not None:
            _add_pose_axes(active_overview.scene, mujoco, cage_target, length=0.06)
        rgb = active_overview.render().copy()
        active_overview.update_scene(
            data, camera="eye_on_hand", scene_option=render_options
        )
        eye = cv2.resize(active_overview.render().copy(), (288, 216), interpolation=cv2.INTER_AREA)
        rgb[480:696, 648:936] = eye
        active_overview.update_scene(
            data, camera="cage_closeup", scene_option=render_options
        )
        cage_closeup = cv2.resize(
            active_overview.render().copy(), (360, 270), interpolation=cv2.INTER_AREA
        )
        rgb[430:700, 20:380] = cage_closeup
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (20, 430), (379, 699), (0, 165, 255), 2)
        cv2.rectangle(frame, (648, 480), (935, 695), (255, 255, 255), 1)
        cv2.putText(
            frame, "physical gripper/Cage", (28, 420), cv2.FONT_HERSHEY_SIMPLEX,
            0.50, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "eye_on_hand", (654, 470), cv2.FONT_HERSHEY_SIMPLEX,
            0.50, (255, 255, 255), 1, cv2.LINE_AA
        )
        lines = [
            "dynamic Cage alpha={:.4f} | {}".format(alpha, phase),
            "t={:.2f}s  detection={}  misses={}".format(
                elapsed_s, metrics.get("detected"), metrics.get("consecutive_misses")
            ),
            "TCP-Cage: {:.1f} mm / {:.1f} deg".format(
                metrics.get("position_error_mm", float("nan")),
                metrics.get("orientation_error_deg", float("nan")),
            ),
        ]
        for line_index, line in enumerate(lines):
            cv2.putText(
                frame, line, (20, 34 + 28 * line_index), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA
            )
        if writer is not None:
            writer.write(frame)
        if show_window:
            cv2.imshow("Dynamic Cage tracking", frame)
            window_created = True
            delay_ms = int(round(1000.0 / float(video_fps))) if realtime_display else 1
            cv2.waitKey(max(delay_ms, 1))

    try:
        for episode_index in tqdm(
            range(int(episodes)), desc="Dynamic Cage evaluation", unit="episode"
        ):
            trajectory = None
            initial_joints = None
            trajectory_total_turn_deg = None
            trajectory_endpoint_ratio = None
            trajectory_setup_metrics = None
            rejection_counts = {
                "initial_near_pose_ik_or_collision": 0,
                "initial_causal_window_not_visible": 0,
                "trajectory_total_turn_too_large": 0,
                "trajectory_endpoint_too_close": 0,
                "trajectory_too_close_to_base": 0,
                "trajectory_too_close_to_base_axis": 0,
                "trajectory_too_far_from_base": 0,
                "trajectory_height_out_of_bounds": 0,
            }
            for _ in range(max_attempts):
                turn_sign_mode = str(
                    config["trajectory"].get("turn_sign_mode", "random_v1")
                )
                fixed_validation_profiles = bool(
                    config["trajectory"].get("validation_profiles", [])
                )
                if fixed_validation_profiles:
                    profile_count = len(config["trajectory"]["validation_profiles"])
                    proposal_track_id = int(seed) % profile_count + episode_index
                elif turn_sign_mode == "alternating_track_id_v1":
                    proposal_track_id = episode_index
                else:
                    proposal_track_id = next_track_id
                    next_track_id += 1
                proposal = generator.generate(proposal_track_id)
                proposal_turn_deg, proposal_endpoint_ratio = _open_curve_metrics(proposal)
                if proposal_turn_deg > maximum_total_turn_deg:
                    rejection_counts["trajectory_total_turn_too_large"] += 1
                    continue
                if proposal_endpoint_ratio < minimum_endpoint_ratio:
                    rejection_counts["trajectory_endpoint_too_close"] += 1
                    continue
                proposal_metrics = _trajectory_workspace_and_speed_metrics(proposal)
                if proposal_metrics["minimum_base_distance_m"] < minimum_base_distance:
                    rejection_counts["trajectory_too_close_to_base"] += 1
                    continue
                if (
                    proposal_metrics["minimum_base_xy_distance_m"]
                    < minimum_base_xy_distance
                ):
                    rejection_counts["trajectory_too_close_to_base_axis"] += 1
                    continue
                if proposal_metrics["maximum_base_distance_m"] > maximum_base_distance:
                    rejection_counts["trajectory_too_far_from_base"] += 1
                    continue
                if (
                    proposal_metrics["minimum_height_m"] < minimum_height
                    or proposal_metrics["maximum_height_m"] > maximum_height
                ):
                    rejection_counts["trajectory_height_out_of_bounds"] += 1
                    continue
                proposal_joints, _, proposal_reason = _near_object_initial_joints(
                    observation_planner,
                    proposal,
                    initial_standoffs,
                    minimum_initial_visibility,
                    initial_visibility_lookahead,
                    initial_trend_offsets,
                )
                if proposal_joints is None:
                    rejection_counts[proposal_reason] += 1
                if proposal_joints is not None:
                    trajectory, initial_joints = proposal, proposal_joints
                    trajectory_total_turn_deg = proposal_turn_deg
                    trajectory_endpoint_ratio = proposal_endpoint_ratio
                    trajectory_setup_metrics = proposal_metrics
                    break
            if trajectory is None:
                raise RuntimeError(
                    "Could not initialize a near-object PPO track after {} attempts: {}".format(
                        max_attempts, rejection_counts
                    )
                )
            data = ur5e.new_data()
            selected_trajectories.append(trajectory)
            ur5e.reset_data(data, initial_joints)
            data.mocap_pos[target_mocap_id] = trajectory.true_positions[0]
            ur5e.mujoco.mj_forward(ur5e.model, data)
            reconstructor = CausalFixedAlphaCageReconstructor(config, alpha)
            cage_ik = TimeConstrainedDifferentialIK(ur5e, ik_config)
            command_q = ur5e.joints(data)
            initial_object_distance_m = float(
                np.linalg.norm(
                    ur5e.tcp_pose6(data)[:3] - trajectory.true_positions[0]
                )
            )
            consecutive_misses = 0
            cage_target = None
            phase = "visual_warmup_hold"
            ppo_started = False
            ppo_start_time = None
            failed = False
            failure_reason = ""
            latest_metrics = {}
            trace = []
            open_video() if episode_index == 0 else None
            global_control_step = 0

            for frame_index, timestamp in enumerate(trajectory.timestamps):
                timestamp = float(timestamp)
                true_position = trajectory.true_positions[frame_index].copy()
                data.mocap_pos[target_mocap_id] = true_position
                ur5e.mujoco.mj_forward(ur5e.model, data)
                detection = camera.capture(data, timestamp, frame_index)
                if detection.observation.valid:
                    consecutive_misses = 0
                else:
                    consecutive_misses += 1
                filtered_frame, reconstruction = reconstructor.update(detection.observation)
                if consecutive_misses >= miss_limit:
                    failed = True
                    failure_reason = "consecutive_visual_misses"
                    if filtered_frame is not None:
                        data.mocap_pos[target_mocap_id] = filtered_frame.position
                        ur5e.mujoco.mj_forward(ur5e.model, data)
                    break

                if detection.observation.valid:
                    position_source = "raw_rgbd"
                elif filtered_frame is not None:
                    position_source = "kalman_predict_only"
                else:
                    position_source = "unavailable_hold"

                current_tcp = ur5e.tcp_pose6(data)
                if frame_index < warmup_frames:
                    # These five frames only initialize the visual motion state.
                    # The robot must not react to RGB-D or simulator truth yet.
                    cage_target = None
                    phase = "visual_warmup_hold"
                elif reconstruction is not None:
                    cage_target = reconstruction["pose6"].copy()
                    error = relative_pose_error6(cage_target, current_tcp)
                    position_error = float(np.linalg.norm(error[:3]))
                    orientation_error = float(np.rad2deg(np.linalg.norm(error[3:])))
                    if (
                        not ppo_started
                        and position_error < switch_position
                        and orientation_error < switch_orientation
                    ):
                        ppo_started = True
                        ppo_start_time = timestamp
                    if ppo_started:
                        phase = "ppo_local_tracking"
                    else:
                        phase = "cage_jacobian_approach"
                        cage_ik.reset(
                            current_tcp, cage_target, visual_dt, ur5e.joint_velocities(data)
                        )
                else:
                    cage_target = None
                    phase = "waiting_for_causal_cage_hold"

                for substep in range(control_steps_per_visual):
                    if frame_index + 1 < len(trajectory.timestamps):
                        fraction = float(substep + 1) / control_steps_per_visual
                        moving_target = (
                            (1.0 - fraction) * trajectory.true_positions[frame_index]
                            + fraction * trajectory.true_positions[frame_index + 1]
                        )
                    else:
                        moving_target = trajectory.true_positions[frame_index]
                    data.mocap_pos[target_mocap_id] = moving_target
                    if phase == "cage_jacobian_approach" and cage_target is not None:
                        switch_error = relative_pose_error6(
                            cage_target, ur5e.tcp_pose6(data)
                        )
                        if (
                            float(np.linalg.norm(switch_error[:3])) < switch_position
                            and float(np.rad2deg(np.linalg.norm(switch_error[3:])))
                            < switch_orientation
                        ):
                            ppo_started = True
                            ppo_start_time = timestamp + substep * control_dt
                            phase = "ppo_local_tracking"
                    if phase == "ppo_local_tracking" and cage_target is not None:
                        raw = build_ppo_state(
                            cage_target,
                            ur5e.tcp_pose6(data),
                            ur5e.joints(data),
                            ur5e.joint_velocities(data),
                        )[None, :]
                        normalized = trainer.normalize_state(raw, update=False)
                        tensor = torch.as_tensor(
                            normalized, dtype=torch.float32, device=trainer.device
                        )
                        with torch.no_grad():
                            decision = trainer.sample_for_environment(
                                tensor, raw, stage, deterministic=True
                            )
                        executed = decision["executed_action"].detach().cpu().numpy()[0]
                        desired_velocity = np.clip(executed, -1.0, 1.0) * action_limit
                    elif phase == "cage_jacobian_approach" and cage_ik.goal_pose6 is not None:
                        desired_velocity = cage_ik.command(
                            data, (substep + 1) * control_dt
                        ).joint_velocity_radps
                    else:
                        desired_velocity = np.zeros(6, dtype=float)
                    command_q = np.clip(
                        command_q + desired_velocity * control_dt, safe_lower, safe_upper
                    )
                    ur5e.step_position_target(data, command_q)
                    global_control_step += 1
                    if cage_target is not None:
                        error = relative_pose_error6(cage_target, ur5e.tcp_pose6(data))
                        position_error_mm = 1000.0 * float(np.linalg.norm(error[:3]))
                        orientation_error = float(np.rad2deg(np.linalg.norm(error[3:])))
                    else:
                        position_error_mm = float("nan")
                        orientation_error = float("nan")
                    latest_metrics = {
                        "detected": bool(detection.detected),
                        "consecutive_misses": consecutive_misses,
                        "position_error_mm": position_error_mm,
                        "orientation_error_deg": orientation_error,
                    }
                    if (
                        overview is not None
                        and video_stride is not None
                        and (show_window or episode_index == 0)
                        and global_control_step % video_stride == 0
                    ):
                        render_frame(
                            data, trajectory, cage_target, phase,
                            timestamp + (substep + 1) * control_dt, latest_metrics
                        )

                trace.append(
                    {
                        "frame": frame_index,
                        "timestamp_s": timestamp,
                        "detected": bool(detection.detected),
                        "measurement_accepted": bool(
                            filtered_frame is not None and filtered_frame.measurement_accepted
                        ),
                        "position_source": position_source,
                        "phase": phase,
                        "cage_available": cage_target is not None,
                        "position_error_mm": latest_metrics.get("position_error_mm"),
                        "orientation_error_deg": latest_metrics.get("orientation_error_deg"),
                        "reconstruction": (
                            None
                            if reconstruction is None
                            else {
                                key: (
                                    value.tolist() if isinstance(value, np.ndarray) else value
                                )
                                for key, value in reconstruction.items()
                                if key != "radius_audit"
                            }
                        ),
                    }
                )

            cage_rows = [row for row in trace if row["cage_available"]]
            position_values = np.asarray(
                [row["position_error_mm"] for row in cage_rows], dtype=float
            )
            orientation_values = np.asarray(
                [row["orientation_error_deg"] for row in cage_rows], dtype=float
            )
            summary = {
                "episode": episode_index,
                "track_id": int(trajectory.track_id),
                "trajectory_family": str(trajectory.family),
                "failed": failed,
                "failure_reason": failure_reason,
                "duration_s": float(trajectory.timestamps[-1]),
                "trajectory_total_turn_deg": trajectory_total_turn_deg,
                "trajectory_endpoint_displacement_ratio": trajectory_endpoint_ratio,
                "trajectory_setup_metrics": trajectory_setup_metrics,
                "trajectory_rejection_counts": rejection_counts,
                "initial_tcp_object_distance_m": initial_object_distance_m,
                "visual_frames": len(trace),
                "detection_rate": float(np.mean([row["detected"] for row in trace])) if trace else 0.0,
                "ppo_started": ppo_started,
                "ppo_start_time_s": ppo_start_time,
                "cage_frame_count": len(cage_rows),
                "cage_position_error_mean_mm": float(np.nanmean(position_values)) if len(position_values) else None,
                "cage_position_error_p95_mm": float(np.nanpercentile(position_values, 95)) if len(position_values) else None,
                "cage_orientation_error_mean_deg": float(np.nanmean(orientation_values)) if len(orientation_values) else None,
                "cage_orientation_error_p95_deg": float(np.nanpercentile(orientation_values, 95)) if len(orientation_values) else None,
                "trace": trace,
            }
            summaries.append(summary)
    finally:
        camera.close()
        observation_planner.close()
        if writer is not None:
            writer.release()
        if overview is not None:
            overview.close()
        if window_created:
            cv2.destroyWindow("Dynamic Cage tracking")

    report = {
        "contract": "near_initial_pose_hold5_then_causal_cage_jacobian_or_ppo_v3",
        "checkpoint": str(checkpoint_path),
        "checkpoint_update": int(checkpoint.get("completed_updates", -1)),
        "ppo_stage": str(stage["name"]),
        "normalized_arc_position": alpha,
        "cage_reconstruction_source": "rgbd_kalman_causal_local_spline_only",
        "visual_hold_warmup_frames": warmup_frames,
        "ppo_switch_position_m": switch_position,
        "ppo_switch_orientation_deg": switch_orientation,
        "detection_rate_hz": float(config["timing"]["detection_rate_hz"]),
        "control_rate_hz": float(settings["control_rate_hz"]),
        "episode_count": len(summaries),
        "episode_success_rate": float(
            np.mean([not row["failed"] and row["ppo_started"] for row in summaries])
        ) if summaries else 0.0,
        "visual_failure_rate": float(np.mean([row["failed"] for row in summaries])) if summaries else 0.0,
        "episodes": summaries,
        "video": None if video_output is None else str(video_output),
        "live_window": bool(show_window),
        "realtime_display": bool(realtime_display),
        "gripper_visualization": bool(show_gripper),
        "trajectories": None if trajectory_output is None else str(trajectory_output),
    }
    if trajectory_output is not None:
        save_generated_trajectories(selected_trajectories, trajectory_output)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
