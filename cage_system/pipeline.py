import json
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

from cage_system.audit_io import open_audit_text 
from cage_system.cage.candidates import CandidateGenerator, SpeedRadiusMapper, transform_from_config
from cage_system.cage.est import DisturbanceTape, MujocoESTEvaluator
from cage_system.cage.feasibility import CandidateFeasibilityChecker, WorkspaceFeasibilityBackend
from cage_system.cage.internal_region import CageInternalRegion
from cage_system.cage.plane import LocalMotionPlaneEstimator
from cage_system.cage.selector import CandidateSelector
from cage_system.geometry import make_transform, transform_to_pose6
from cage_system.io import validate_trajectory_detection_period
from cage_system.learning.dataset import CageDatasetArrays, make_cage_input
from cage_system.learning.history_quality import evaluate_history_quality
from cage_system.learning.state_arc_dataset import (
    STATE_ARC_DATASET_CONTRACT,
    CageStateArcDatasetArrays,
)
from cage_system.trajectory.generator import GeneratedTrajectory
from cage_system.trajectory.offline import fit_true_trajectory
from cage_system.trajectory.online import CausalHistoryFrame
from cage_system.types import ObjectState


def _array(value):
    return np.asarray(value).tolist()


class LabelGenerator:
    def __init__(self, config, feasibility_backend=None):
        self.config = config
        self.dataset_contract = str(
            config["labeling"].get("dataset_contract", "legacy_history_pose_v1")
        )
        self.state_arc_mode = self.dataset_contract == STATE_ARC_DATASET_CONTRACT
        self.nominal_detection_period_s = 1.0 / float(
            config["timing"]["detection_rate_hz"]
        )
        gripper = config["gripper"]
        self.region = CageInternalRegion(
            gripper["internal_region_endpoint_frame"],
            gripper["safety_gap_m"],
        )
        transform_tcp_endpoint = transform_from_config(gripper["T_tcp_endpoint"])
        self.candidate_generator = CandidateGenerator(config["candidate"], transform_tcp_endpoint)
        if feasibility_backend is None:
            if config["labeling"]["feasibility_backend"] != "workspace":
                raise RuntimeError(
                    "The requested production feasibility backend must be supplied explicitly"
                )
            feasibility_backend = WorkspaceFeasibilityBackend(config["robot"])
        self.feasibility = CandidateFeasibilityChecker(
            config["robot"],
            config["candidate"],
            config["timing"],
            feasibility_backend,
        )
        self.est = MujocoESTEvaluator(
            config["est"], gripper, config["vision"], self.region
        )
        self.selector = CandidateSelector(config["est"])

    def generate(self, trajectories: List[GeneratedTrajectory], audit_path: Optional[str] = None):
        validate_trajectory_detection_period(
            trajectories, self.nominal_detection_period_s
        )
        inputs, labels, track_ids, timestamps = [], [], [], []
        best_poses, radii, arc_starts, arc_ends = [], [], [], []
        est_means, est_stds = [], []
        audit_handle = None
        if audit_path:
            audit_destination = Path(audit_path)
            audit_handle = open_audit_text(audit_destination, "w")
        try:
            for trajectory in trajectories:
                track_samples = self._generate_track(trajectory, audit_handle)
                if self.state_arc_mode:
                    for sample in track_samples:
                        (
                            state_input,
                            arc_target,
                            best_pose,
                            timestamp,
                            radius,
                            arc_start,
                            arc_end,
                            est_mean,
                            est_std,
                        ) = sample
                        inputs.append(state_input)
                        labels.append(arc_target)
                        best_poses.append(best_pose)
                        track_ids.append(trajectory.track_id)
                        timestamps.append(timestamp)
                        radii.append(radius)
                        arc_starts.append(arc_start)
                        arc_ends.append(arc_end)
                        est_means.append(est_mean)
                        est_stds.append(est_std)
                else:
                    for input_matrix, label, timestamp in track_samples:
                        inputs.append(input_matrix)
                        labels.append(label)
                        track_ids.append(trajectory.track_id)
                        timestamps.append(timestamp)
        finally:
            if audit_handle is not None:
                audit_handle.close()
        if self.state_arc_mode:
            if not inputs:
                return CageStateArcDatasetArrays.empty(
                    self.nominal_detection_period_s
                )
            return CageStateArcDatasetArrays(
                np.asarray(inputs, dtype=float),
                np.asarray(labels, dtype=float).reshape(-1, 1),
                np.asarray(best_poses, dtype=float),
                np.asarray(track_ids, dtype=int),
                np.asarray(timestamps, dtype=float),
                np.asarray(radii, dtype=float),
                np.asarray(arc_starts, dtype=float),
                np.asarray(arc_ends, dtype=float),
                np.asarray(est_means, dtype=float),
                np.asarray(est_stds, dtype=float),
                self.nominal_detection_period_s,
            ).validate()
        if not inputs:
            return CageDatasetArrays(
                np.empty((0, 5, 4)),
                np.empty((0, 6)),
                np.empty((0,), dtype=int),
                np.empty((0,)),
                self.nominal_detection_period_s,
            )
        return CageDatasetArrays(
            np.asarray(inputs),
            np.asarray(labels),
            np.asarray(track_ids),
            np.asarray(timestamps),
            self.nominal_detection_period_s,
        ).validate()

    def _generate_track(self, trajectory: GeneratedTrajectory, audit_handle):
        config = self.config
        offline = fit_true_trajectory(
            trajectory.timestamps,
            trajectory.true_positions,
            config["trajectory"],
        )
        history = deque(maxlen=5)
        plane_estimator = LocalMotionPlaneEstimator(config["plane"])
        whole_curve_times = np.linspace(offline.start_time, offline.end_time, 160)
        whole_curve_points = offline.spline.evaluate(whole_curve_times)
        whole_curve_centered = whole_curve_points - np.mean(whole_curve_points, axis=0)
        whole_curve_covariance = (
            whole_curve_centered.T @ whole_curve_centered
            / max(len(whole_curve_points) - 1, 1)
        )
        whole_curve_eigenvalues, whole_curve_eigenvectors = np.linalg.eigh(
            whole_curve_covariance
        )
        whole_curve_order = np.argsort(whole_curve_eigenvalues)
        whole_curve_eigenvalues = whole_curve_eigenvalues[whole_curve_order]
        whole_curve_eigenvectors = whole_curve_eigenvectors[:, whole_curve_order]
        whole_curve_plane_ratio = float(
            whole_curve_eigenvalues[1] / max(whole_curve_eigenvalues[2], 1e-12)
        )
        global_plane_normal = None
        if (not self.state_arc_mode) and whole_curve_plane_ratio >= float(
            config["plane"]["global_planar_curve_ratio_threshold"]
        ):
            global_plane_normal = whole_curve_eigenvectors[:, 0]
            reference = np.asarray(
                config["plane"]["base_reference_direction"], dtype=float
            )
            if np.dot(global_plane_normal, reference) < 0.0:
                global_plane_normal = -global_plane_normal
            # Whole-curve PCA only anchors the sign/initial estimate.  It must
            # not replace the per-sample local plane on a quasi-planar track.
            plane_estimator.set_track_reference(global_plane_normal)
        radius_mapper = SpeedRadiusMapper(config["candidate"])
        current_joints = np.asarray(config["robot"]["nominal_joint_state_rad"], dtype=float)
        current_joint_velocity = np.zeros(6)
        previous_best_transform = None
        previous_best_angle = None
        previous_best_timestamp = None
        results = []
        for sample_index, (timestamp, true_position) in enumerate(
            zip(offline.timestamps, offline.positions)
        ):
            history_before_current = list(history)
            timestamp = float(timestamp)
            position, velocity, speed, tangent, acceleration, curvature = offline.spline.kinematics(timestamp)
            current_truth_frame = CausalHistoryFrame(
                timestamp=timestamp,
                position=np.asarray(true_position, dtype=float).copy(),
                speed=speed,
                confidence=1.0,
                measurement_accepted=True,
                curvature=curvature,
                tangent=np.asarray(tangent, dtype=float).copy(),
            )
            minimum_plane_frames = int(
                config["labeling"].get("minimum_local_plane_history_frames", 1)
            )
            if self.state_arc_mode and sample_index + 1 < minimum_plane_frames:
                self._write_audit(
                    audit_handle,
                    {
                        "track_id": trajectory.track_id,
                        "sample_index": sample_index,
                        "label_timestamp": timestamp,
                        "dataset_contract": self.dataset_contract,
                        "valid": False,
                        "invalid_reason": "local_plane_history_warmup",
                        "available_plane_frames": sample_index + 1,
                        "required_plane_frames": minimum_plane_frames,
                    },
                )
                history.append(current_truth_frame)
                continue
            before = max(offline.start_time, timestamp - float(config["plane"]["window_before_s"]))
            if self.state_arc_mode:
                # The new state contract deliberately waits for enough past
                # samples and estimates its plane causally.  No future point or
                # whole-track PCA normal is used to define the network input.
                available_times = np.asarray(offline.timestamps[: sample_index + 1])
                local_times = available_times[available_times >= before - 1e-12]
            else:
                after = min(offline.end_time, timestamp + float(config["plane"]["window_after_s"]))
                local_times = np.linspace(before, after, 24)
            plane = plane_estimator.estimate(
                offline.spline.evaluate(local_times),
                position,
                tangent,
                curvature,
                acceleration,
            )
            radius, radius_audit = radius_mapper.map(speed, timestamp)
            audit = {
                "track_id": trajectory.track_id,
                "sample_index": sample_index,
                "label_timestamp": timestamp,
                "history_timestamps": [item.timestamp for item in history_before_current],
                "history_excludes_current": True,
                "geometry_source": "generated_true_timestamps_and_positions",
                "geometry_uses_observations": False,
                "geometry_uses_kalman": False,
                "history_source": "strictly_previous_generated_true_positions",
                "current_position": _array(position),
                "velocity": _array(velocity),
                "speed": speed,
                "tangent": _array(tangent),
                "acceleration": _array(acceleration),
                "curvature": curvature,
                "plane_normal": _array(plane.normal),
                "dataset_contract": self.dataset_contract,
                "orientation_branch_contract": (
                    "candidate_z_equals_stabilized_positive_motion_plane_normal"
                ),
                "plane_near_line": plane.is_near_line,
                "whole_curve_plane_ratio": whole_curve_plane_ratio,
                "whole_curve_plane_reference_available": global_plane_normal is not None,
                "used_global_planar_curve": False,
                "used_local_plane_estimate": True,
                "radius": radius,
                "radius_mapping": radius_audit,
                "valid": False,
                "invalid_reason": "",
            }
            history_quality = evaluate_history_quality(
                history_before_current,
                timestamp,
                config["timing"],
                config["history_quality"],
            )
            audit["history_quality"] = {
                "valid": history_quality.valid,
                "reason": history_quality.reason,
                "missed_detections": history_quality.missed_detections,
                "maximum_interval_s": history_quality.maximum_interval_s,
                "curvature_m_inv": history_quality.curvature_m_inv,
                "five_frame_turn_deg": history_quality.five_frame_turn_deg,
            }
            if not history_quality.valid:
                audit["invalid_reason"] = history_quality.reason
                history.append(current_truth_frame)
                self._write_audit(audit_handle, audit)
                continue
            if timestamp >= offline.end_time - 1e-6:
                audit["invalid_reason"] = "no_future_curve_after_current_time"
                history.append(current_truth_frame)
                self._write_audit(audit_handle, audit)
                continue

            arc = None
            intersection_error = "recorded_future_tail_does_not_reach_candidate_circle"
            nominal_horizon = float(config["candidate"]["future_window_s"])
            maximum_horizon = float(config["candidate"]["future_window_max_s"])
            reach_horizon = 1.15 * radius / max(float(speed), 0.04)
            adaptive_horizon = min(
                maximum_horizon, max(nominal_horizon, reach_horizon)
            )
            horizons = [
                nominal_horizon,
                adaptive_horizon,
                maximum_horizon,
            ]
            horizons = list(dict.fromkeys(horizons))
            audit["arc_search_horizons_s"] = horizons
            for horizon in horizons:
                future_end = min(offline.end_time, timestamp + horizon)
                if future_end <= timestamp + 1e-6:
                    continue
                try:
                    arc = self.candidate_generator.construct_arc(
                        position,
                        tangent,
                        plane.normal,
                        radius,
                        timestamp,
                        future_end,
                        lambda query_time: offline.spline.evaluate(query_time),
                    )
                    break
                except ValueError as error:
                    intersection_error = (
                        "recorded_future_tail_does_not_reach_candidate_circle"
                        if str(error) == "future_curve_does_not_reach_candidate_circle"
                        else str(error)
                    )
            if arc is None:
                audit["invalid_reason"] = intersection_error
                history.append(current_truth_frame)
                self._write_audit(audit_handle, audit)
                continue

            if self.state_arc_mode and arc.degenerate:
                audit["invalid_reason"] = "degenerate_candidate_arc"
                history.append(current_truth_frame)
                self._write_audit(audit_handle, audit)
                continue

            # Candidate sampling is geometry-only. The preceding winning pose
            # must not bias the current candidate set or EST optimum.
            angles = self.candidate_generator.sample_arc(arc, None)
            candidates = self.candidate_generator.build_poses(arc, angles)
            circle_angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
            circle = arc.center + arc.radius * (
                np.cos(circle_angles)[:, None] * arc.u_axis
                + np.sin(circle_angles)[:, None] * arc.v_axis
            )
            arc_angles = np.linspace(arc.angle_start, arc.angle_end, 48)
            arc_points = arc.center + arc.radius * (
                np.cos(arc_angles)[:, None] * arc.u_axis
                + np.sin(arc_angles)[:, None] * arc.v_axis
            )
            future_times, future_points = offline.future_segment(
                timestamp, float(config["candidate"]["future_window_max_s"]), 80
            )
            audit.update(
                {
                    "candidate_circle": _array(circle),
                    "tangent_intersection": _array(arc.tangent_intersection),
                    "future_intersection": _array(arc.future_intersection),
                    "candidate_arc": _array(arc_points),
                    "arc_degenerate": arc.degenerate,
                    "arc_clipped": arc.clipped,
                    "future_curve_times": _array(future_times),
                    "future_curve": _array(future_points),
                }
            )
            tape_seed = int(config["system"]["seed"]) + 1000003 * trajectory.track_id + sample_index
            repeat_count = int(config["est"]["repeat_count"])
            tape_seeds = [tape_seed + 15485863 * repeat for repeat in range(repeat_count)]
            tapes = [
                DisturbanceTape.generate(config["est"], seed, arc=arc)
                for seed in tape_seeds
            ]
            audit["est_repeat_count"] = repeat_count
            audit["est_repeat_seeds"] = tape_seeds
            object_state = ObjectState(
                timestamp,
                position.copy(),
                velocity.copy(),
            )
            candidate_audits = []
            est_results = {}
            if previous_best_timestamp is None:
                available_dt = float(
                    config["labeling"]["initial_label_transition_time_s"]
                )
            else:
                available_dt = max(
                    float(timestamp) - float(previous_best_timestamp),
                    1.0 / float(config["timing"]["detection_rate_hz"]),
                )
            for candidate in candidates:
                transform_base_endpoint = make_transform(
                    candidate.endpoint_position_base, candidate.endpoint_rotation_base
                )
                radius_error = abs(np.linalg.norm(candidate.endpoint_position_base - position) - radius)
                plane_error = abs(np.dot(candidate.endpoint_position_base - position, plane.normal))
                # The supervision contract is directional, not merely plane
                # parallelism: candidate z must equal the stabilized positive
                # local normal.  A negative dot product is never accepted as an
                # equivalent orientation.
                z_alignment = float(
                    np.dot(candidate.endpoint_rotation_base[:, 2], plane.normal)
                )
                opening_alignment = float(
                    np.dot(
                        candidate.endpoint_rotation_base[:, 0],
                        (position - candidate.endpoint_position_base) / radius,
                    )
                )
                initial_capture_membership = self.region.evaluate(
                    transform_base_endpoint, object_state.position
                )
                if radius_error > 1e-7:
                    candidate.rejection_reasons.append("endpoint_not_on_circle")
                if plane_error > 1e-7:
                    candidate.rejection_reasons.append("endpoint_not_in_motion_plane")
                if abs(z_alignment - 1.0) > 1e-7:
                    candidate.rejection_reasons.append("gripper_plane_misaligned")
                if opening_alignment < 0.999:
                    candidate.rejection_reasons.append("opening_direction_invalid")
                candidate.feasible = not candidate.rejection_reasons
                if candidate.feasible:
                    feasibility_joints = (
                        np.asarray(
                            config["robot"]["nominal_joint_state_rad"], dtype=float
                        )
                        if self.state_arc_mode
                        else current_joints
                    )
                    self.feasibility.check(
                        candidate,
                        feasibility_joints,
                        np.zeros(6) if self.state_arc_mode else current_joint_velocity,
                        available_dt,
                        None if self.state_arc_mode else previous_best_transform,
                    )
                if candidate.feasible:
                    est_runs = [
                        self.est.evaluate(
                            object_state,
                            transform_base_endpoint,
                            tape,
                        )
                        for tape in tapes
                    ]
                    est_results[candidate.candidate_id] = est_runs
                    occupancy_ratios = np.asarray(
                        [run.occupancy_ratio for run in est_runs], dtype=float
                    )
                    candidate.metrics.update(
                        {
                            "est_repeat_count": repeat_count,
                            "est_occupancy_by_repeat": occupancy_ratios.tolist(),
                            "est_occupancy_mean": float(np.mean(occupancy_ratios)),
                            "est_inside_nodes_by_repeat": [run.inside_nodes for run in est_runs],
                            "est_outside_nodes_by_repeat": [run.escaped_nodes for run in est_runs],
                            "est_illegal_nodes_by_repeat": [run.illegal_nodes for run in est_runs],
                            "est_total_nodes_by_repeat": [run.total_nodes for run in est_runs],
                        }
                    )
                candidate_audits.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "endpoint_position": _array(candidate.endpoint_position_base),
                        "candidate_radius_m": radius,
                        "tcp_pose6": _array(transform_to_pose6(candidate.tcp_transform_base)),
                        "rotation_matrix": _array(candidate.endpoint_rotation_base),
                        "arc_angle": candidate.arc_angle,
                        "normal_branch": candidate.normal_branch,
                        "feasible": candidate.feasible,
                        "rejection_reasons": candidate.rejection_reasons,
                        "root_inside_capture_set": initial_capture_membership.inside,
                        "root_capture_margin": initial_capture_membership.minimum_margin,
                        "metrics": candidate.metrics,
                    }
                )
            selection = self.selector.select(candidates)
            audit["candidates"] = candidate_audits
            audit["ranked_candidate_ids"] = selection.ranked_candidate_ids
            if selection.valid:
                best = selection.best
                label = transform_to_pose6(best.tcp_transform_base)
                if self.state_arc_mode:
                    signed_turn = float(
                        np.dot(np.cross(tangent, acceleration), plane.normal)
                    )
                    signed_curvature = float(
                        curvature
                        * (1.0 if signed_turn >= 0.0 else -1.0)
                    )
                    state_input = np.concatenate(
                        [
                            np.asarray(position, dtype=float),
                            np.asarray(velocity, dtype=float),
                            np.asarray(acceleration, dtype=float),
                            np.asarray(plane.normal, dtype=float),
                            np.asarray([signed_curvature], dtype=float),
                        ]
                    )
                    arc_span = float(arc.angle_end - arc.angle_start)
                    normalized_arc = float(
                        np.clip(
                            (best.arc_angle - arc.angle_start) / arc_span,
                            0.0,
                            1.0,
                        )
                    )
                    best_occupancies = np.asarray(
                        best.metrics["est_occupancy_by_repeat"], dtype=float
                    )
                    results.append(
                        (
                            state_input,
                            normalized_arc,
                            label,
                            timestamp,
                            radius,
                            arc.angle_start,
                            arc.angle_end,
                            float(np.mean(best_occupancies)),
                            float(np.std(best_occupancies)),
                        )
                    )
                    audit["best_normalized_arc_position"] = normalized_arc
                    audit["signed_curvature"] = signed_curvature
                else:
                    input_matrix = make_cage_input(history_before_current)
                    results.append((input_matrix, label, timestamp))
                audit["valid"] = True
                audit["best_candidate_id"] = best.candidate_id
                audit["best_tcp_pose6"] = _array(label)
                best_est_runs = est_results[best.candidate_id]
                best_est = best_est_runs[0]
                audit["best_est_repeat_summary"] = [
                    {
                        "inside_nodes": run.inside_nodes,
                        "outside_nodes": run.escaped_nodes,
                        "illegal_nodes": run.illegal_nodes,
                        "total_nodes": run.total_nodes,
                        "occupancy_ratio": run.occupancy_ratio,
                    }
                    for run in best_est_runs
                ]
                audit["best_est_nodes_base"] = _array(best_est.nodes_base)
                audit["best_est_node_status"] = _array(best_est.node_status)
                audit["best_est_node_parent_indices"] = _array(
                    best_est.node_parent_indices
                )
                audit["best_est_node_times"] = _array(best_est.node_times)
                if not self.state_arc_mode:
                    previous_best_transform = best.tcp_transform_base.copy()
                    previous_best_angle = best.arc_angle
                    previous_best_timestamp = float(timestamp)
                if not self.state_arc_mode and best.ik_joints is not None:
                    previous_joints = current_joints.copy()
                    current_joints = best.ik_joints.copy()
                    current_joint_velocity = (current_joints - previous_joints) / available_dt
            else:
                audit["invalid_reason"] = selection.reason
            history.append(current_truth_frame)
            self._write_audit(audit_handle, audit)
        return results

    @staticmethod
    def _write_audit(handle, record):
        if handle is not None:
            def numpy_json(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.bool_):
                    return bool(value)
                if isinstance(value, np.integer):
                    return int(value)
                if isinstance(value, np.floating):
                    return float(value)
                raise TypeError("Unsupported audit value {}".format(type(value).__name__))

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=numpy_json,
                )
                + "\n"
            )
            handle.flush()
