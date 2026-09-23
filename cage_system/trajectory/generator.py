from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from cage_system.geometry import normalize, plane_basis, rotation_as_matrix
from cage_system.types import PointObservation


@dataclass
class GeneratedTrajectory:
    track_id: int
    family: str
    timestamps: np.ndarray
    true_positions: np.ndarray
    observations: List[PointObservation]
    # Reference normal at the beginning of the quasi-planar track. Label
    # generation estimates a new local normal at every sample.
    plane_normal: np.ndarray
    # Visual/detection period under which this trajectory was generated.
    # Legacy archives do not contain it and load with ``None`` so consumers
    # can reject them explicitly instead of silently interpreting 50 ms data
    # as a different visual clock.
    nominal_detection_period_s: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    curvature_min_m_inv: float
    curvature_max_m_inv: float
    five_frame_curvature_range_min_m_inv: float
    five_frame_curvature_range_max_m_inv: float
    local_plane_total_variation_deg: float
    local_plane_max_step_deg: float
    five_frame_turn_min_deg: float
    five_frame_turn_max_deg: float
    five_frame_plane_ratio_min: float

    def as_dict(self):
        return {
            "curvature_min_m_inv": self.curvature_min_m_inv,
            "curvature_max_m_inv": self.curvature_max_m_inv,
            "five_frame_curvature_range_min_m_inv": (
                self.five_frame_curvature_range_min_m_inv
            ),
            "five_frame_curvature_range_max_m_inv": (
                self.five_frame_curvature_range_max_m_inv
            ),
            "local_plane_total_variation_deg": self.local_plane_total_variation_deg,
            "local_plane_max_step_deg": self.local_plane_max_step_deg,
            "five_frame_turn_min_deg": self.five_frame_turn_min_deg,
            "five_frame_turn_max_deg": self.five_frame_turn_max_deg,
            "five_frame_plane_ratio_min": self.five_frame_plane_ratio_min,
        }


def trajectory_diagnostics(
    timestamps: np.ndarray,
    positions: np.ndarray,
    history_frames: int = 5,
) -> TrajectoryDiagnostics:
    """Measure curvature variation and osculating-plane continuity.

    Diagnostics are computed from the generated Cartesian samples themselves,
    so acceptance covers the data saved to the archive rather than only the
    randomly sampled trajectory parameters.
    """

    timestamps = np.asarray(timestamps, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(timestamps) < max(history_frames, 7):
        raise ValueError("A trajectory needs at least seven samples")
    if positions.shape != (len(timestamps), 3):
        raise ValueError("positions must have shape [sample_count, 3]")
    velocity = np.gradient(positions, timestamps, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, timestamps, axis=0, edge_order=2)
    cross = np.cross(velocity, acceleration)
    speed = np.linalg.norm(velocity, axis=1)
    curvature = np.linalg.norm(cross, axis=1) / np.maximum(speed**3, 1e-12)

    # One-sided endpoint derivatives are less accurate. Use the nearest stable
    # interior value so endpoint history windows are checked without accepting
    # numerical artifacts.
    stable_curvature = curvature.copy()
    stable_curvature[:2] = stable_curvature[2]
    stable_curvature[-2:] = stable_curvature[-3]
    window_ranges = np.asarray(
        [
            np.ptp(stable_curvature[index : index + history_frames])
            for index in range(len(stable_curvature) - history_frames + 1)
        ],
        dtype=float,
    )
    tangent = velocity / np.maximum(speed[:, None], 1e-12)
    five_frame_turns = []
    five_frame_plane_ratios = []
    for index in range(len(tangent) - history_frames + 1):
        dot = float(np.clip(
            np.dot(tangent[index], tangent[index + history_frames - 1]), -1.0, 1.0
        ))
        five_frame_turns.append(np.degrees(np.arccos(dot)))
        window = positions[index : index + history_frames]
        centered_window = window - np.mean(window, axis=0)
        window_eigenvalues = np.linalg.eigvalsh(centered_window.T @ centered_window)
        five_frame_plane_ratios.append(
            float(window_eigenvalues[1] / max(window_eigenvalues[2], 1e-12))
        )

    normal_norms = np.linalg.norm(cross, axis=1)
    valid = normal_norms > 1e-10
    valid[:2] = False
    valid[-2:] = False
    normals = cross[valid] / normal_norms[valid, None]
    if len(normals) < 3:
        raise ValueError("Trajectory does not have enough valid local-plane estimates")
    for index in range(1, len(normals)):
        if np.dot(normals[index], normals[index - 1]) < 0.0:
            normals[index] *= -1.0
    pairwise_dots = np.clip(normals @ normals.T, -1.0, 1.0)
    total_variation = np.degrees(np.arccos(np.min(pairwise_dots)))
    step_dots = np.clip(np.sum(normals[1:] * normals[:-1], axis=1), -1.0, 1.0)
    maximum_step = np.max(np.degrees(np.arccos(step_dots)))

    stable_slice = stable_curvature[2:-2]
    return TrajectoryDiagnostics(
        curvature_min_m_inv=float(np.min(stable_slice)),
        curvature_max_m_inv=float(np.max(stable_slice)),
        five_frame_curvature_range_min_m_inv=float(np.min(window_ranges)),
        five_frame_curvature_range_max_m_inv=float(np.max(window_ranges)),
        local_plane_total_variation_deg=float(total_variation),
        local_plane_max_step_deg=float(maximum_step),
        five_frame_turn_min_deg=float(np.min(five_frame_turns)),
        five_frame_turn_max_deg=float(np.max(five_frame_turns)),
        five_frame_plane_ratio_min=float(np.min(five_frame_plane_ratios)),
    )


class PlanarTrajectoryGenerator:
    """Generate gently twisting quasi-planar tracks for Cage supervision.

    Curvature changes smoothly throughout every track and a small, bounded
    torsion makes successive local motion planes different without turning the
    data into strongly spatial trajectories.
    """

    def __init__(self, config: Dict, seed: int = 0):
        self.config = config
        self.rng = np.random.RandomState(seed)

    def _random_plane(self):
        normal = normalize(self.rng.normal(size=3))
        if normal[2] < 0.25:
            normal[2] = abs(normal[2]) + 0.35
            normal = normalize(normal)
        preferred = normalize(self.rng.normal(size=3))
        u_axis, v_axis = plane_basis(normal, preferred)
        bounds = np.asarray(self.config["plane_origin_bounds"], dtype=float)
        origin = self.rng.uniform(bounds[:, 0], bounds[:, 1])
        return origin, u_axis, v_axis, normal

    def generate(self, track_id: int) -> GeneratedTrajectory:
        maximum_attempts = int(self.config["generation_max_attempts"])
        rejection_counts = {}
        for _ in range(maximum_attempts):
            trajectory = self._generate_candidate(track_id)
            diagnostics = trajectory_diagnostics(
                trajectory.timestamps,
                trajectory.true_positions,
                int(self.config["curvature_history_frames"]),
            )
            reason = self._rejection_reason(diagnostics)
            if reason is None:
                trajectory.observations = self._make_observations(
                    trajectory.timestamps, trajectory.true_positions
                )
                return trajectory
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        raise RuntimeError(
            "Could not generate an accepted quasi-planar trajectory after {} attempts; "
            "rejections={}".format(maximum_attempts, rejection_counts)
        )

    def _generate_candidate(self, track_id: int) -> GeneratedTrajectory:
        validation_profiles = self.config.get("validation_profiles", [])
        profile = None
        if validation_profiles:
            profile = validation_profiles[int(track_id) % len(validation_profiles)]
        duration = self.rng.uniform(*self.config["duration_s"])
        nominal_dt = float(self.config["sample_period_s"])
        timestamp_mode = str(
            self.config.get("timestamp_mode", "jittered_period_v1")
        )
        if timestamp_mode == "fixed_period_v1":
            count = max(8, int(np.round(duration / nominal_dt)) + 1)
            timestamps = nominal_dt * np.arange(count, dtype=float)
            duration = float(timestamps[-1])
        elif timestamp_mode == "jittered_period_v1":
            count = max(8, int(np.floor(duration / nominal_dt)) + 1)
            increments = nominal_dt * self.rng.uniform(0.92, 1.08, size=count - 1)
            timestamps = np.concatenate(([0.0], np.cumsum(increments)))
            timestamps *= duration / timestamps[-1]
        else:
            raise ValueError("Unknown trajectory timestamp_mode '{}'".format(timestamp_mode))
        phase = timestamps / duration
        speed_bounds = (
            profile.get("speed_bounds_mps", self.config["speed_bounds_mps"])
            if profile is not None
            else self.config["speed_bounds_mps"]
        )
        arc_length = np.clip(
            self.rng.uniform(*speed_bounds) * duration,
            float(self.config["path_length_bounds_m"][0]),
            float(self.config["path_length_bounds_m"][1]),
        )
        speed_variation_bounds = self.config.get("speed_variation_fraction_bounds")
        if speed_variation_bounds is None:
            distance = arc_length * phase
        else:
            variation = self.rng.uniform(*speed_variation_bounds)
            cycles = self.rng.uniform(
                *self.config.get("speed_variation_cycles_bounds", [0.75, 1.25])
            )
            phase_offset = self.rng.uniform(0.0, 2.0 * np.pi)
            relative_speed = 1.0 + variation * np.sin(
                2.0 * np.pi * cycles * phase + phase_offset
            )
            progress_increment = 0.5 * (
                relative_speed[:-1] + relative_speed[1:]
            ) * np.diff(phase)
            normalized_progress = np.concatenate(
                ([0.0], np.cumsum(progress_increment))
            )
            normalized_progress /= max(float(normalized_progress[-1]), 1e-12)
            distance = arc_length * normalized_progress

        origin, tangent, normal_direction, reference_normal = self._random_plane()
        if profile is not None and "origin_bounds" in profile:
            origin_bounds = np.asarray(profile["origin_bounds"], dtype=float)
            origin = self.rng.uniform(origin_bounds[:, 0], origin_bounds[:, 1])
        turn_sign_mode = str(self.config.get("turn_sign_mode", "random_v1"))
        if profile is not None and "turn_sign" in profile:
            turn_sign = float(profile["turn_sign"])
        elif turn_sign_mode == "random_v1":
            turn_sign = float(self.rng.choice([-1.0, 1.0]))
        elif turn_sign_mode == "alternating_track_id_v1":
            turn_sign = -1.0 if int(track_id) % 2 == 0 else 1.0
        else:
            raise ValueError("Unknown trajectory turn_sign_mode '{}'".format(turn_sign_mode))
        normal_direction *= turn_sign
        reference_normal *= turn_sign
        frame = np.column_stack((tangent, normal_direction, reference_normal))

        curvature_low_bounds = (
            profile.get(
                "curvature_low_bounds_m_inv",
                self.config["curvature_low_bounds_m_inv"],
            )
            if profile is not None
            else self.config["curvature_low_bounds_m_inv"]
        )
        curvature_high_bounds = (
            profile.get(
                "curvature_high_bounds_m_inv",
                self.config["curvature_high_bounds_m_inv"],
            )
            if profile is not None
            else self.config["curvature_high_bounds_m_inv"]
        )
        curvature_low = self.rng.uniform(*curvature_low_bounds)
        curvature_high = self.rng.uniform(*curvature_high_bounds)
        curvature_trend = None if profile is None else profile.get("curvature_trend")
        if curvature_trend == "up":
            ramp_up = True
        elif curvature_trend == "down":
            ramp_up = False
        else:
            ramp_up = bool(self.rng.randint(0, 2))
        start_curvature, end_curvature = (
            (curvature_low, curvature_high)
            if ramp_up
            else (curvature_high, curvature_low)
        )
        profile_beta = self.rng.uniform(*self.config["curvature_profile_beta_bounds"])
        eased_phase = phase + profile_beta * phase * (1.0 - phase)
        curvature_profile = (
            start_curvature
            + (end_curvature - start_curvature) * eased_phase
        )

        total_twist_rad = np.radians(
            self.rng.uniform(*self.config["local_plane_twist_bounds_deg"])
        ) * float(self.rng.choice([-1.0, 1.0]))
        torsion_shape = 1.0 + self.rng.uniform(-0.25, 0.25) * np.sin(np.pi * phase)
        shape_integral = np.trapz(torsion_shape, distance)
        torsion_profile = total_twist_rad * torsion_shape / max(shape_integral, 1e-12)

        positions = np.empty((count, 3), dtype=float)
        positions[0] = origin
        for index in range(1, count):
            ds = distance[index] - distance[index - 1]
            curvature_mid = 0.5 * (
                curvature_profile[index - 1] + curvature_profile[index]
            )
            torsion_mid = 0.5 * (
                torsion_profile[index - 1] + torsion_profile[index]
            )
            angular_velocity = torsion_mid * frame[:, 0] + curvature_mid * frame[:, 2]
            increment = rotation_as_matrix(Rotation.from_rotvec(angular_velocity * ds))
            next_frame = increment @ frame
            midpoint_tangent = normalize(frame[:, 0] + next_frame[:, 0])
            positions[index] = positions[index - 1] + ds * midpoint_tangent
            frame = next_frame

        profile_name = "curvature_ramp_up" if ramp_up else "curvature_ramp_down"
        if turn_sign_mode == "alternating_track_id_v1":
            direction_name = "negative_turn" if turn_sign < 0.0 else "positive_turn"
            profile_name = "{}_{}".format(direction_name, profile_name)
        if profile is not None:
            profile_name = str(profile.get("name", profile_name))
        return GeneratedTrajectory(
            track_id=track_id,
            family=profile_name,
            timestamps=timestamps,
            true_positions=positions,
            observations=[],
            plane_normal=reference_normal,
            nominal_detection_period_s=nominal_dt,
        )

    def _rejection_reason(self, diagnostics: TrajectoryDiagnostics):
        config = self.config
        if diagnostics.curvature_max_m_inv > float(config["maximum_curvature_m_inv"]):
            return "curvature_above_maximum"
        minimum_range, maximum_range = config[
            "five_frame_curvature_range_bounds_m_inv"
        ]
        if diagnostics.five_frame_curvature_range_min_m_inv < float(minimum_range):
            return "five_frame_curvature_change_too_small"
        if diagnostics.five_frame_curvature_range_max_m_inv > float(maximum_range):
            return "five_frame_curvature_change_too_large"
        if diagnostics.five_frame_turn_min_deg < float(
            config.get("minimum_five_frame_turn_deg", 0.0)
        ):
            return "five_frame_turn_too_small"
        if diagnostics.five_frame_turn_max_deg > float(config["maximum_five_frame_turn_deg"]):
            return "five_frame_turn_too_large"
        if diagnostics.five_frame_plane_ratio_min < float(
            config.get("minimum_five_frame_plane_ratio", 0.0)
        ):
            return "five_frame_plane_spread_too_small"
        minimum_plane, maximum_plane = config["local_plane_total_variation_bounds_deg"]
        if diagnostics.local_plane_total_variation_deg < float(minimum_plane):
            return "local_plane_variation_too_small"
        if diagnostics.local_plane_total_variation_deg > float(maximum_plane):
            return "local_plane_variation_too_large"
        if diagnostics.local_plane_max_step_deg > float(
            config["local_plane_max_step_deg"]
        ):
            return "local_plane_step_too_large"
        return None

    def _make_observations(self, timestamps, positions):
        observations = []
        position_std = float(self.config["observation_noise_std_m"])
        for index, timestamp in enumerate(timestamps):
            dropped = bool(
                index > 0 and self.rng.rand() < float(self.config["dropout_probability"])
            )
            position = positions[index] + self.rng.normal(scale=position_std, size=3)
            if not dropped and self.rng.rand() < float(self.config["outlier_probability"]):
                position += self.rng.normal(scale=float(self.config["outlier_std_m"]), size=3)
            observations.append(
                PointObservation(
                    timestamp=float(timestamp),
                    position=position,
                    valid=not dropped,
                    source_index=index,
                )
            )
        return observations
