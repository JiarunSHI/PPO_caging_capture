from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from cage_system.trajectory.kalman import ConstantAccelerationKalman3D
from cage_system.trajectory.spline import SharedCubicSpline3D
from cage_system.types import FilteredState, PointObservation


@dataclass
class OfflineTrajectory:
    observations_base: List[PointObservation]
    filtered_states: List[FilteredState]
    spline: SharedCubicSpline3D

    @property
    def start_time(self):
        return self.spline.start_time

    @property
    def end_time(self):
        return self.spline.end_time

    def future_segment(self, timestamp: float, horizon_s: float, count: int = 80):
        end = min(self.end_time, float(timestamp) + float(horizon_s))
        times = np.linspace(float(timestamp), end, int(count))
        return times, self.spline.evaluate(times)


@dataclass
class GroundTruthTrajectory:
    """Continuous geometry fitted directly to generated, noise-free samples."""

    timestamps: np.ndarray
    positions: np.ndarray
    spline: SharedCubicSpline3D

    @property
    def start_time(self):
        return self.spline.start_time

    @property
    def end_time(self):
        return self.spline.end_time

    def future_segment(self, timestamp: float, horizon_s: float, count: int = 80):
        end = min(self.end_time, float(timestamp) + float(horizon_s))
        times = np.linspace(float(timestamp), end, int(count))
        return times, self.spline.evaluate(times)


def transform_observation_to_base(
    observation: PointObservation, transform_base_camera: np.ndarray
) -> PointObservation:
    rotation_base_camera = transform_base_camera[:3, :3]
    position = rotation_base_camera @ observation.position + transform_base_camera[:3, 3]
    return PointObservation(
        timestamp=float(observation.timestamp),
        position=position,
        valid=observation.valid,
        source_index=observation.source_index,
    )


def preprocess_observations(
    observations: List[PointObservation], transform_base_camera: Optional[np.ndarray] = None
) -> List[PointObservation]:
    transform = np.eye(4) if transform_base_camera is None else np.asarray(transform_base_camera)
    transformed = [transform_observation_to_base(item, transform) for item in observations]
    transformed.sort(key=lambda item: item.timestamp)
    unique = []
    for observation in transformed:
        if unique and abs(observation.timestamp - unique[-1].timestamp) < 1e-9:
            # Keep the latest valid reading for a duplicate hardware timestamp.
            if observation.valid:
                unique[-1] = observation
        else:
            unique.append(observation)
    return unique


def fit_offline_trajectory(
    observations: List[PointObservation], trajectory_config, kalman_config, transform_base_camera=None
) -> OfflineTrajectory:
    base_observations = preprocess_observations(observations, transform_base_camera)
    if len(base_observations) < 4:
        raise ValueError("At least four unique observations are needed for an offline trajectory")
    kalman = ConstantAccelerationKalman3D(kalman_config)
    filtered = []
    for observation in base_observations:
        filtered.append(
            kalman.update(observation.timestamp, observation.position if observation.valid else None)
        )
    times = np.asarray([item.timestamp for item in filtered])
    positions = np.asarray([item.position for item in filtered])
    confidences = np.asarray([max(item.confidence, 0.1) for item in filtered])
    spline = SharedCubicSpline3D(
        trajectory_config["spline_control_spacing_s"],
        trajectory_config["spline_regularization"],
    ).fit(times, positions, confidences)
    return OfflineTrajectory(base_observations, filtered, spline)


def fit_true_trajectory(
    timestamps: np.ndarray,
    true_positions: np.ndarray,
    trajectory_config,
) -> GroundTruthTrajectory:
    """Fit label geometry without observations, dropout, outliers, or Kalman state."""

    times = np.asarray(timestamps, dtype=float)
    positions = np.asarray(true_positions, dtype=float)
    if times.ndim != 1 or len(times) < 4:
        raise ValueError("At least four ground-truth timestamps are required")
    if positions.shape != (len(times), 3):
        raise ValueError("Ground-truth positions must have shape [sample_count, 3]")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(positions)):
        raise ValueError("Ground-truth trajectory values must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Ground-truth timestamps must be strictly increasing")

    # Unit weights are intentional: generated truth has no detector confidence
    # and must not inherit observation/Kalman confidence semantics.
    spline = SharedCubicSpline3D(
        trajectory_config["spline_control_spacing_s"],
        trajectory_config["spline_regularization"],
    ).fit(times, positions)
    return GroundTruthTrajectory(times.copy(), positions.copy(), spline)
