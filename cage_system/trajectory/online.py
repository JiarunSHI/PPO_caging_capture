from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
from cage_system.trajectory.kalman import ConstantAccelerationKalman3D
from cage_system.trajectory.spline import SharedCubicSpline3D
from cage_system.types import PointObservation
from cage_system.geometry import normalize


@dataclass
class CausalHistoryFrame:
    timestamp: float
    position: np.ndarray
    speed: float
    confidence: float
    measurement_accepted: bool
    curvature: float
    tangent: np.ndarray


class OnlineTrajectoryEstimator:
    """Causal filter + rolling local spline; never reads a future observation."""

    def __init__(self, trajectory_config, kalman_config, timing_config):
        self.trajectory_config = trajectory_config
        self.timing_config = timing_config
        self.filter = ConstantAccelerationKalman3D(kalman_config)
        self.window_s = float(trajectory_config["online_window_s"])
        self.points: Deque[Tuple[float, np.ndarray]] = deque()
        self.stable_tangent: Optional[np.ndarray] = None
        self.innovation_gate_warmup_measurements = max(
            1, int(kalman_config.get("innovation_gate_warmup_measurements", 1))
        )
        self.bootstrap_measurements = max(
            1, int(kalman_config.get("visual_bootstrap_measurements", 1))
        )
        self.bootstrap_observations = deque(maxlen=self.bootstrap_measurements)
        self.bootstrap_position_std_floor = float(
            kalman_config.get(
                "bootstrap_position_std_floor_m",
                np.sqrt(self.filter.measurement_variance),
            )
        )
        self.bootstrap_velocity_std_floor = float(
            kalman_config.get("bootstrap_velocity_std_floor_mps", 0.03)
        )
        self.bootstrap_acceleration_std_floor = float(
            kalman_config.get("bootstrap_acceleration_std_floor_mps2", 0.20)
        )
        self.bootstrap_max_residual = float(
            kalman_config.get("bootstrap_max_residual_m", float("inf"))
        )
        self.accepted_measurement_count = 0

    def _fit_visual_bootstrap(self):
        times = np.asarray([item[0] for item in self.bootstrap_observations], dtype=float)
        positions = np.asarray(
            [item[1] for item in self.bootstrap_observations], dtype=float
        )
        if len(times) < self.bootstrap_measurements or np.any(np.diff(times) <= 0.0):
            return None
        relative_time = times - times[-1]
        design = np.column_stack(
            (np.ones(len(times)), relative_time, 0.5 * relative_time ** 2)
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
        residuals = positions - design @ coefficients
        if float(np.max(np.linalg.norm(residuals, axis=1))) > self.bootstrap_max_residual:
            return None

        # Coefficients are [position, velocity, acceleration] at the newest
        # visual timestamp.  Preserve their within-axis correlations.
        state = np.concatenate((coefficients[0], coefficients[1], coefficients[2]))
        covariance = np.zeros((9, 9), dtype=float)
        gram_inverse = np.linalg.pinv(design.T @ design)
        residual_dof = max(len(times) - design.shape[1], 1)
        variance_floors = np.square(
            [
                self.bootstrap_position_std_floor,
                self.bootstrap_velocity_std_floor,
                self.bootstrap_acceleration_std_floor,
            ]
        )
        for axis in range(3):
            fitted_variance = float(
                np.sum(residuals[:, axis] ** 2) / residual_dof
            )
            effective_variance = max(self.filter.measurement_variance, fitted_variance)
            block = effective_variance * gram_inverse
            for component in range(3):
                block[component, component] = max(
                    block[component, component], variance_floors[component]
                )
            indices = [axis, axis + 3, axis + 6]
            covariance[np.ix_(indices, indices)] = block
        return state, covariance

    def _finish_frame(self, timestamp, position, velocity, confidence, accepted):
        timestamp = float(timestamp)
        position = np.asarray(position, dtype=float).copy()
        kalman_velocity = np.asarray(velocity, dtype=float).copy()
        self.points.append((timestamp, position.copy()))
        while self.points and timestamp - self.points[0][0] > self.window_s:
            self.points.popleft()

        # During the visual bootstrap these points are raw RGB-D positions and
        # the velocity is a provisional visual fit.  After initialization they
        # are the causal Kalman posterior state.
        curvature = 0.0
        if len(self.points) >= 4:
            times = np.array([item[0] for item in self.points])
            positions = np.array([item[1] for item in self.points])
            age = timestamp - times
            weights = 1.0 + float(self.trajectory_config["recent_weight_gain"]) * np.exp(
                -age / max(self.window_s * 0.35, 1e-3)
            )
            try:
                spline = SharedCubicSpline3D(
                    self.trajectory_config["spline_control_spacing_s"],
                    self.trajectory_config["spline_regularization"],
                ).fit(times, positions, weights)
                curvature_velocity = spline.evaluate(timestamp, 1)
                curvature_acceleration = spline.evaluate(timestamp, 2)
                spline_speed = float(np.linalg.norm(curvature_velocity))
                curvature = (
                    0.0
                    if spline_speed < 1e-6
                    else float(
                        np.linalg.norm(
                            np.cross(curvature_velocity, curvature_acceleration)
                        )
                        / spline_speed**3
                    )
                )
            except (ValueError, np.linalg.LinAlgError):
                pass

        speed = float(np.linalg.norm(kalman_velocity))
        if speed >= 1e-6:
            raw_tangent = kalman_velocity / speed
        elif len(self.points) >= 2:
            delta = self.points[-1][1] - self.points[-2][1]
            delta_norm = float(np.linalg.norm(delta))
            raw_tangent = (
                delta / delta_norm
                if delta_norm >= 1e-9
                else np.array([1.0, 0.0, 0.0])
            )
        else:
            raw_tangent = np.array([1.0, 0.0, 0.0])
        if self.stable_tangent is None:
            tangent = raw_tangent
        else:
            alpha = float(self.trajectory_config.get("tangent_update_alpha", 0.35))
            proposed = normalize((1.0 - alpha) * self.stable_tangent + alpha * raw_tangent)
            dot = float(np.clip(np.dot(self.stable_tangent, proposed), -1.0, 1.0))
            step = float(np.arccos(dot))
            maximum_step = np.deg2rad(
                float(self.trajectory_config.get("causal_tangent_max_step_deg", 12.0))
            )
            if step > maximum_step:
                fraction = maximum_step / max(step, 1e-12)
                tangent = normalize(
                    (1.0 - fraction) * self.stable_tangent + fraction * proposed
                )
            else:
                tangent = proposed
        self.stable_tangent = tangent.copy()
        return CausalHistoryFrame(
            timestamp,
            position,
            speed,
            float(confidence),
            bool(accepted),
            curvature,
            tangent,
        )

    def update(self, observation: PointObservation) -> CausalHistoryFrame:
        if self.bootstrap_measurements > 1 and self.filter.timestamp is None:
            if not observation.valid:
                raise ValueError(
                    "The visual bootstrap requires a valid position measurement"
                )
            timestamp = float(observation.timestamp)
            position = np.asarray(observation.position, dtype=float)
            if position.shape != (3,) or np.any(~np.isfinite(position)):
                raise ValueError("Visual bootstrap positions must be finite 3-vectors")
            self.bootstrap_observations.append((timestamp, position.copy()))
            bootstrap = self._fit_visual_bootstrap()
            if bootstrap is not None:
                state, covariance = bootstrap
                self.filter.initialize_state(state, covariance, timestamp)
                self.accepted_measurement_count = self.bootstrap_measurements
                velocity = state[3:6]
            elif len(self.bootstrap_observations) >= 2:
                first_time, first_position = self.bootstrap_observations[0]
                velocity = (position - first_position) / max(
                    timestamp - first_time, 1e-12
                )
            else:
                velocity = np.zeros(3, dtype=float)
            return self._finish_frame(timestamp, position, velocity, 1.0, True)

        bypass_gate = bool(
            observation.valid
            and self.accepted_measurement_count
            < self.innovation_gate_warmup_measurements
        )
        filtered = self.filter.update(
            observation.timestamp,
            observation.position if observation.valid else None,
            bypass_innovation_gate=bypass_gate,
        )
        if filtered.measurement_accepted:
            self.accepted_measurement_count += 1
        return self._finish_frame(
            filtered.timestamp,
            filtered.position.copy(),
            filtered.velocity.copy(),
            filtered.confidence,
            bool(filtered.measurement_accepted),
        )

    def clear_causal_curve_history(self):
        """Start a new accepted-frame segment without discarding Kalman state."""

        self.points.clear()
        self.stable_tangent = None

    def predict_short_segment(self, sample_count: int = 16):
        """Return a short continuous causal extension, not a single future point."""
        if self.filter.timestamp is None:
            raise RuntimeError("No online state is available")
        horizon = float(self.timing_config["online_prediction_horizon_s"])
        future_dt = np.linspace(0.0, horizon, int(sample_count))
        position = self.filter.state[:3]
        velocity = self.filter.state[3:6]
        acceleration = self.filter.state[6:9]
        decay = np.exp(-future_dt / max(horizon, 1e-3))[:, None]
        points = (
            position
            + future_dt[:, None] * velocity
            + 0.5 * future_dt[:, None] ** 2 * acceleration * decay
        )
        return self.filter.timestamp + future_dt, points
