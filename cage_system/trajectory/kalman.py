from typing import Optional

import numpy as np

from cage_system.types import FilteredState


class ConstantAccelerationKalman3D:
    """Nine-state causal Kalman filter with variable timestamp spacing."""

    def __init__(self, config):
        self.measurement_variance = float(config["measurement_variance_m2"])
        self.jerk_density = float(config["jerk_spectral_density"])
        self.gate = float(config["innovation_gate_chi2"])
        self.max_predict_gap = float(config["max_predict_only_gap_s"])
        self.initial_variances = np.concatenate(
            [
                np.full(3, float(config["initial_position_variance"])),
                np.full(3, float(config["initial_velocity_variance"])),
                np.full(3, float(config["initial_acceleration_variance"])),
            ]
        )
        self.state = np.zeros(9, dtype=float)
        self.covariance = np.diag(self.initial_variances)
        self.timestamp: Optional[float] = None
        self.last_measurement_timestamp: Optional[float] = None
        self.last_innovation_mahalanobis = float("nan")
        self.last_innovation_gate_bypassed = False

    def reset(self, position=None, timestamp=None):
        self.state = np.zeros(9, dtype=float)
        if position is not None:
            self.state[:3] = np.asarray(position, dtype=float)
        self.covariance = np.diag(self.initial_variances)
        self.timestamp = None if timestamp is None else float(timestamp)
        self.last_measurement_timestamp = self.timestamp

    def initialize_state(self, state, covariance, timestamp):
        """Initialize the full local-motion posterior from a visual bootstrap fit."""

        state = np.asarray(state, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if state.shape != (9,) or np.any(~np.isfinite(state)):
            raise ValueError("Kalman bootstrap state must be a finite 9-vector")
        if covariance.shape != (9, 9) or np.any(~np.isfinite(covariance)):
            raise ValueError("Kalman bootstrap covariance must be finite and 9x9")
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
        self.state = state.copy()
        self.covariance = covariance
        self.timestamp = float(timestamp)
        self.last_measurement_timestamp = self.timestamp
        self.last_innovation_mahalanobis = float("nan")
        self.last_innovation_gate_bypassed = False

    @staticmethod
    def transition(dt: float) -> np.ndarray:
        identity = np.eye(3)
        transition = np.eye(9)
        transition[:3, 3:6] = dt * identity
        transition[:3, 6:9] = 0.5 * dt * dt * identity
        transition[3:6, 6:9] = dt * identity
        return transition

    def process_noise(self, dt: float) -> np.ndarray:
        # Integrated white jerk, arranged as position/velocity/acceleration blocks.
        block = np.array(
            [
                [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
                [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
                [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
            ]
        ) * self.jerk_density
        noise = np.zeros((9, 9), dtype=float)
        for axis in range(3):
            indices = [axis, axis + 3, axis + 6]
            noise[np.ix_(indices, indices)] = block
        return noise

    def predict(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        dt = timestamp - self.timestamp
        if dt <= 0.0:
            raise ValueError("Kalman timestamps must be strictly increasing")
        transition = self.transition(dt)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + self.process_noise(dt)
        self.timestamp = timestamp

    def update(
        self,
        timestamp: float,
        position: Optional[np.ndarray],
        bypass_innovation_gate: bool = False,
    ) -> FilteredState:
        timestamp = float(timestamp)
        self.last_innovation_mahalanobis = float("nan")
        self.last_innovation_gate_bypassed = False
        if self.timestamp is None:
            if position is None:
                raise ValueError("The filter must be initialized by a valid position measurement")
            self.reset(position, timestamp)
            return self._result(True, 1.0)

        self.predict(timestamp)
        accepted = False
        confidence = 0.0
        if position is not None:
            measurement = np.asarray(position, dtype=float)
            if measurement.shape != (3,) or np.any(~np.isfinite(measurement)):
                return self._result(False, 0.0)
            observation = np.zeros((3, 9), dtype=float)
            observation[:, :3] = np.eye(3)
            covariance = np.eye(3) * self.measurement_variance
            innovation = measurement - observation @ self.state
            innovation_covariance = observation @ self.covariance @ observation.T + covariance
            mahalanobis = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
            self.last_innovation_mahalanobis = mahalanobis
            self.last_innovation_gate_bypassed = bool(
                bypass_innovation_gate and np.isfinite(mahalanobis)
            )
            # A newly initialized position-only filter has not yet observed
            # enough motion to make its velocity/acceleration prediction a
            # reliable outlier test. Callers may therefore accept a small,
            # configured number of valid startup measurements while still
            # applying the ordinary Kalman correction.
            if np.isfinite(mahalanobis) and (
                bool(bypass_innovation_gate) or mahalanobis <= self.gate
            ):
                gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
                self.state = self.state + gain @ innovation
                joseph_left = np.eye(9) - gain @ observation
                self.covariance = (
                    joseph_left @ self.covariance @ joseph_left.T + gain @ covariance @ gain.T
                )
                self.last_measurement_timestamp = timestamp
                accepted = True
                confidence = float(np.exp(-0.5 * mahalanobis / 3.0))

        if self.last_measurement_timestamp is not None:
            gap = timestamp - self.last_measurement_timestamp
            if gap > self.max_predict_gap:
                confidence = 0.0
            elif not accepted:
                confidence = max(0.0, 1.0 - gap / self.max_predict_gap)
        return self._result(accepted, confidence)

    def _result(self, accepted: bool, confidence: float) -> FilteredState:
        return FilteredState(
            timestamp=float(self.timestamp),
            position=self.state[:3].copy(),
            velocity=self.state[3:6].copy(),
            acceleration=self.state[6:9].copy(),
            covariance=self.covariance.copy(),
            measurement_accepted=accepted,
            confidence=float(confidence),
        )
