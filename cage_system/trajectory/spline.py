from typing import Optional

import numpy as np
from scipy.interpolate import BSpline

from cage_system.geometry import normalize


class SharedCubicSpline3D:
    """Regularized 3-D cubic B-spline with shared time knots for x/y/z."""

    def __init__(self, control_spacing_s: float, regularization: float):
        self.control_spacing_s = float(control_spacing_s)
        self.regularization = float(regularization)
        self.spline: Optional[BSpline] = None
        self.start_time = 0.0
        self.end_time = 0.0

    @staticmethod
    def _basis_matrix(knots: np.ndarray, degree: int, times: np.ndarray) -> np.ndarray:
        basis_count = len(knots) - degree - 1
        return BSpline(knots, np.eye(basis_count), degree, extrapolate=True)(times)

    def fit(self, times: np.ndarray, positions: np.ndarray, weights=None):
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        if len(times) < 4 or positions.shape != (len(times), 3):
            raise ValueError("At least four timestamped 3-D positions are required")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("Spline timestamps must be strictly increasing")
        self.start_time, self.end_time = float(times[0]), float(times[-1])
        duration = self.end_time - self.start_time
        interior_count = max(0, int(np.ceil(duration / self.control_spacing_s)) - 1)
        max_interior = max(0, len(times) - 4)
        interior_count = min(interior_count, max_interior)
        interior = np.linspace(self.start_time, self.end_time, interior_count + 2)[1:-1]
        degree = 3
        knots = np.concatenate(
            [
                np.repeat(self.start_time, degree + 1),
                interior,
                np.repeat(self.end_time, degree + 1),
            ]
        )
        basis = self._basis_matrix(knots, degree, times)
        if weights is None:
            weights = np.ones(len(times), dtype=float)
        weights = np.asarray(weights, dtype=float)
        weighted_basis = basis * weights[:, None]
        basis_count = basis.shape[1]
        if basis_count >= 3:
            second_difference = np.diff(np.eye(basis_count), n=2, axis=0)
            penalty = second_difference.T @ second_difference
        else:
            penalty = np.zeros((basis_count, basis_count))
        lhs = basis.T @ weighted_basis + self.regularization * penalty
        rhs = basis.T @ (positions * weights[:, None])
        coefficients = np.linalg.solve(lhs + 1e-10 * np.eye(basis_count), rhs)
        self.spline = BSpline(knots, coefficients, degree, extrapolate=True)
        return self

    def evaluate(self, times, derivative: int = 0) -> np.ndarray:
        if self.spline is None:
            raise RuntimeError("Spline must be fitted before evaluation")
        evaluator = self.spline if derivative == 0 else self.spline.derivative(derivative)
        return np.asarray(evaluator(times), dtype=float)

    def sample(self, start: float, end: float, count: int) -> np.ndarray:
        return self.evaluate(np.linspace(float(start), float(end), int(count)))

    def kinematics(self, timestamp: float, low_speed_threshold: float = 1e-4):
        position = self.evaluate(timestamp, 0)
        velocity = self.evaluate(timestamp, 1)
        acceleration = self.evaluate(timestamp, 2)
        speed = float(np.linalg.norm(velocity))
        if speed > low_speed_threshold:
            tangent = velocity / speed
        else:
            delta = min(0.02, max(1e-3, 0.1 * (self.end_time - self.start_time)))
            before = max(self.start_time, timestamp - delta)
            after = min(self.end_time, timestamp + delta)
            tangent = normalize(self.evaluate(after) - self.evaluate(before), np.array([1.0, 0.0, 0.0]))
        cross = np.cross(velocity, acceleration)
        curvature = 0.0 if speed < low_speed_threshold else float(np.linalg.norm(cross) / speed ** 3)
        return position, velocity, speed, tangent, acceleration, curvature

