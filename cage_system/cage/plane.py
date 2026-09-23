from dataclasses import dataclass
from typing import Optional

import numpy as np

from cage_system.geometry import normalize


@dataclass
class PlaneEstimate:
    origin: np.ndarray
    normal: np.ndarray
    is_near_line: bool
    eigenvalues: np.ndarray
    confidence: float


class LocalMotionPlaneEstimator:
    def __init__(self, config):
        self.curvature_threshold = float(config["curvature_threshold"])
        self.line_ratio_threshold = float(config["line_ratio_threshold"])
        self.update_alpha = float(config["normal_update_alpha"])
        self.maximum_step_rad = np.radians(float(config["maximum_normal_step_deg"]))
        self.maximum_reference_deviation_rad = np.radians(
            float(config["maximum_reference_deviation_deg"])
        )
        self.reference = normalize(np.asarray(config["base_reference_direction"], dtype=float))
        self.previous_normal: Optional[np.ndarray] = None
        self.track_reference_normal: Optional[np.ndarray] = None

    def reset(self):
        self.previous_normal = None
        self.track_reference_normal = None

    def set_track_reference(self, normal: np.ndarray):
        normal = normalize(np.asarray(normal, dtype=float))
        self.track_reference_normal = normal.copy()
        self.previous_normal = normal.copy()

    @staticmethod
    def _limit_direction(source: np.ndarray, target: np.ndarray, maximum_angle: float):
        cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
        angle = float(np.arccos(cosine))
        if angle <= maximum_angle or angle < 1e-9:
            return target
        fraction = maximum_angle / angle
        sine = np.sin(angle)
        if abs(sine) < 1e-8:
            return normalize((1.0 - fraction) * source + fraction * target)
        return normalize(
            np.sin((1.0 - fraction) * angle) / sine * source
            + np.sin(fraction * angle) / sine * target
        )

    def estimate(
        self,
        points: np.ndarray,
        origin: np.ndarray,
        tangent: np.ndarray,
        curvature: float,
        acceleration: Optional[np.ndarray] = None,
    ):
        points = np.asarray(points, dtype=float)
        tangent = normalize(tangent, np.array([1.0, 0.0, 0.0]))
        centered = points - np.mean(points, axis=0)
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        line_ratio = float(eigenvalues[1] / max(eigenvalues[2], 1e-12))
        derivative_normal = None
        if acceleration is not None and curvature >= self.curvature_threshold:
            tangent_cross_acceleration = np.cross(tangent, np.asarray(acceleration, dtype=float))
            if np.linalg.norm(tangent_cross_acceleration) > 1e-9:
                derivative_normal = normalize(tangent_cross_acceleration)
        pca_plane_available = line_ratio >= self.line_ratio_threshold
        near_line = derivative_normal is None and not pca_plane_available

        if derivative_normal is not None:
            proposed = derivative_normal
        elif pca_plane_available:
            proposed = normalize(eigenvectors[:, 0])
        else:
            if self.previous_normal is not None:
                proposed = self.previous_normal
            else:
                reference_projected = self.reference - np.dot(self.reference, tangent) * tangent
                if np.linalg.norm(reference_projected) < 1e-8:
                    fallback = np.eye(3)[int(np.argmin(np.abs(tangent)))]
                    reference_projected = fallback - np.dot(fallback, tangent) * tangent
                proposed = normalize(reference_projected)
        if self.previous_normal is None and self.track_reference_normal is None:
            reference_projected = self.reference - np.dot(self.reference, tangent) * tangent
            if np.linalg.norm(reference_projected) > 1e-8:
                reference_projected = normalize(reference_projected)
                if np.dot(proposed, reference_projected) < 0.0:
                    proposed = -proposed
        previous_projected = None
        if self.previous_normal is not None:
            previous_projected = self.previous_normal - np.dot(
                self.previous_normal, tangent
            ) * tangent
            previous_projected = normalize(previous_projected, self.previous_normal)
            if np.dot(proposed, previous_projected) < 0.0:
                proposed = -proposed
            proposed = normalize(
                (1.0 - self.update_alpha) * previous_projected
                + self.update_alpha * proposed
            )
        # The motion-plane normal must be perpendicular to the current tangent.
        projected = proposed - np.dot(proposed, tangent) * tangent
        if np.linalg.norm(projected) > 1e-9:
            proposed = normalize(projected)
        if self.track_reference_normal is not None:
            reference_projected = self.track_reference_normal - np.dot(
                self.track_reference_normal, tangent
            ) * tangent
            reference_projected = normalize(reference_projected, proposed)
            if np.dot(proposed, reference_projected) < 0.0:
                proposed = -proposed
            proposed = self._limit_direction(
                reference_projected,
                proposed,
                self.maximum_reference_deviation_rad,
            )
        if previous_projected is not None:
            if np.dot(proposed, previous_projected) < 0.0:
                proposed = -proposed
            proposed = self._limit_direction(
                previous_projected,
                proposed,
                self.maximum_step_rad,
            )
        self.previous_normal = proposed.copy()
        confidence = float(np.clip(1.0 - eigenvalues[0] / max(eigenvalues[1], 1e-12), 0.0, 1.0))
        return PlaneEstimate(np.asarray(origin).copy(), proposed, near_line, eigenvalues, confidence)
