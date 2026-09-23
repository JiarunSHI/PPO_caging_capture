from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from cage_system.geometry import make_transform, normalize, orthonormal_frame, plane_basis, rotation_as_matrix
from cage_system.types import CandidatePose


@dataclass
class CandidateArc:
    center: np.ndarray
    radius: float
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    tangent_intersection: np.ndarray
    future_intersection: np.ndarray
    angle_start: float
    angle_end: float
    degenerate: bool
    clipped: bool


class SpeedRadiusMapper:
    def __init__(self, config):
        self.config = config
        self.previous_radius: Optional[float] = None
        self.previous_timestamp: Optional[float] = None

    def reset(self):
        self.previous_radius = None
        self.previous_timestamp = None

    def map(self, speed: float, timestamp: float) -> Tuple[float, dict]:
        lower = float(self.config["radius_min_m"])
        upper = float(self.config["radius_max_m"])
        if lower > upper:
            raise ValueError("Invalid follow-standoff radius interval")
        speed_lower, speed_upper = [
            float(item) for item in self.config["speed_radius_speed_bounds_mps"]
        ]
        fraction = (max(float(speed), 0.0) - speed_lower) / (speed_upper - speed_lower)
        raw = lower + fraction * (upper - lower)
        clipped = float(np.clip(raw, lower, upper))
        if self.previous_radius is None:
            radius = clipped
        else:
            alpha = float(self.config["radius_lowpass_alpha"])
            lowpass = alpha * self.previous_radius + (1.0 - alpha) * clipped
            dt = max(float(timestamp) - float(self.previous_timestamp), 1e-6)
            maximum_change = float(self.config["radius_max_rate_mps"]) * dt
            radius = float(
                np.clip(lowpass, self.previous_radius - maximum_change, self.previous_radius + maximum_change)
            )
            radius = float(np.clip(radius, lower, upper))
        self.previous_radius = radius
        self.previous_timestamp = float(timestamp)
        return radius, {
            "raw_radius": raw,
            "lower_bound": lower,
            "upper_bound": upper,
            "clipped_radius": clipped,
            "smoothed_radius": radius,
            "speed_bounds_mps": [speed_lower, speed_upper],
            "mapping": "linear_speed_to_candidate_radius_then_clip_and_smooth",
        }


class CandidateGenerator:
    def __init__(self, config, transform_tcp_endpoint: np.ndarray):
        self.config = config
        self.transform_tcp_endpoint = np.asarray(transform_tcp_endpoint, dtype=float)

    def construct_arc(
        self,
        current_position: np.ndarray,
        tangent: np.ndarray,
        normal: np.ndarray,
        radius: float,
        current_time: float,
        future_end_time: float,
        curve_query: Callable[[float], np.ndarray],
    ) -> CandidateArc:
        center = np.asarray(current_position, dtype=float)
        u_axis, v_axis = plane_basis(normal, tangent)
        tangent_intersection = center + radius * u_axis

        sample_count = int(self.config["intersection_samples"])
        times = np.linspace(float(current_time), float(future_end_time), sample_count)

        def signed_distance(timestamp):
            displacement = np.asarray(curve_query(float(timestamp))) - center
            in_plane = displacement - np.dot(displacement, normal) * normal
            return float(np.linalg.norm(in_plane) - radius)

        values = np.array([signed_distance(item) for item in times])
        crossing = None
        for index in range(1, len(times)):
            if values[index - 1] <= 0.0 and values[index] >= 0.0:
                crossing = (times[index - 1], times[index])
                break
        if crossing is None:
            raise ValueError("future_curve_does_not_reach_candidate_circle")
        intersection_time = brentq(signed_distance, crossing[0], crossing[1], xtol=1e-10)
        displacement = np.asarray(curve_query(intersection_time)) - center
        displacement -= np.dot(displacement, normal) * normal
        future_intersection = center + radius * normalize(displacement)
        future_direction = normalize(future_intersection - center)
        future_angle = float(
            np.arctan2(np.dot(future_direction, v_axis), np.dot(future_direction, u_axis))
        )
        future_angle = float((future_angle + np.pi) % (2.0 * np.pi) - np.pi)
        # The candidate arc is always the exact minor arc between the forward
        # tangent intersection and the first future-curve intersection.  Near a
        # straight line it is allowed to collapse to one pose; widening it or
        # clipping a high-curvature arc would move one or both endpoints away
        # from the two agreed geometric intersections.
        degenerate = abs(future_angle) < 1e-8
        clipped = False
        angle_start, angle_end = 0.0, future_angle
        return CandidateArc(
            center=center,
            radius=float(radius),
            normal=normalize(normal),
            u_axis=u_axis,
            v_axis=v_axis,
            tangent_intersection=tangent_intersection,
            future_intersection=future_intersection,
            angle_start=angle_start,
            angle_end=angle_end,
            degenerate=degenerate,
            clipped=clipped,
        )

    def sample_arc(self, arc: CandidateArc, previous_best_angle: Optional[float] = None) -> np.ndarray:
        count = int(self.config["nominal_samples_per_arc"])
        angles = list(np.linspace(arc.angle_start, arc.angle_end, count))
        if previous_best_angle is not None:
            dense_count = int(self.config["previous_best_dense_samples"])
            width = np.deg2rad(float(self.config["previous_best_dense_width_deg"]))
            dense_center = float(np.clip(previous_best_angle, min(arc.angle_start, arc.angle_end), max(arc.angle_start, arc.angle_end)))
            angles.extend(np.linspace(dense_center - width, dense_center + width, dense_count))
        lower, upper = sorted((arc.angle_start, arc.angle_end))
        angles = sorted(float(np.clip(item, lower, upper)) for item in angles)
        tolerance = np.deg2rad(float(self.config["duplicate_angle_tolerance_deg"]))
        # Both geometric boundaries are mandatory. Deduplicate only interior
        # samples so dense sampling near the previous optimum cannot replace
        # the exact tangent or first-future-curve intersection.
        if abs(upper - lower) < 1e-12:
            return np.asarray([float(arc.angle_start)])
        unique = [lower]
        for angle in angles:
            if angle - lower < tolerance or upper - angle < tolerance:
                continue
            if abs(angle - unique[-1]) >= tolerance:
                unique.append(angle)
        unique.append(upper)
        return np.asarray(unique)

    def build_poses(self, arc: CandidateArc, angles: np.ndarray) -> List[CandidatePose]:
        candidates = []
        inverse_tcp_endpoint = np.linalg.inv(self.transform_tcp_endpoint)
        branches = [int(item) for item in self.config["orientation_branches"]]
        if branches != [1]:
            raise ValueError(
                "Candidate poses must use only the stabilized positive "
                "motion-plane normal branch"
            )
        for angle_index, angle in enumerate(angles):
            direction = np.cos(angle) * arc.u_axis + np.sin(angle) * arc.v_axis
            endpoint = arc.center + arc.radius * normalize(direction)
            # Exact projection back to both the plane and circle.
            endpoint = arc.center + arc.radius * normalize(
                endpoint - arc.center - np.dot(endpoint - arc.center, arc.normal) * arc.normal
            )
            x_axis = normalize(arc.center - endpoint)
            for branch in branches:
                rotation = orthonormal_frame(x_axis, branch * arc.normal)
                transform_base_endpoint = make_transform(endpoint, rotation)
                transform_base_tcp = transform_base_endpoint @ inverse_tcp_endpoint
                candidates.append(
                    CandidatePose(
                        candidate_id="a{:03d}_z{:+d}".format(angle_index, branch),
                        endpoint_position_base=endpoint,
                        endpoint_rotation_base=rotation,
                        tcp_transform_base=transform_base_tcp,
                        arc_angle=float(angle),
                        normal_branch=branch,
                    )
                )
        return candidates


def transform_from_config(config) -> np.ndarray:
    position = np.asarray(config["translation"], dtype=float)
    rotation = rotation_as_matrix(Rotation.from_rotvec(np.asarray(config["rotation_vector"], dtype=float)))
    return make_transform(position, rotation)
