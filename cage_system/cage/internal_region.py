from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from cage_system.geometry import transform_points


@dataclass
class ContainmentResult:
    inside: bool
    escaped_opening: bool
    illegal_penetration: bool
    minimum_margin: float
    point_endpoint: np.ndarray


class CageInternalRegion:
    """True configurable U-channel free space expressed in endpoint frame E.

    +x points from the rear boundary toward the open mouth. The mouth is an
    escape surface, not a physical Cap. Side, vertical and rear limits model
    the gripper inner surfaces. The moving object is the agreed point mass.
    """

    def __init__(self, region_config: Dict, safety_gap_m: float):
        self.rear_x = float(region_config["rear_x"])
        self.mouth_x = float(region_config["mouth_x"])
        self.y_min = float(region_config["inner_y_min"])
        self.y_max = float(region_config["inner_y_max"])
        self.z_min = float(region_config["inner_z_min"])
        self.z_max = float(region_config["inner_z_max"])
        self.gap = float(safety_gap_m)
        if not (self.rear_x < self.mouth_x and self.y_min < self.y_max and self.z_min < self.z_max):
            raise ValueError("Invalid gripper internal region bounds")

    def feasible_radius_interval(self) -> Tuple[float, float]:
        return self.rear_x + self.gap, self.mouth_x - self.gap

    def evaluate(
        self,
        transform_base_endpoint: np.ndarray,
        object_position_base: np.ndarray,
    ) -> ContainmentResult:
        transform_endpoint_base = np.linalg.inv(transform_base_endpoint)
        point = transform_points(
            transform_endpoint_base, np.asarray(object_position_base, dtype=float)[None, :]
        )[0]
        margins = np.array(
            [
                point[0] - (self.rear_x + self.gap),
                (self.mouth_x - self.gap) - point[0],
                point[1] - (self.y_min + self.gap),
                (self.y_max - self.gap) - point[1],
                point[2] - (self.z_min + self.gap),
                (self.z_max - self.gap) - point[2],
            ]
        )
        minimum_margin = float(np.min(margins))
        opening_escape = bool(point[0] > self.mouth_x - self.gap)
        side_or_rear_violation = bool(
            point[0] < self.rear_x + self.gap
            or point[1] < self.y_min + self.gap
            or point[1] > self.y_max - self.gap
            or point[2] < self.z_min + self.gap
            or point[2] > self.z_max - self.gap
        )
        return ContainmentResult(
            inside=bool(minimum_margin >= 0.0),
            escaped_opening=opening_escape,
            illegal_penetration=side_or_rear_violation,
            minimum_margin=minimum_margin,
            point_endpoint=point,
        )
