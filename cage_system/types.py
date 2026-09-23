from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PointObservation:
    """Timestamped target-center observation in the Base frame."""

    timestamp: float
    position: np.ndarray
    valid: bool = True
    source_index: int = -1


@dataclass
class FilteredState:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    covariance: np.ndarray
    measurement_accepted: bool
    confidence: float


@dataclass
class TrajectorySample:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    tangent: np.ndarray
    acceleration: np.ndarray
    curvature: float
    plane_normal: np.ndarray
    confidence: float = 1.0


@dataclass
class ObjectState:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass
class CandidatePose:
    candidate_id: str
    endpoint_position_base: np.ndarray
    endpoint_rotation_base: np.ndarray
    tcp_transform_base: np.ndarray
    arc_angle: float
    normal_branch: int
    ik_joints: Optional[np.ndarray] = None
    feasible: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)


@dataclass
class ESTMetrics:
    inside_nodes: int
    escaped_nodes: int
    illegal_nodes: int
    total_nodes: int
    occupancy_ratio: float
    always_retained_ratio: float
    terminal_retained_ratio: float
    opening_escape_ratio: float
    illegal_ratio: float
    nodes_base: np.ndarray
    node_status: np.ndarray
    node_parent_indices: np.ndarray
    node_times: np.ndarray
