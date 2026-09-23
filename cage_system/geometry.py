from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


EPS = 1e-10


def rotation_from_matrix(matrix: np.ndarray) -> Rotation:
    constructor = getattr(Rotation, "from_matrix", None)
    return constructor(matrix) if constructor is not None else Rotation.from_dcm(matrix)


def rotation_as_matrix(rotation: Rotation) -> np.ndarray:
    converter = getattr(rotation, "as_matrix", None)
    return converter() if converter is not None else rotation.as_dcm()


def normalize(vector: np.ndarray, fallback: np.ndarray = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > EPS:
        return vector / norm
    if fallback is None:
        raise ValueError("Cannot normalize a near-zero vector")
    return normalize(np.asarray(fallback, dtype=float))


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def make_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ transform[:3, :3].T + transform[:3, 3]


def pose6_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return make_transform(pose[:3], rotation_as_matrix(Rotation.from_rotvec(pose[3:])))


def transform_to_pose6(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [transform[:3, 3], rotation_from_matrix(transform[:3, :3]).as_rotvec()]
    )


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    return rotation_from_matrix(matrix).as_quat()


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    return float(np.linalg.norm(rotation_from_matrix(relative).as_rotvec()))


def orthonormal_frame(x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    z_axis = normalize(z_axis)
    x_projected = np.asarray(x_axis, dtype=float) - np.dot(x_axis, z_axis) * z_axis
    x_axis = normalize(x_projected)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(rotation) < 0.0:
        raise RuntimeError("Constructed frame is not right handed")
    return rotation


def plane_basis(normal: np.ndarray, preferred_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = normalize(normal)
    projected = preferred_u - np.dot(preferred_u, normal) * normal
    if np.linalg.norm(projected) < EPS:
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        projected = axis - np.dot(axis, normal) * normal
    u_axis = normalize(projected)
    v_axis = normalize(np.cross(normal, u_axis))
    return u_axis, v_axis
