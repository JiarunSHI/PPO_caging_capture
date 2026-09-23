from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from cage_system.geometry import make_transform, normalize, orthonormal_frame, pose6_to_transform, transform_to_pose6
from cage_system.learning.history_quality import HISTORY_QUALITY_CONTRACT


HISTORY_LENGTH = 5
POSITION_DIM = 3
SPEED_DIM = 1
FRAME_DIM = POSITION_DIM + SPEED_DIM
CAGE_INPUT_DIM = HISTORY_LENGTH * FRAME_DIM
CAGE_LABEL_DIM = 6
CAGE_GEOMETRY_CONTRACT = (
    "single_positive_normal_strict_three_est_mean_analytic_multistart_ik_v6"
)
CAGE_DATASET_SCHEMA_VERSION = 10
HISTORY_CANONICALIZATION_CONTRACT = (
    "last_history_origin_negative_history_tangent_positive_pca_normal_v1"
)


def make_cage_input(history_frames) -> np.ndarray:
    if len(history_frames) != HISTORY_LENGTH:
        raise ValueError("Cage input must contain t-5..t-1 exactly; current t is not an input")
    matrix = np.stack(
        [np.concatenate([frame.position, np.array([frame.speed])]) for frame in history_frames]
    )
    if matrix.shape != (HISTORY_LENGTH, FRAME_DIM):
        raise ValueError("Each history frame must be position_xyz + scalar_linear_speed")
    return matrix


def history_reference_transform(history_matrix: np.ndarray, base_reference) -> np.ndarray:
    """Build a causal Base-to-history reference using only the five inputs."""

    history = np.asarray(history_matrix, dtype=float)
    if history.shape != (HISTORY_LENGTH, FRAME_DIM):
        raise ValueError("History reference requires one [5,4] input")
    positions = history[:, :3]
    tangent = normalize(positions[-1] - positions[0], np.array([1.0, 0.0, 0.0]))
    centered = positions - np.mean(positions, axis=0)
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    reference = normalize(np.asarray(base_reference, dtype=float))
    if abs(float(np.dot(normal, tangent))) > 0.2:
        normal = reference - np.dot(reference, tangent) * tangent
        if np.linalg.norm(normal) < 1e-8:
            axis = np.eye(3)[int(np.argmin(np.abs(tangent)))]
            normal = axis - np.dot(axis, tangent) * tangent
    normal = normalize(normal)
    if np.dot(normal, reference) < 0.0:
        normal = -normal
    # Candidate Cage x points from its endpoint back to the object, hence the
    # negative historical tangent is the useful causal orientation reference.
    rotation = orthonormal_frame(-tangent, normal)
    return make_transform(positions[-1], rotation)


def canonicalize_cage_batch(inputs, labels, base_reference):
    """Remove global translation/orientation nuisance without future data."""

    inputs = np.asarray(inputs, dtype=float)
    labels = None if labels is None else np.asarray(labels, dtype=float)
    canonical_inputs = np.empty_like(inputs)
    canonical_labels = None if labels is None else np.empty_like(labels)
    references = np.empty((len(inputs), 4, 4), dtype=float)
    for index, history in enumerate(inputs):
        reference = history_reference_transform(history, base_reference)
        references[index] = reference
        canonical_inputs[index] = history
        canonical_inputs[index, :, :3] = (
            history[:, :3] - reference[:3, 3]
        ) @ reference[:3, :3]
        if labels is not None:
            relative = np.linalg.inv(reference) @ pose6_to_transform(labels[index])
            canonical_labels[index] = transform_to_pose6(relative)
    return canonical_inputs, canonical_labels, references


@dataclass
class CageDatasetArrays:
    inputs: np.ndarray
    labels: np.ndarray
    track_ids: np.ndarray
    label_timestamps: np.ndarray
    nominal_detection_period_s: Optional[float] = None
    input_source: str = "causal_point_estimator"
    geometry_contract: str = CAGE_GEOMETRY_CONTRACT

    def validate(self):
        if self.inputs.ndim != 3 or self.inputs.shape[1:] != (HISTORY_LENGTH, FRAME_DIM):
            raise ValueError("Expected Cage inputs shaped [N, 5, 4]")
        if self.labels.shape != (len(self.inputs), CAGE_LABEL_DIM):
            raise ValueError("Expected Cage labels shaped [N, 6]")
        if len(self.track_ids) != len(self.inputs) or len(self.label_timestamps) != len(self.inputs):
            raise ValueError("Dataset metadata length mismatch")
        if self.geometry_contract != CAGE_GEOMETRY_CONTRACT:
            raise ValueError("Cage dataset uses an obsolete gripper/TCP geometry contract")
        if self.nominal_detection_period_s is not None and (
            not np.isfinite(float(self.nominal_detection_period_s))
            or float(self.nominal_detection_period_s) <= 0.0
        ):
            raise ValueError("Cage dataset detection period must be positive and finite")
        return self

    def require_detection_period(self, expected_period_s: float):
        expected = float(expected_period_s)
        if self.nominal_detection_period_s is None:
            raise ValueError(
                "Cage dataset has no nominal_detection_period_s; regenerate it "
                "instead of mixing an old visual-clock dataset"
            )
        actual = float(self.nominal_detection_period_s)
        if not np.isclose(actual, expected, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "Cage dataset uses a {:.6f} s detection period, expected "
                "{:.6f} s".format(actual, expected)
            )
        return self

    def save(self, path: str):
        if self.nominal_detection_period_s is None:
            raise ValueError(
                "Refusing to save a Cage dataset without nominal_detection_period_s"
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            inputs=self.inputs,
            labels=self.labels,
            track_ids=self.track_ids,
            label_timestamps=self.label_timestamps,
            input_semantics=np.array("rows=t-5..t-1; columns=position_xyz+linear_speed_magnitude"),
            label_semantics=np.array("current_t_tcp_xyz_rotvec_base"),
            object_model=np.array("point_mass"),
            schema_version=np.array(CAGE_DATASET_SCHEMA_VERSION, dtype=int),
            input_source=np.array(self.input_source),
            geometry_contract=np.array(self.geometry_contract),
            history_quality_contract=np.array(HISTORY_QUALITY_CONTRACT),
            nominal_detection_period_s=np.array(
                float(self.nominal_detection_period_s), dtype=float
            ),
        )

    @classmethod
    def load(cls, path: str, allow_unfiltered_legacy: bool = False):
        archive = np.load(path, allow_pickle=False)
        schema_version = int(archive["schema_version"]) if "schema_version" in archive else 1
        object_model = str(archive["object_model"]) if "object_model" in archive else "legacy_pose"
        geometry_contract = (
            str(archive["geometry_contract"])
            if "geometry_contract" in archive
            else "legacy_large_cage"
        )
        history_quality_contract = (
            str(archive["history_quality_contract"])
            if "history_quality_contract" in archive
            else "legacy_unfiltered_history"
        )
        legacy_allowed = bool(
            allow_unfiltered_legacy
            and schema_version == 6
            and history_quality_contract == "legacy_unfiltered_history"
        )
        if (
            not (schema_version == CAGE_DATASET_SCHEMA_VERSION or legacy_allowed)
            or object_model != "point_mass"
            or geometry_contract != CAGE_GEOMETRY_CONTRACT
            or (
                not legacy_allowed
                and history_quality_contract != HISTORY_QUALITY_CONTRACT
            )
        ):
            raise ValueError(
                "Unsupported Cage dataset: expected the filtered five-frame history quality contract"
            )
        input_source = (
            str(archive["input_source"])
            if "input_source" in archive
            else "causal_point_estimator"
        )
        nominal_detection_period_s = (
            float(archive["nominal_detection_period_s"])
            if "nominal_detection_period_s" in archive
            else None
        )
        if (
            schema_version == CAGE_DATASET_SCHEMA_VERSION
            and nominal_detection_period_s is None
        ):
            raise ValueError(
                "Cage dataset schema 10 requires nominal_detection_period_s"
            )
        return cls(
            archive["inputs"],
            archive["labels"],
            archive["track_ids"],
            archive["label_timestamps"],
            nominal_detection_period_s,
            input_source,
            geometry_contract,
        ).validate()


def split_by_track(track_ids: np.ndarray, validation_fraction: float, seed: int):
    unique = np.unique(track_ids)
    rng = np.random.RandomState(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(unique) * validation_fraction)))
    validation_tracks = set(shuffled[:validation_count].tolist())
    validation_mask = np.array([item in validation_tracks for item in track_ids])
    if np.all(validation_mask) or not np.any(validation_mask):
        raise ValueError("At least two trajectories are required for a leakage-free train/validation split")
    return ~validation_mask, validation_mask


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, inputs: np.ndarray):
        flattened = inputs.reshape(len(inputs), -1)
        mean = flattened.mean(axis=0)
        std = flattened.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean, std)

    def transform(self, inputs: np.ndarray):
        flattened = inputs.reshape(len(inputs), -1)
        return (flattened - self.mean) / self.std
