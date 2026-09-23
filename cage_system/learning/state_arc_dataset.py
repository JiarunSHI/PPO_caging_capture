"""Geometry-only Cage supervision using a normalized minor-arc target."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


STATE_ARC_INPUT_DIM = 13
STATE_ARC_TARGET_DIM = 1
STATE_ARC_DATASET_SCHEMA_VERSION = 1
STATE_ARC_DATASET_CONTRACT = (
    "current_base_position_velocity_acceleration_plane_normal_signed_curvature_"
    "to_best_normalized_minor_arc_v1"
)


@dataclass
class CageStateArcDatasetArrays:
    """One deterministic state and the EST winner's relative arc coordinate."""

    inputs: np.ndarray
    targets: np.ndarray
    best_cage_poses: np.ndarray
    track_ids: np.ndarray
    label_timestamps: np.ndarray
    candidate_radii: np.ndarray
    arc_start_angles: np.ndarray
    arc_end_angles: np.ndarray
    best_est_means: np.ndarray
    best_est_stds: np.ndarray
    nominal_detection_period_s: Optional[float] = None

    def validate(self):
        count = len(self.inputs)
        if self.inputs.shape != (count, STATE_ARC_INPUT_DIM):
            raise ValueError("Expected state-arc inputs shaped [N,13]")
        if self.targets.shape != (count, STATE_ARC_TARGET_DIM):
            raise ValueError("Expected normalized arc targets shaped [N,1]")
        if self.best_cage_poses.shape != (count, 6):
            raise ValueError("Expected audited best Cage poses shaped [N,6]")
        one_dimensional = (
            self.track_ids,
            self.label_timestamps,
            self.candidate_radii,
            self.arc_start_angles,
            self.arc_end_angles,
            self.best_est_means,
            self.best_est_stds,
        )
        if any(np.asarray(item).shape != (count,) for item in one_dimensional):
            raise ValueError("State-arc dataset metadata length mismatch")
        if count:
            if not np.all(np.isfinite(self.inputs)):
                raise ValueError("State-arc inputs must be finite")
            if not np.all(np.isfinite(self.targets)):
                raise ValueError("State-arc targets must be finite")
            if np.any(self.targets < -1e-9) or np.any(self.targets > 1.0 + 1e-9):
                raise ValueError("Normalized arc targets must be inside [0,1]")
            if np.any(np.abs(self.arc_end_angles - self.arc_start_angles) < 1e-8):
                raise ValueError("Degenerate arcs cannot be state-arc supervision")
        if self.nominal_detection_period_s is None or not np.isfinite(
            float(self.nominal_detection_period_s)
        ) or float(self.nominal_detection_period_s) <= 0.0:
            raise ValueError("State-arc dataset requires a positive nominal period")
        return self

    @property
    def labels(self):
        return self.targets

    def require_detection_period(self, expected_period_s: float):
        expected = float(expected_period_s)
        actual = float(self.nominal_detection_period_s)
        if not np.isclose(actual, expected, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "State-arc dataset uses a {:.6f} s detection period, expected "
                "{:.6f} s".format(actual, expected)
            )
        return self

    def save(self, path: str):
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            inputs=self.inputs,
            normalized_arc_targets=self.targets,
            best_cage_poses=self.best_cage_poses,
            track_ids=self.track_ids,
            label_timestamps=self.label_timestamps,
            candidate_radii=self.candidate_radii,
            arc_start_angles=self.arc_start_angles,
            arc_end_angles=self.arc_end_angles,
            best_est_means=self.best_est_means,
            best_est_stds=self.best_est_stds,
            input_semantics=np.array(
                "current_t_base_position3_velocity3_acceleration3_plane_normal3_signed_curvature1"
            ),
            target_semantics=np.array(
                "best_est_candidate_signed_minor_arc_fraction_tangent_zero_future_one"
            ),
            dataset_contract=np.array(STATE_ARC_DATASET_CONTRACT),
            schema_version=np.array(STATE_ARC_DATASET_SCHEMA_VERSION, dtype=int),
            object_model=np.array("point_mass"),
            input_source=np.array("clean_true_trajectory_spline_kinematics"),
            nominal_detection_period_s=np.array(
                float(self.nominal_detection_period_s), dtype=float
            ),
        )

    @classmethod
    def load(cls, path: str):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive.get("dataset_contract", "")) != STATE_ARC_DATASET_CONTRACT:
                raise ValueError("Unsupported state-arc Cage dataset contract")
            if int(archive["schema_version"]) != STATE_ARC_DATASET_SCHEMA_VERSION:
                raise ValueError("Unsupported state-arc Cage dataset schema")
            arrays = cls(
                inputs=archive["inputs"].copy(),
                targets=archive["normalized_arc_targets"].copy(),
                best_cage_poses=archive["best_cage_poses"].copy(),
                track_ids=archive["track_ids"].copy(),
                label_timestamps=archive["label_timestamps"].copy(),
                candidate_radii=archive["candidate_radii"].copy(),
                arc_start_angles=archive["arc_start_angles"].copy(),
                arc_end_angles=archive["arc_end_angles"].copy(),
                best_est_means=archive["best_est_means"].copy(),
                best_est_stds=archive["best_est_stds"].copy(),
                nominal_detection_period_s=float(archive["nominal_detection_period_s"]),
            )
        return arrays.validate()

    @classmethod
    def empty(cls, nominal_detection_period_s: float):
        return cls(
            np.empty((0, STATE_ARC_INPUT_DIM), dtype=float),
            np.empty((0, STATE_ARC_TARGET_DIM), dtype=float),
            np.empty((0, 6), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            float(nominal_detection_period_s),
        ).validate()

    @classmethod
    def concatenate(cls, arrays, nominal_detection_period_s: float):
        nonempty = [item for item in arrays if len(item.inputs)]
        if not nonempty:
            return cls.empty(nominal_detection_period_s)
        return cls(
            np.concatenate([item.inputs for item in nonempty]),
            np.concatenate([item.targets for item in nonempty]),
            np.concatenate([item.best_cage_poses for item in nonempty]),
            np.concatenate([item.track_ids for item in nonempty]),
            np.concatenate([item.label_timestamps for item in nonempty]),
            np.concatenate([item.candidate_radii for item in nonempty]),
            np.concatenate([item.arc_start_angles for item in nonempty]),
            np.concatenate([item.arc_end_angles for item in nonempty]),
            np.concatenate([item.best_est_means for item in nonempty]),
            np.concatenate([item.best_est_stds for item in nonempty]),
            float(nominal_detection_period_s),
        ).validate()


def load_generated_label_arrays(path: str):
    """Load either the legacy visual-history dataset or the new state dataset."""

    with np.load(path, allow_pickle=False) as archive:
        contract = str(archive.get("dataset_contract", ""))
    if contract == STATE_ARC_DATASET_CONTRACT:
        return CageStateArcDatasetArrays.load(path)
    from cage_system.learning.dataset import CageDatasetArrays

    return CageDatasetArrays.load(path)
