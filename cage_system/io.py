from pathlib import Path
from typing import List

import numpy as np

from cage_system.trajectory.generator import GeneratedTrajectory, trajectory_diagnostics
from cage_system.types import PointObservation


TRAJECTORY_ARCHIVE_SCHEMA_VERSION = 5


def validate_trajectory_detection_period(
    trajectories: List[GeneratedTrajectory], expected_period_s: float
) -> None:
    """Reject legacy or mismatched trajectory clocks before label generation."""

    expected = float(expected_period_s)
    if not np.isfinite(expected) or expected <= 0.0:
        raise ValueError("Expected detection period must be positive and finite")
    for trajectory in trajectories:
        declared = trajectory.nominal_detection_period_s
        if declared is None:
            raise ValueError(
                "Trajectory archive has no nominal_detection_period_s; "
                "regenerate it with the configured visual clock"
            )
        declared = float(declared)
        if not np.isclose(declared, expected, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "Trajectory {} uses a {:.6f} s detection period, expected "
                "{:.6f} s; do not mix old visual-clock archives".format(
                    int(trajectory.track_id), declared, expected
                )
            )
        timestamps = np.asarray(
            [item.timestamp for item in trajectory.observations], dtype=float
        )
        if len(timestamps) >= 2:
            intervals = np.diff(timestamps)
            if np.any(~np.isfinite(intervals)) or np.any(intervals <= 0.0):
                raise ValueError(
                    "Trajectory {} has invalid observation timestamps".format(
                        int(trajectory.track_id)
                    )
                )
            # Generated timestamps intentionally carry small jitter.  A broad
            # relative check preserves that augmentation while separating the
            # old 50 ms and new 100 ms archives unambiguously.
            median_interval = float(np.median(intervals))
            if not np.isclose(median_interval, expected, rtol=0.20, atol=1e-9):
                raise ValueError(
                    "Trajectory {} observations have median interval {:.6f} s, "
                    "expected approximately {:.6f} s".format(
                        int(trajectory.track_id), median_interval, expected
                    )
                )


def save_generated_trajectories(trajectories: List[GeneratedTrajectory], path: str):
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    periods = [item.nominal_detection_period_s for item in trajectories]
    if any(item is None for item in periods):
        raise ValueError(
            "Every trajectory must declare nominal_detection_period_s before saving"
        )
    nominal_period = float(periods[0])
    if any(
        not np.isclose(float(item), nominal_period, rtol=1e-6, atol=1e-9)
        for item in periods[1:]
    ):
        raise ValueError("A trajectory archive cannot mix detection periods")
    validate_trajectory_detection_period(trajectories, nominal_period)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    true_offsets = [0]
    observation_offsets = [0]
    true_times, true_positions = [], []
    observation_times, observation_positions, observation_sources, observation_valid = [], [], [], []
    for trajectory in trajectories:
        true_times.append(trajectory.timestamps)
        true_positions.append(trajectory.true_positions)
        true_offsets.append(true_offsets[-1] + len(trajectory.timestamps))
        observation_times.append(np.array([item.timestamp for item in trajectory.observations]))
        observation_positions.append(np.array([item.position for item in trajectory.observations]))
        observation_sources.append(np.array([item.source_index for item in trajectory.observations]))
        observation_valid.append(np.array([item.valid for item in trajectory.observations], dtype=bool))
        observation_offsets.append(observation_offsets[-1] + len(trajectory.observations))
    diagnostics = [
        trajectory_diagnostics(item.timestamps, item.true_positions)
        for item in trajectories
    ]
    np.savez_compressed(
        destination,
        track_ids=np.array([item.track_id for item in trajectories], dtype=int),
        families=np.array([item.family for item in trajectories]),
        plane_normals=np.array([item.plane_normal for item in trajectories]),
        true_offsets=np.array(true_offsets, dtype=int),
        true_times=np.concatenate(true_times),
        true_positions=np.concatenate(true_positions),
        observation_offsets=np.array(observation_offsets, dtype=int),
        observation_times=np.concatenate(observation_times),
        observation_positions=np.concatenate(observation_positions),
        observation_sources=np.concatenate(observation_sources),
        observation_valid=np.concatenate(observation_valid),
        curvature_min_m_inv=np.array(
            [item.curvature_min_m_inv for item in diagnostics], dtype=float
        ),
        curvature_max_m_inv=np.array(
            [item.curvature_max_m_inv for item in diagnostics], dtype=float
        ),
        five_frame_curvature_range_min_m_inv=np.array(
            [item.five_frame_curvature_range_min_m_inv for item in diagnostics],
            dtype=float,
        ),
        five_frame_curvature_range_max_m_inv=np.array(
            [item.five_frame_curvature_range_max_m_inv for item in diagnostics],
            dtype=float,
        ),
        local_plane_total_variation_deg=np.array(
            [item.local_plane_total_variation_deg for item in diagnostics], dtype=float
        ),
        local_plane_max_step_deg=np.array(
            [item.local_plane_max_step_deg for item in diagnostics], dtype=float
        ),
        five_frame_turn_max_deg=np.array(
            [item.five_frame_turn_max_deg for item in diagnostics], dtype=float
        ),
        schema_version=np.array(TRAJECTORY_ARCHIVE_SCHEMA_VERSION, dtype=int),
        object_model=np.array("point_mass"),
        nominal_detection_period_s=np.array(nominal_period, dtype=float),
    )


def load_generated_trajectories(path: str) -> List[GeneratedTrajectory]:
    # NpzFile is lazy: every ``archive[key]`` access decompresses that member.
    # Cache each required member once before iterating over tracks; otherwise a
    # large archive repeatedly decompresses offsets and observation arrays for
    # every trajectory and label generation can spend minutes before it starts.
    with np.load(path, allow_pickle=False) as archive:
        schema_version = (
            int(archive["schema_version"]) if "schema_version" in archive else 1
        )
        object_model = (
            str(archive["object_model"])
            if "object_model" in archive
            else "legacy_pose"
        )
        if schema_version not in (2, 3, 4, TRAJECTORY_ARCHIVE_SCHEMA_VERSION) or object_model != "point_mass":
            raise ValueError(
                "Unsupported trajectory archive: expected schema_version=2/3/4/5 "
                "and object_model=point_mass"
            )
        nominal_detection_period_s = (
            float(archive["nominal_detection_period_s"])
            if "nominal_detection_period_s" in archive
            else None
        )
        if (
            schema_version == TRAJECTORY_ARCHIVE_SCHEMA_VERSION
            and nominal_detection_period_s is None
        ):
            raise ValueError(
                "Trajectory schema 5 requires nominal_detection_period_s"
            )
        track_ids = archive["track_ids"]
        families = archive["families"]
        plane_normals = archive["plane_normals"]
        true_offsets = archive["true_offsets"]
        true_times = archive["true_times"]
        true_positions = archive["true_positions"]
        observation_offsets = archive["observation_offsets"]
        observation_times = archive["observation_times"]
        observation_positions = archive["observation_positions"]
        observation_sources = archive["observation_sources"]
        observation_valid = (
            archive["observation_valid"]
            if "observation_valid" in archive
            else np.ones(len(observation_times), dtype=bool)
        )

    trajectories = []
    for index, track_id in enumerate(track_ids):
        true_slice = slice(true_offsets[index], true_offsets[index + 1])
        observation_slice = slice(
            observation_offsets[index], observation_offsets[index + 1]
        )
        observations = [
            PointObservation(float(timestamp), position.copy(), bool(valid), int(source))
            for timestamp, position, source, valid in zip(
                observation_times[observation_slice],
                observation_positions[observation_slice],
                observation_sources[observation_slice],
                observation_valid[observation_slice],
            )
        ]
        trajectories.append(
            GeneratedTrajectory(
                int(track_id),
                str(families[index]),
                true_times[true_slice].copy(),
                true_positions[true_slice].copy(),
                observations,
                plane_normals[index].copy(),
                nominal_detection_period_s,
            )
        )
    return trajectories
