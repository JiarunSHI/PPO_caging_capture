from dataclasses import dataclass
from typing import Sequence

import numpy as np


HISTORY_QUALITY_CONTRACT = (
    "five_previous_accepted_frames_curvature_turn_plane_gate_v3"
)


@dataclass(frozen=True)
class HistoryQualityResult:
    valid: bool
    reason: str
    missed_detections: int
    maximum_interval_s: float
    curvature_m_inv: float
    five_frame_turn_deg: float
    five_frame_plane_ratio: float = float("nan")


def maximum_history_interval_s(timing_config, quality_config) -> float:
    return float(quality_config["maximum_interval_factor"]) / float(
        timing_config["detection_rate_hz"]
    )


def estimate_history_curvature_m_inv(
    history_frames: Sequence,
    minimum_speed_mps: float,
) -> float:
    """Read causal rolling-spline curvature, with a position-only fallback.

    Production frames carry a curvature estimate made when each observation
    arrived, using only the estimator's past rolling window.  The quadratic
    fallback exists for simple external frame objects and tests.
    """

    last_frame = history_frames[-1] if history_frames else None
    if last_frame is not None and hasattr(last_frame, "curvature"):
        if float(last_frame.speed) < float(minimum_speed_mps):
            return 0.0
        curvature = float(last_frame.curvature)
        return curvature if np.isfinite(curvature) else float("inf")

    # Compatibility fallback for external callers that provide only timestamp
    # and position fields. Production CausalHistoryFrame objects carry the
    # curvature of the causal rolling spline computed at frame arrival time.
    timestamps = np.asarray([frame.timestamp for frame in history_frames], dtype=float)
    positions = np.asarray([frame.position for frame in history_frames], dtype=float)
    if len(timestamps) < 3 or positions.shape != (len(timestamps), 3):
        return float("inf")
    relative_time = timestamps - timestamps[-1]
    design = np.column_stack(
        (np.ones_like(relative_time), relative_time, relative_time**2)
    )
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    velocity = coefficients[1]
    acceleration = 2.0 * coefficients[2]
    speed = float(np.linalg.norm(velocity))
    if speed < float(minimum_speed_mps):
        return 0.0
    curvature = float(np.linalg.norm(np.cross(velocity, acceleration)) / speed**3)
    return curvature if np.isfinite(curvature) else float("inf")


def estimate_five_frame_turn_deg(history_frames: Sequence) -> float:
    if len(history_frames) < 2:
        return float("inf")
    if all(hasattr(frame, "tangent") for frame in history_frames):
        tangents = np.asarray([frame.tangent for frame in history_frames], dtype=float)
    else:
        positions = np.asarray([frame.position for frame in history_frames], dtype=float)
        if positions.shape != (len(history_frames), 3):
            return float("inf")
        tangents = np.gradient(positions, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    valid = norms > 1e-8
    if np.sum(valid) < 2:
        return 0.0
    tangents = tangents[valid] / norms[valid, None]
    dot = float(np.clip(np.dot(tangents[0], tangents[-1]), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def estimate_five_frame_plane_ratio(history_frames: Sequence) -> float:
    """Return the middle/largest PCA variance ratio of the five positions."""

    positions = np.asarray([frame.position for frame in history_frames], dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 3:
        return 0.0
    centered = positions - np.mean(positions, axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
    return float(eigenvalues[1] / max(eigenvalues[2], 1e-12))


def evaluate_history_quality(
    history_frames: Sequence,
    current_timestamp: float,
    timing_config,
    quality_config,
    enforce_maximum_interval: bool = True,
) -> HistoryQualityResult:
    expected_length = int(timing_config["history_length"])
    if len(history_frames) != expected_length:
        return HistoryQualityResult(
            False,
            "insufficient_five_frame_history",
            expected_length - len(history_frames),
            float("inf"),
            float("inf"),
            float("inf"),
        )

    timestamps = np.asarray(
        [frame.timestamp for frame in history_frames] + [float(current_timestamp)],
        dtype=float,
    )
    if not np.all(np.isfinite(timestamps)):
        return HistoryQualityResult(
            False,
            "nonfinite_history_timestamp",
            0,
            float("inf"),
            float("inf"),
            float("inf"),
        )
    intervals = np.diff(timestamps)
    maximum_interval = float(np.max(intervals))
    if np.any(intervals <= 0.0):
        return HistoryQualityResult(
            False,
            "nonincreasing_history_timestamp",
            0,
            maximum_interval,
            float("inf"),
            float("inf"),
        )

    missed = int(
        sum(not bool(getattr(frame, "measurement_accepted", True)) for frame in history_frames)
    )
    # Predict-only Kalman rows causally fill missed or rejected measurements,
    # but the configured limit decides whether such a window may be consumed.
    # Setting the limit to zero requires five consecutive measurement-accepted
    # history frames before Cage inference or supervision can resume.
    maximum_missed = int(quality_config["maximum_missed_detections"])
    if missed > maximum_missed:
        return HistoryQualityResult(
            False,
            "history_too_many_missed_detections",
            missed,
            maximum_interval,
            float("inf"),
            float("inf"),
        )

    allowed_interval = maximum_history_interval_s(timing_config, quality_config)
    if enforce_maximum_interval and maximum_interval > allowed_interval + 1e-12:
        return HistoryQualityResult(
            False,
            "history_interval_too_large",
            missed,
            maximum_interval,
            float("inf"),
            float("inf"),
        )

    curvature = estimate_history_curvature_m_inv(
        history_frames,
        float(quality_config["minimum_speed_for_curvature_mps"]),
    )
    minimum_curvature = float(quality_config.get("minimum_curvature_m_inv", 0.0))
    if curvature < minimum_curvature:
        return HistoryQualityResult(
            False,
            "history_curvature_too_small",
            missed,
            maximum_interval,
            curvature,
            float("inf"),
        )
    if curvature > float(quality_config["maximum_curvature_m_inv"]):
        return HistoryQualityResult(
            False,
            "history_curvature_too_large",
            missed,
            maximum_interval,
            curvature,
            float("inf"),
        )
    turn = estimate_five_frame_turn_deg(history_frames)
    if turn < float(quality_config.get("minimum_five_frame_turn_deg", 0.0)):
        return HistoryQualityResult(
            False,
            "history_five_frame_turn_too_small",
            missed,
            maximum_interval,
            curvature,
            turn,
        )
    if turn > float(quality_config["maximum_five_frame_turn_deg"]):
        return HistoryQualityResult(
            False,
            "history_five_frame_turn_too_large",
            missed,
            maximum_interval,
            curvature,
            turn,
        )

    plane_ratio = estimate_five_frame_plane_ratio(history_frames)
    if plane_ratio < float(
        quality_config.get("minimum_five_frame_plane_ratio", 0.0)
    ):
        return HistoryQualityResult(
            False,
            "history_five_frame_plane_spread_too_small",
            missed,
            maximum_interval,
            curvature,
            turn,
            plane_ratio,
        )

    return HistoryQualityResult(
        True, "", missed, maximum_interval, curvature, turn, plane_ratio
    )
