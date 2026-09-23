"""Training-only privileged reset helpers for local PPO tracking.

Ground-truth trajectory coordinates are allowed only to construct MuJoCo
reset/recovery states.  They must never be copied into a PPO observation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryProjection:
    indices: np.ndarray
    distances_m: np.ndarray


def truth_history_matrix(timestamps, positions, start_index, length=5):
    """Return one privileged [5, xyz+speed] window from a known track."""
    timestamps = np.asarray(timestamps, dtype=float)
    positions = np.asarray(positions, dtype=float)
    start = int(start_index)
    length = int(length)
    if positions.shape != (len(timestamps), 3):
        raise ValueError("Known trajectory positions must have shape [N,3]")
    if start < 0 or start + length > len(timestamps):
        raise IndexError("Privileged history window is outside the trajectory")
    velocity = np.gradient(positions, timestamps, axis=0, edge_order=2)
    speed = np.linalg.norm(velocity, axis=1)
    return np.column_stack(
        (positions[start : start + length], speed[start : start + length])
    )


def project_predictions_to_known_trajectory(
    predicted_positions,
    true_positions,
    last_detected_position,
    lookahead_samples,
    maximum_distance_m,
):
    """Project three KF predictions to an ordered nearby true-track segment.

    Matching is spatial, not an equal-timestamp lookup.  A forward, unique,
    local index constraint prevents a curved or self-near path from mapping
    the predictions backward or repeatedly to the same point.
    """
    predicted = np.asarray(predicted_positions, dtype=float)
    truth = np.asarray(true_positions, dtype=float)
    last = np.asarray(last_detected_position, dtype=float)
    if predicted.shape != (3, 3):
        raise ValueError("Recovery requires exactly three predicted xyz points")
    if truth.ndim != 2 or truth.shape[1] != 3 or not len(truth):
        raise ValueError("Known trajectory must have shape [N,3]")
    anchor = int(np.argmin(np.linalg.norm(truth - last, axis=1)))
    cursor = anchor + 1
    selected = []
    distances = []
    for point in predicted:
        upper = min(len(truth), cursor + int(lookahead_samples))
        if cursor >= upper:
            raise ValueError("known_trajectory_recovery_window_exhausted")
        local_distances = np.linalg.norm(truth[cursor:upper] - point, axis=1)
        relative = int(np.argmin(local_distances))
        index = cursor + relative
        distance = float(local_distances[relative])
        if distance > float(maximum_distance_m):
            raise ValueError("kalman_prediction_has_no_nearby_true_track_point")
        selected.append(index)
        distances.append(distance)
        cursor = index + 1
    return TrajectoryProjection(
        np.asarray(selected, dtype=int), np.asarray(distances, dtype=float)
    )


def recovery_history_matrix(
    last_two_detected_positions,
    matched_true_positions,
    sample_period_s,
):
    """Build the training-only recovery Cage input with consistent speeds."""
    detected = np.asarray(last_two_detected_positions, dtype=float)
    matched = np.asarray(matched_true_positions, dtype=float)
    if detected.shape != (2, 3) or matched.shape != (3, 3):
        raise ValueError("Recovery history is two detected plus three matched xyz rows")
    positions = np.vstack((detected, matched))
    times = np.arange(5, dtype=float) * float(sample_period_s)
    velocity = np.gradient(positions, times, axis=0, edge_order=2)
    return np.column_stack((positions, np.linalg.norm(velocity, axis=1)))
