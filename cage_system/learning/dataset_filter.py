from collections import Counter, deque

import numpy as np

from cage_system.io import validate_trajectory_detection_period
from cage_system.learning.dataset import CageDatasetArrays
from cage_system.learning.history_quality import evaluate_history_quality
from cage_system.trajectory.online import OnlineTrajectoryEstimator


def filter_cage_labels_by_history_quality(config, trajectories, arrays):
    """Keep only labels whose exact causal source window passes production gates."""

    arrays.validate()
    nominal_detection_period_s = 1.0 / float(
        config["timing"]["detection_rate_hz"]
    )
    validate_trajectory_detection_period(
        trajectories, nominal_detection_period_s
    )
    if arrays.nominal_detection_period_s is not None:
        arrays.require_detection_period(nominal_detection_period_s)
    label_lookup = {}
    for index, (track_id, timestamp) in enumerate(
        zip(arrays.track_ids, arrays.label_timestamps)
    ):
        label_lookup[(int(track_id), round(float(timestamp), 9))] = index

    keep = np.zeros(len(arrays.inputs), dtype=bool)
    reasons = Counter()
    matched = np.zeros(len(arrays.inputs), dtype=bool)
    for trajectory in trajectories:
        online = OnlineTrajectoryEstimator(
            config["trajectory"], config["kalman"], config["timing"]
        )
        history = deque(maxlen=int(config["timing"]["history_length"]))
        for observation in trajectory.observations:
            timestamp = float(observation.timestamp)
            quality = evaluate_history_quality(
                list(history),
                timestamp,
                config["timing"],
                config["history_quality"],
            )
            index = label_lookup.get(
                (int(trajectory.track_id), round(timestamp, 9))
            )
            if index is not None:
                matched[index] = True
                if quality.valid:
                    keep[index] = True
                else:
                    reasons[quality.reason] += 1
            current_frame = online.update(observation)
            history.append(current_frame)

    unmatched = int(np.sum(~matched))
    if unmatched:
        reasons["label_timestamp_not_found_in_source_trajectory"] += unmatched
    filtered = CageDatasetArrays(
        arrays.inputs[keep].copy(),
        arrays.labels[keep].copy(),
        arrays.track_ids[keep].copy(),
        arrays.label_timestamps[keep].copy(),
        nominal_detection_period_s,
        arrays.input_source,
        arrays.geometry_contract,
    ).validate()
    summary = {
        "input_labels": int(len(arrays.inputs)),
        "kept_labels": int(np.sum(keep)),
        "removed_labels": int(np.sum(~keep)),
        "removed_by_reason": dict(sorted(reasons.items())),
    }
    return filtered, summary
