from pathlib import Path

import numpy as np

from cage_system.audit_io import open_audit_text
from cage_system.learning.dataset import CageDatasetArrays


def filter_all_zero_est_labels(input_path: str, audit_path: str, output_path: str):
    """Remove labels for which every feasible candidate has zero EST occupancy."""

    arrays = CageDatasetArrays.load(input_path)
    nonzero = set()
    valid_audit_count = 0
    with open_audit_text(Path(audit_path), "r") as handle:
        import json

        for line in handle:
            record = json.loads(line)
            if not record.get("valid", False):
                continue
            valid_audit_count += 1
            best_id = record.get("best_candidate_id")
            best_mean = 0.0
            for candidate in record.get("candidates", []):
                if candidate.get("candidate_id") == best_id:
                    best_mean = float(
                        candidate.get("metrics", {}).get("est_occupancy_mean", 0.0)
                    )
                    break
            if best_mean > 0.0:
                nonzero.add(
                    (
                        int(record["track_id"]),
                        round(float(record["label_timestamp"]), 9),
                    )
                )
    if valid_audit_count != len(arrays.inputs):
        raise ValueError("Audit valid-record count does not match the label archive")
    keep = np.asarray(
        [
            (int(track_id), round(float(timestamp), 9)) in nonzero
            for track_id, timestamp in zip(arrays.track_ids, arrays.label_timestamps)
        ],
        dtype=bool,
    )
    filtered = CageDatasetArrays(
        arrays.inputs[keep].copy(),
        arrays.labels[keep].copy(),
        arrays.track_ids[keep].copy(),
        arrays.label_timestamps[keep].copy(),
        arrays.nominal_detection_period_s,
        arrays.input_source,
        arrays.geometry_contract,
    ).validate()
    filtered.save(output_path)
    return {
        "input_labels": int(len(arrays.inputs)),
        "kept_labels": int(np.sum(keep)),
        "removed_all_zero_est_ties": int(np.sum(~keep)),
    }
