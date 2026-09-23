from typing import List

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cage_system.learning.dataset import make_cage_input
from cage_system.learning.online_interface import HistoricalCageInference


class TargetSequenceProvider:
    def __init__(self, timestamps: np.ndarray, cage_pose6: np.ndarray):
        if len(timestamps) < 2:
            raise ValueError("A dynamic target sequence needs at least two predictions")
        self.timestamps = np.asarray(timestamps, dtype=float)
        self.pose6 = np.asarray(cage_pose6, dtype=float)
        self.duration = float(self.timestamps[-1] - self.timestamps[0])
        self.start = float(self.timestamps[0])
        self.rotations = Rotation.from_rotvec(self.pose6[:, 3:])
        self.slerp = Slerp(self.timestamps, self.rotations)

    def reset(self):
        return None

    def __call__(self, elapsed_s: float):
        timestamp = float(np.clip(self.start + elapsed_s, self.timestamps[0], self.timestamps[-1]))
        position = np.array(
            [np.interp(timestamp, self.timestamps, self.pose6[:, axis]) for axis in range(3)]
        )
        rotvec = self.slerp([timestamp]).as_rotvec()[0]
        return np.concatenate([position, rotvec])


def build_frozen_cage_sequences(config, trajectories, frozen_cage_network):
    sequences = []
    workspace_min = np.asarray(config["robot"]["workspace_min_m"], dtype=float)
    workspace_max = np.asarray(config["robot"]["workspace_max_m"], dtype=float)
    for trajectory in trajectories:
        inference = HistoricalCageInference(
            frozen_cage_network,
            config["trajectory"],
            config["kalman"],
            config["timing"],
            config["history_quality"],
        )
        timestamps, predictions = [], []
        for observation in trajectory.observations:
            prediction = inference.predict_current_then_update(observation)
            if (
                prediction is not None
                and np.all(prediction[:3] >= workspace_min)
                and np.all(prediction[:3] <= workspace_max)
            ):
                timestamps.append(observation.timestamp)
                predictions.append(prediction)
        if len(timestamps) >= 2:
            sequences.append(TargetSequenceProvider(np.asarray(timestamps), np.asarray(predictions)))
    if not sequences:
        raise RuntimeError("No trajectory produced enough causal Cage predictions")
    return sequences
