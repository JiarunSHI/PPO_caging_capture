import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cage_system.config import load_config
from cage_system.audit_io import open_audit_text
from cage_system.geometry import pose6_to_transform, transform_to_pose6
from cage_system.io import load_generated_trajectories, save_generated_trajectories
from cage_system.learning.dataset import canonicalize_cage_batch
from cage_system.learning.online_interface import HistoricalCageInference
from cage_system.learning.history_quality import evaluate_history_quality
from cage_system.pipeline import LabelGenerator
from cage_system.trajectory.generator import (
    PlanarTrajectoryGenerator,
    trajectory_diagnostics,
)
from cage_system.trajectory.online import OnlineTrajectoryEstimator
from cage_system.types import PointObservation


class _EchoNetwork:
    def __init__(self):
        self.last_input = None
        self.nominal_detection_period_s = 0.05

    def predict(self, matrix):
        self.last_input = matrix.copy()
        return np.concatenate([matrix[-1, :3], np.zeros(3)])


class DataFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/default.yaml")

    def _observation(self, index, position=None):
        if position is None:
            position = np.array([0.01 * index, 0.0, 0.4])
        return PointObservation(
            0.05 * index,
            np.asarray(position, dtype=float),
        )

    def test_trajectory_archive_preserves_predict_only_dropout_events(self):
        trajectory = PlanarTrajectoryGenerator(
            self.config["trajectory"], seed=410
        ).generate(0)
        trajectory.observations[2].valid = False
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "tracks.npz")
            save_generated_trajectories([trajectory], path)
            loaded = load_generated_trajectories(path)[0]
        self.assertEqual(len(loaded.observations), len(trajectory.observations))
        self.assertFalse(loaded.observations[2].valid)
        self.assertEqual(
            loaded.observations[2].timestamp,
            trajectory.observations[2].timestamp,
        )

    def test_history_canonicalization_round_trips_base_pose(self):
        times = np.linspace(0.0, 0.2, 5)
        positions = np.column_stack(
            (0.4 + times, -0.1 + 0.4 * times**2, 0.45 + 0.1 * times)
        )
        inputs = np.concatenate(
            [positions, np.full((5, 1), 0.2)], axis=1
        )[None, ...]
        labels = np.array([[0.55, -0.04, 0.48, 0.3, -0.2, 0.1]])
        _, relative_labels, references = canonicalize_cage_batch(
            inputs, labels, [0.0, 0.0, 1.0]
        )
        reconstructed = transform_to_pose6(
            references[0] @ pose6_to_transform(relative_labels[0])
        )
        np.testing.assert_allclose(reconstructed[:3], labels[0, :3], atol=1e-10)
        relative_rotation = (
            pose6_to_transform(reconstructed)[:3, :3].T
            @ pose6_to_transform(labels[0])[:3, :3]
        )
        np.testing.assert_allclose(relative_rotation, np.eye(3), atol=1e-10)

    def test_current_measurement_is_not_in_current_cage_input(self):
        first_network = _EchoNetwork()
        second_network = _EchoNetwork()
        first = HistoricalCageInference(
            first_network,
            self.config["trajectory"],
            self.config["kalman"],
            self.config["timing"],
            self.config["history_quality"],
        )
        second = HistoricalCageInference(
            second_network,
            self.config["trajectory"],
            self.config["kalman"],
            self.config["timing"],
            self.config["history_quality"],
        )
        for index in range(5):
            first.predict_current_then_update(self._observation(index))
            second.predict_current_then_update(self._observation(index))
        prediction_a = first.predict_current_then_update(
            self._observation(5, np.array([0.05, 0.0, 0.4]))
        )
        prediction_b = second.predict_current_then_update(
            self._observation(5, np.array([9.0, -4.0, 2.0]))
        )
        np.testing.assert_allclose(first_network.last_input, second_network.last_input)
        np.testing.assert_allclose(prediction_a, prediction_b)

    def test_completed_five_frame_window_predicts_without_next_observation(self):
        network = _EchoNetwork()
        inference = HistoricalCageInference(
            network,
            self.config["trajectory"],
            self.config["kalman"],
            self.config["timing"],
            self.config["history_quality"],
        )
        for index in range(4):
            self.assertIsNone(
                inference.update_then_predict(self._observation(index))
            )
        prediction = inference.update_then_predict(self._observation(4))
        self.assertIsNotNone(prediction)
        self.assertEqual(network.last_input.shape, (5, 4))
        np.testing.assert_allclose(
            network.last_input[:, :3],
            np.asarray([self._observation(index).position for index in range(5)]),
            atol=3e-3,
        )
        self.assertAlmostEqual(inference.last_target_timestamp, 0.25)

    def test_missing_first_visual_detection_does_not_create_fake_history(self):
        network = _EchoNetwork()
        inference = HistoricalCageInference(
            network,
            self.config["trajectory"],
            self.config["kalman"],
            self.config["timing"],
            self.config["history_quality"],
        )
        missing = PointObservation(0.0, np.zeros(3), valid=False)
        self.assertIsNone(inference.predict_current_then_update(missing))
        self.assertEqual(len(inference.history), 0)
        inference.predict_current_then_update(self._observation(1))
        self.assertEqual(len(inference.history), 1)

    def test_cage_speed_is_pure_causal_kalman_speed(self):
        estimator = OnlineTrajectoryEstimator(
            self.config["trajectory"],
            self.config["kalman"],
            self.config["timing"],
        )
        for index in range(12):
            timestamp = 0.05 * index
            position = np.array(
                [0.4 + 0.12 * timestamp, 0.03 * timestamp**2, 0.45]
            )
            frame = estimator.update(PointObservation(timestamp, position))
            self.assertAlmostEqual(
                frame.speed,
                float(np.linalg.norm(estimator.filter.state[3:6])),
                places=12,
            )

    def test_missing_frame_is_causally_imputed_without_suppressing_cage(self):
        network = _EchoNetwork()
        inference = HistoricalCageInference(
            network,
            self.config["trajectory"],
            self.config["kalman"],
            self.config["timing"],
            self.config["history_quality"],
        )
        for index in range(5):
            self.assertIsNone(
                inference.predict_current_then_update(self._observation(index))
            )
        self.assertIsNotNone(
            inference.predict_current_then_update(self._observation(5))
        )
        missing = PointObservation(0.30, np.zeros(3), valid=False)
        self.assertIsNotNone(inference.predict_current_then_update(missing))
        self.assertTrue(inference.last_quality.valid)
        for index in range(7, 12):
            self.assertIsNotNone(
                inference.predict_current_then_update(self._observation(index))
            )
        self.assertIsNotNone(
            inference.predict_current_then_update(self._observation(12))
        )
        self.assertTrue(inference.ready)

    def test_history_quality_imputes_miss_but_rejects_gap_curvature_and_turn(self):
        class Frame:
            def __init__(self, index, accepted=True, curvature=1.0):
                self.timestamp = 0.05 * index
                self.position = np.array([0.01 * index, 0.0, 0.4])
                self.speed = 0.2
                self.measurement_accepted = accepted
                self.curvature = curvature

        normal = [Frame(index) for index in range(5)]
        quality = evaluate_history_quality(
            normal, 0.25, self.config["timing"], self.config["history_quality"]
        )
        self.assertTrue(quality.valid)

        missed = [Frame(index, accepted=index != 2) for index in range(5)]
        quality = evaluate_history_quality(
            missed, 0.25, self.config["timing"], self.config["history_quality"]
        )
        self.assertTrue(quality.valid)
        self.assertEqual(quality.missed_detections, 1)

        quality = evaluate_history_quality(
            normal, 0.31, self.config["timing"], self.config["history_quality"]
        )
        self.assertEqual(quality.reason, "history_interval_too_large")

        curved = [Frame(index, curvature=9.0) for index in range(5)]
        quality = evaluate_history_quality(
            curved, 0.25, self.config["timing"], self.config["history_quality"]
        )
        self.assertEqual(quality.reason, "history_curvature_too_large")

        turning = [Frame(index) for index in range(5)]
        angles = np.radians([0.0, 8.0, 16.0, 24.0, 32.0])
        for frame, angle in zip(turning, angles):
            frame.tangent = np.array([np.cos(angle), np.sin(angle), 0.0])
        quality = evaluate_history_quality(
            turning, 0.25, self.config["timing"], self.config["history_quality"]
        )
        self.assertEqual(quality.reason, "history_five_frame_turn_too_large")

    def test_label_audit_has_strictly_past_history(self):
        config = copy.deepcopy(self.config)
        config["trajectory"]["duration_s"] = [2.5, 2.5]
        config["trajectory"]["dropout_probability"] = 0.0
        config["trajectory"]["outlier_probability"] = 0.0
        config["est"]["node_count"] = 30
        config["est"]["minimum_retention_ratio"] = 0.0
        config["candidate"]["radius_min_m"] = 0.110
        config["candidate"]["radius_max_m"] = 0.120
        config["candidate"]["nominal_samples_per_arc"] = 4
        config["candidate"]["orientation_branches"] = [1]
        trajectory = PlanarTrajectoryGenerator(config["trajectory"], seed=13).generate(3)
        with tempfile.TemporaryDirectory() as directory:
            audit_path = str(Path(directory) / "audit.jsonl.gz")
            arrays = LabelGenerator(config).generate([trajectory], audit_path)
            self.assertEqual(arrays.inputs.shape[1:], (5, 4))
            self.assertEqual(arrays.labels.shape[1:], (6,))
            with open_audit_text(audit_path, "r") as handle:
                records = [json.loads(line) for line in handle]
            valid_records = [record for record in records if record["valid"]]
            self.assertGreater(len(valid_records), 0)
            for record in valid_records:
                self.assertEqual(len(record["history_timestamps"]), 5)
                self.assertLess(max(record["history_timestamps"]), record["label_timestamp"])
                self.assertTrue(record["history_excludes_current"])

    def test_generated_tracks_satisfy_curvature_and_local_plane_contract(self):
        trajectory_config = self.config["trajectory"]
        generator = PlanarTrajectoryGenerator(trajectory_config, seed=91)
        for track_id in range(16):
            trajectory = generator.generate(track_id)
            diagnostics = trajectory_diagnostics(
                trajectory.timestamps,
                trajectory.true_positions,
                trajectory_config["curvature_history_frames"],
            )
            curvature_minimum, curvature_maximum = trajectory_config[
                "five_frame_curvature_range_bounds_m_inv"
            ]
            plane_minimum, plane_maximum = trajectory_config[
                "local_plane_total_variation_bounds_deg"
            ]
            self.assertGreaterEqual(
                diagnostics.five_frame_curvature_range_min_m_inv,
                curvature_minimum,
            )
            self.assertLessEqual(
                diagnostics.five_frame_curvature_range_max_m_inv,
                curvature_maximum,
            )
            self.assertLessEqual(
                diagnostics.curvature_max_m_inv,
                trajectory_config["maximum_curvature_m_inv"],
            )
            self.assertGreaterEqual(
                diagnostics.local_plane_total_variation_deg,
                plane_minimum,
            )
            self.assertLessEqual(
                diagnostics.local_plane_total_variation_deg,
                plane_maximum,
            )
            self.assertLessEqual(
                diagnostics.local_plane_max_step_deg,
                trajectory_config["local_plane_max_step_deg"],
            )
            self.assertLessEqual(
                diagnostics.five_frame_turn_max_deg,
                trajectory_config["maximum_five_frame_turn_deg"],
            )


if __name__ == "__main__":
    unittest.main()
