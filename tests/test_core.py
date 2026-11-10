import copy
import unittest

import numpy as np

from cage_system.cage.candidates import CandidateGenerator, SpeedRadiusMapper
from cage_system.cage.est import DisturbanceTape, MujocoESTEvaluator
from cage_system.cage.feasibility import CandidateFeasibilityChecker
from cage_system.cage.internal_region import CageInternalRegion
from cage_system.cage.plane import LocalMotionPlaneEstimator
from cage_system.cage.selector import CandidateSelector
from cage_system.config import load_config
from cage_system.geometry import make_transform
from cage_system.trajectory.kalman import ConstantAccelerationKalman3D
from cage_system.trajectory.spline import SharedCubicSpline3D
from cage_system.types import CandidatePose, ObjectState


class CoreAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/default.yaml")

    def test_kalman_uses_real_time_and_rejects_large_jump(self):
        kalman = ConstantAccelerationKalman3D(self.config["kalman"])
        kalman.update(0.0, np.zeros(3))
        kalman.update(0.04, np.array([0.004, 0.0, 0.0]))
        result = kalman.update(0.11, np.array([2.0, 0.0, 0.0]))
        self.assertFalse(result.measurement_accepted)
        self.assertLess(result.position[0], 0.1)

    def test_shared_spline_derivatives_are_smooth(self):
        times = np.linspace(0.0, 1.0, 31)
        positions = np.column_stack((times ** 2, np.sin(times), 0.2 * times ** 3))
        spline = SharedCubicSpline3D(0.12, 1e-8).fit(times, positions)
        velocity = spline.evaluate(0.5, 1)
        acceleration = spline.evaluate(0.5, 2)
        np.testing.assert_allclose(velocity, [1.0, np.cos(0.5), 0.15], atol=2e-2)
        np.testing.assert_allclose(acceleration, [2.0, -np.sin(0.5), 0.6], atol=7e-2)

    def test_line_plane_normal_is_continuous(self):
        estimator = LocalMotionPlaneEstimator(self.config["plane"])
        points = np.column_stack((np.linspace(0, 1, 20), np.zeros(20), np.zeros(20)))
        first = estimator.estimate(points, points[5], np.array([1.0, 0.0, 0.0]), 0.0)
        noisy = points.copy()
        noisy[:, 1] += np.linspace(0.0, 1e-5, len(noisy))
        second = estimator.estimate(noisy, noisy[6], np.array([1.0, 0.0, 0.0]), 0.0)
        self.assertGreater(np.dot(first.normal, second.normal), 0.999)
        self.assertGreater(np.dot(first.normal, np.array([0.0, 0.0, 1.0])), 0.99)

    def test_speed_radius_is_linear_then_rate_limited(self):
        mapper = SpeedRadiusMapper(self.config["candidate"])
        first, audit = mapper.map(0.05, 0.0)
        second, _ = mapper.map(0.45, 0.05)
        self.assertGreaterEqual(first, self.config["candidate"]["radius_min_m"])
        self.assertLessEqual(second, self.config["candidate"]["radius_max_m"])
        self.assertEqual(audit["mapping"], "linear_speed_to_candidate_radius_then_clip_and_smooth")
        maximum_change = self.config["candidate"]["radius_max_rate_mps"] * 0.05
        self.assertLessEqual(abs(second - first), maximum_change + 1e-12)

    def test_candidate_circle_is_planar_and_forward(self):
        generator = CandidateGenerator(self.config["candidate"], np.eye(4))
        arc = generator.construct_arc(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            0.1,
            0.0,
            1.0,
            lambda t: np.array([t, 0.15 * t * t, 0.0]),
        )
        self.assertGreater(arc.tangent_intersection[0], 0.0)
        arc_points = arc.center + arc.radius * (
            np.cos([arc.angle_start, arc.angle_end])[:, None] * arc.u_axis
            + np.sin([arc.angle_start, arc.angle_end])[:, None] * arc.v_axis
        )
        np.testing.assert_allclose(arc_points[0], arc.tangent_intersection, atol=1e-9)
        np.testing.assert_allclose(arc_points[-1], arc.future_intersection, atol=1e-9)
        angles = generator.sample_arc(arc)
        candidates = generator.build_poses(arc, angles)
        region = CageInternalRegion(
            self.config["gripper"]["internal_region_endpoint_frame"],
            self.config["gripper"]["safety_gap_m"],
        )
        for candidate in candidates:
            self.assertEqual(candidate.normal_branch, 1)
            self.assertAlmostEqual(np.linalg.norm(candidate.endpoint_position_base), 0.1, places=8)
            self.assertAlmostEqual(candidate.endpoint_position_base[2], 0.0, places=8)
            self.assertAlmostEqual(candidate.endpoint_rotation_base[2, 2], 1.0, places=8)
            direction_to_object = -candidate.endpoint_position_base
            direction_to_object /= np.linalg.norm(direction_to_object)
            np.testing.assert_allclose(
                candidate.endpoint_rotation_base[:, 0], direction_to_object, atol=1e-9
            )
            np.testing.assert_allclose(
                candidate.endpoint_rotation_base[:, :2].T @ np.array([0.0, 0.0, 1.0]),
                np.zeros(2),
                atol=1e-9,
            )
            self.assertFalse(
                region.evaluate(
                    make_transform(
                        candidate.endpoint_position_base,
                        candidate.endpoint_rotation_base,
                    ),
                    np.zeros(3),
                ).inside
            )

    def test_opposite_normal_branch_is_rejected(self):
        candidate_config = copy.deepcopy(self.config["candidate"])
        candidate_config["orientation_branches"] = [1, -1]
        generator = CandidateGenerator(candidate_config, np.eye(4))
        arc = generator.construct_arc(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            0.1,
            0.0,
            1.0,
            lambda t: np.array([t, 0.1 * t * t, 0.0]),
        )
        with self.assertRaisesRegex(ValueError, "positive motion-plane normal"):
            generator.build_poses(arc, generator.sample_arc(arc))

    def test_straight_arc_collapses_at_the_two_true_intersections(self):
        generator = CandidateGenerator(self.config["candidate"], np.eye(4))
        arc = generator.construct_arc(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            0.1,
            0.0,
            1.0,
            lambda t: np.array([t, 0.0, 0.0]),
        )
        self.assertTrue(arc.degenerate)
        self.assertAlmostEqual(arc.angle_start, 0.0)
        self.assertAlmostEqual(arc.angle_end, 0.0)
        np.testing.assert_allclose(arc.tangent_intersection, arc.future_intersection)
        self.assertEqual(len(generator.sample_arc(arc)), 1)

    def test_previous_best_dense_sampling_cannot_replace_arc_boundaries(self):
        generator = CandidateGenerator(self.config["candidate"], np.eye(4))
        arc = generator.construct_arc(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            0.1,
            0.0,
            2.0,
            lambda t: np.array([t, 0.15 * t * t, 0.0]),
        )
        angles = generator.sample_arc(arc, previous_best_angle=arc.angle_end)
        self.assertTrue(np.any(np.isclose(angles, arc.angle_start, atol=1e-14)))
        self.assertTrue(np.any(np.isclose(angles, arc.angle_end, atol=1e-14)))

    def test_point_mass_controls_containment(self):
        gripper = self.config["gripper"]
        region = CageInternalRegion(
            gripper["internal_region_endpoint_frame"],
            gripper["safety_gap_m"],
        )
        result = region.evaluate(
            np.eye(4),
            np.array([0.10, 0.062, 0.0]),
        )
        self.assertFalse(result.inside)
        self.assertTrue(result.illegal_penetration)

    def test_est_reuses_deterministic_disturbance_tape(self):
        gripper = self.config["gripper"]
        region = CageInternalRegion(
            gripper["internal_region_endpoint_frame"],
            gripper["safety_gap_m"],
        )
        est_config = copy.deepcopy(self.config["est"])
        est_config["node_count"] = 50
        evaluator = MujocoESTEvaluator(
            est_config, gripper, self.config["vision"], region
        )
        tape = DisturbanceTape.generate(est_config, 7)
        state = ObjectState(
            0.0,
            np.array([0.10, 0.0, 0.0]),
            np.zeros(3),
        )
        first = evaluator.evaluate(state, np.eye(4), tape)
        second = evaluator.evaluate(state, np.eye(4), tape)
        self.assertEqual(first.total_nodes, est_config["node_count"])
        self.assertEqual(first.inside_nodes, second.inside_nodes)
        np.testing.assert_allclose(first.nodes_base, second.nodes_base)

    def test_est_root_outside_capture_set_still_expands(self):
        gripper = self.config["gripper"]
        region = CageInternalRegion(
            gripper["internal_region_endpoint_frame"], gripper["safety_gap_m"]
        )
        est_config = copy.deepcopy(self.config["est"])
        est_config["node_count"] = 80
        evaluator = MujocoESTEvaluator(
            est_config, gripper, self.config["vision"], region
        )
        state = ObjectState(0.0, np.array([0.115, 0.0, 0.40]), np.array([-0.3, 0.0, 0.0]))
        transform = make_transform(np.array([0.0, 0.0, 0.40]), np.eye(3))
        result = evaluator.evaluate(state, transform, DisturbanceTape.generate(est_config, 42))
        self.assertEqual(result.total_nodes, 80)
        self.assertFalse(region.evaluate(transform, state.position).inside)
        self.assertGreater(result.inside_nodes, 0)
        self.assertEqual(result.node_parent_indices[0], -1)
        self.assertTrue(
            all(
                0 <= result.node_parent_indices[index] < index
                for index in range(1, result.total_nodes)
            )
        )
        self.assertLessEqual(np.max(result.node_times), est_config["horizon_s"] + 1e-12)
        integration_dt = est_config["integration_dt_s"]
        np.testing.assert_allclose(
            result.node_times / integration_dt,
            np.round(result.node_times / integration_dt),
            atol=1e-8,
        )

    def test_est_mean_strictly_outranks_joint_change_and_pose_continuity(self):
        selector = CandidateSelector(self.config["est"])
        larger_joint_change = CandidatePose(
            "higher_est",
            np.zeros(3),
            np.eye(3),
            np.eye(4),
            0.0,
            1,
            feasible=True,
            metrics={
                "est_occupancy_mean": 0.10,
                "joint_delta": 0.3,
                "temporal_continuity_cost": 0.1,
            },
        )
        smaller_joint_change = CandidatePose(
            "smaller_joint_change",
            np.zeros(3),
            np.eye(3),
            np.eye(4),
            0.0,
            1,
            feasible=True,
            metrics={
                "est_occupancy_mean": 0.09,
                "joint_delta": 0.1,
                "temporal_continuity_cost": 0.5,
            },
        )
        selection = selector.select([larger_joint_change, smaller_joint_change])
        self.assertTrue(selection.valid)
        self.assertEqual(selection.best.candidate_id, "higher_est")

        same_joint_smoother_pose = CandidatePose(
            "smoother_pose",
            np.zeros(3),
            np.eye(3),
            np.eye(4),
            0.0,
            1,
            feasible=True,
            metrics={
                "est_occupancy_mean": 0.09,
                "joint_delta": 0.1,
                "temporal_continuity_cost": 0.2,
            },
        )
        selection = selector.select([smaller_joint_change, same_joint_smoother_pose])
        self.assertEqual(selection.best.candidate_id, "smaller_joint_change")

    def test_cartesian_prefilter_skips_expensive_ik(self):
        class BackendThatMustNotRun:
            name = "test"

            def solve(self, transform_base_tcp, seed_joints):
                raise AssertionError("IK must not run for an out-of-workspace pose")

        checker = CandidateFeasibilityChecker(
            self.config["robot"],
            self.config["candidate"],
            self.config["timing"],
            BackendThatMustNotRun(),
        )
        transform = np.eye(4)
        transform[:3, 3] = [2.0, 0.0, 0.5]
        candidate = CandidatePose(
            "outside",
            transform[:3, 3].copy(),
            np.eye(3),
            transform,
            0.0,
            1,
        )
        checker.check(candidate, np.zeros(6), np.zeros(6), 0.05)
        self.assertFalse(candidate.feasible)
        self.assertIn("outside_workspace", candidate.rejection_reasons)
        self.assertEqual(
            candidate.metrics["feasibility_stage"], "cartesian_prefilter"
        )

    def test_final_geometry_config_uses_mujoco_without_visual_training(self):
        production = load_config("configs/ur5e_cage_state_arc_data.yaml")
        self.assertTrue(production["system"]["production_mode"])
        self.assertEqual(production["labeling"]["feasibility_backend"], "mujoco")
        self.assertFalse(production["vision"]["require_rendered_cage_training_inputs"])
        self.assertFalse(production["vision"]["enabled"])


if __name__ == "__main__":
    unittest.main()
