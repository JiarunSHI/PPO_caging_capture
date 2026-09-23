import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cage_system.config import load_config
from cage_system.learning.dataset import CageDatasetArrays
from cage_system.perception.eye_on_hand import (
    CameraIntrinsics,
    RedColorMaskDetector,
    unproject_depth_mask,
)


class LearningAndSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/default.yaml")

    def test_cage_training_checkpoint_and_frozen_inference_contract(self):
        try:
            from cage_system.learning.cage_network import FrozenCagePredictor, train_cage_network
        except ImportError as error:
            self.skipTest(str(error))
        rng = np.random.RandomState(5)
        inputs = rng.normal(size=(24, 5, 4)).astype(np.float32)
        labels = np.column_stack(
            (inputs[:, -1, :3] + 0.01, np.zeros((len(inputs), 3)))
        ).astype(np.float32)
        arrays = CageDatasetArrays(
            inputs,
            labels,
            np.repeat(np.arange(4), 6),
            np.arange(len(inputs), dtype=float) * 0.05,
            0.05,
        )
        config = copy.deepcopy(self.config)
        config["cage_network"]["hidden_dims"] = [16, 16]
        config["cage_network"]["batch_size"] = 8
        config["cage_network"]["epochs"] = 2
        config["vision"]["require_rendered_cage_training_inputs"] = False
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = str(Path(directory) / "labels.npz")
            checkpoint_path = str(Path(directory) / "cage.pt")
            arrays.save(dataset_path)
            metrics = train_cage_network(
                config, dataset_path, checkpoint_path, device="cpu"
            )
            predictor = FrozenCagePredictor(checkpoint_path, device="cpu")
            prediction = predictor.predict(inputs[0])
            self.assertEqual(prediction.shape, (6,))
            self.assertTrue(np.all(np.isfinite(prediction)))
            self.assertEqual(metrics["device"], "cpu")

    def test_ppo_state_actor_and_joint_density_shapes(self):
        try:
            import torch
            from cage_system.learning.ppo2 import SharedActorCritic, build_ppo_state
        except ImportError as error:
            self.skipTest(str(error))
        state = build_ppo_state(
            np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
        )
        self.assertEqual(state.shape, (18,))
        model = SharedActorCritic(18, 6, [16])
        actions, log_density, values, entropy = model.sample(
            torch.zeros((3, 18)), deterministic=False
        )
        self.assertEqual(tuple(actions.shape), (3, 6))
        self.assertEqual(tuple(log_density.shape), (3,))
        self.assertEqual(tuple(values.shape), (3,))
        self.assertEqual(tuple(entropy.shape), (3,))
        reevaluated_log_density, _, _ = model.evaluate_actions(
            torch.zeros((3, 18)), actions
        )
        density_ratio = torch.exp(reevaluated_log_density - log_density)
        torch.testing.assert_close(
            density_ratio, torch.ones_like(density_ratio), atol=2e-5, rtol=2e-5
        )
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))

    def test_ppo_state_is_tcp_to_cage_relative_pose_not_absolute_target(self):
        from cage_system.learning.ppo2 import build_ppo_state

        target = np.array([0.5, -0.1, 0.4, 0.0, 0.0, np.pi / 2])
        tcp = np.array([0.48, -0.08, 0.39, 0.0, 0.0, 0.0])
        state = build_ppo_state(target, tcp, np.zeros(6), np.zeros(6))
        np.testing.assert_allclose(state[:3], [0.02, -0.02, 0.01], atol=1e-7)
        np.testing.assert_allclose(
            state[3:6], [0.0, 0.0, np.pi / 2], atol=1e-6
        )
        self.assertFalse(np.allclose(state[:6], target))

    def test_actor_and_critic_do_not_share_trainable_parameters(self):
        try:
            from cage_system.learning.ppo2 import SharedActorCritic
        except ImportError as error:
            self.skipTest(str(error))
        model = SharedActorCritic(18, 6, [16])
        actor_ids = {id(item) for item in model.actor.parameters()}
        critic_ids = {id(item) for item in model.critic.parameters()}
        self.assertTrue(actor_ids)
        self.assertTrue(critic_ids)
        self.assertTrue(actor_ids.isdisjoint(critic_ids))

    def test_ppo_checkpoint_requires_same_cage_reward_contract(self):
        try:
            import torch
            from cage_system.learning.ppo2 import (
                PPO_ACTION_DISTRIBUTION_CONTRACT,
                PPO_CHECKPOINT_FORMAT_VERSION,
                PPO_REWARD_TARGET_CONTRACT,
                PPO2Trainer,
                load_frozen_ppo,
            )
        except ImportError as error:
            self.skipTest(str(error))
        config = copy.deepcopy(self.config["ppo"])
        config["hidden_dims"] = [16]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = str(Path(directory) / "ppo.pt")
            obsolete_path = str(Path(directory) / "ppo_obsolete.pt")
            obsolete_initialization_path = str(
                Path(directory) / "ppo_obsolete_initialization.pt"
            )
            trainer = PPO2Trainer(config, device="cpu")
            raw_state = np.linspace(-0.5, 0.7, 18, dtype=np.float32)
            trainer.normalize_state(raw_state, update=True)
            trainer.save(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(
                checkpoint["format_version"], PPO_CHECKPOINT_FORMAT_VERSION
            )
            self.assertEqual(
                checkpoint["action_distribution"],
                PPO_ACTION_DISTRIBUTION_CONTRACT,
            )
            self.assertEqual(
                checkpoint["reward_target_contract"], PPO_REWARD_TARGET_CONTRACT
            )
            frozen = load_frozen_ppo(checkpoint_path, device="cpu")
            raw_tensor = torch.from_numpy(raw_state).unsqueeze(0)
            expected_tensor = torch.from_numpy(
                trainer.normalize_state(raw_state, update=False)
            ).unsqueeze(0)
            with torch.no_grad():
                frozen_action, _, _, _ = frozen.sample(
                    raw_tensor, deterministic=True
                )
                expected_action, _, _, _ = trainer.model.sample(
                    expected_tensor, deterministic=True
                )
            torch.testing.assert_close(frozen_action, expected_action)
            old_initialization = dict(checkpoint)
            old_initialization["initialization_contract"] = (
                "c7_track_damped_jacobian_bc_mixed_dagger_critic_warmup_v1"
            )
            torch.save(old_initialization, obsolete_initialization_path)
            with self.assertRaises(ValueError):
                load_frozen_ppo(obsolete_initialization_path, device="cpu")
            checkpoint.pop("reward_target_contract")
            torch.save(checkpoint, obsolete_path)
            with self.assertRaises(ValueError):
                load_frozen_ppo(obsolete_path, device="cpu")

    def test_tanh_entropy_penalizes_saturated_action_distribution(self):
        try:
            import torch
            from cage_system.learning.ppo2 import SharedActorCritic
        except ImportError as error:
            self.skipTest(str(error))
        model = SharedActorCritic(18, 6, [16])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.log_std.fill_(-0.7)
        states = torch.zeros((4096, 18))
        actions = torch.zeros((4096, 6))
        torch.manual_seed(19)
        _, _, centered_entropy = model.evaluate_actions(states, actions)
        with torch.no_grad():
            model.mean_head.bias.fill_(4.0)
        torch.manual_seed(19)
        _, _, saturated_entropy = model.evaluate_actions(states, actions)
        self.assertLess(
            float(saturated_entropy.mean()),
            float(centered_entropy.mean()) - 5.0,
        )

    def test_ppo_update_reports_stability_diagnostics(self):
        try:
            import torch
            from cage_system.learning.ppo2 import PPO2Trainer, RolloutBatch
        except ImportError as error:
            self.skipTest(str(error))
        config = copy.deepcopy(self.config["ppo"])
        config["hidden_dims"] = [16]
        config["update_epochs"] = 2
        config["minibatch_size"] = 4
        config["target_kl"] = 1.0
        trainer = PPO2Trainer(config, device="cpu")
        rng = np.random.RandomState(17)
        states = rng.normal(size=(8, 18)).astype(np.float32)
        with torch.no_grad():
            actions, old_logs, old_values, _ = trainer.model.sample(
                torch.from_numpy(states)
            )
            _, _, next_values = trainer.model(torch.from_numpy(states))
        rollout = RolloutBatch(
            states=states,
            actions=actions.numpy(),
            rewards=rng.normal(size=8).astype(np.float32),
            next_states=states.copy(),
            dones=np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
            old_log_densities=old_logs.numpy(),
            old_values=old_values.numpy(),
            next_values=next_values.numpy(),
        )
        metrics = trainer.update(rollout)
        for key in (
            "approx_kl",
            "approx_kl_max",
            "clip_fraction",
            "action_saturation_fraction",
            "explained_variance",
            "kl_early_stopped",
        ):
            self.assertIn(key, metrics)
        self.assertTrue(np.isfinite(metrics["entropy"]))

    def test_gae_bootstraps_collection_truncation_but_stops_recursion(self):
        from cage_system.learning.ppo2 import compute_gae

        truncated = compute_gae(
            rewards=np.array([0.0]),
            values=np.array([0.0]),
            next_values=np.array([2.0]),
            dones=np.array([1.0]),
            gamma=0.9,
            gae_lambda=0.95,
            bootstrap_dones=np.array([0.0]),
        )
        terminated = compute_gae(
            rewards=np.array([0.0]),
            values=np.array([0.0]),
            next_values=np.array([2.0]),
            dones=np.array([1.0]),
            gamma=0.9,
            gae_lambda=0.95,
            bootstrap_dones=np.array([1.0]),
        )
        self.assertAlmostEqual(float(truncated[0]), 1.8)
        self.assertAlmostEqual(float(terminated[0]), 0.0)

    def test_ppo_rollout_skips_policy_inference_while_history_is_abnormal(self):
        try:
            from cage_system.learning.ppo2 import PPO2Trainer
            from cage_system.learning.train_ppo import collect_complete_rollout
        except ImportError as error:
            self.skipTest(str(error))

        class GatedEnvironment:
            def __init__(self):
                self.phase = 0
                self._ready = True
                self.actions = []

            @property
            def action_ready(self):
                return self._ready

            def reset(self):
                self.phase = 0
                self._ready = True
                return np.zeros(18, dtype=np.float32), {}

            def step(self, action):
                self.actions.append(np.asarray(action).copy())
                current = np.zeros(18, dtype=np.float32)
                self.phase += 1
                if self.phase in (1, 2):
                    self._ready = False
                else:
                    self._ready = True
                truncated = self.phase >= 4
                return (
                    current,
                    -0.1,
                    False,
                    truncated,
                    {
                        "reward_cage_pose6": current[:6].copy(),
                        "next_cage_pose6": current[:6].copy(),
                        "action_state_relative_error6": current[:6].copy(),
                        "next_state_relative_error6": current[:6].copy(),
                        "reward_target_contract": "same_cage_pose_as_action_state",
                    },
                )

        config = copy.deepcopy(self.config["ppo"])
        config["hidden_dims"] = [16]
        trainer = PPO2Trainer(config, device="cpu")
        original_sample = trainer.model.sample
        sample_count = [0]

        def counted_sample(*args, **kwargs):
            sample_count[0] += 1
            return original_sample(*args, **kwargs)

        trainer.model.sample = counted_sample
        environment = GatedEnvironment()
        rollout = collect_complete_rollout(trainer, environment, minimum_steps=2)
        self.assertEqual(len(rollout.states), 2)
        self.assertEqual(sample_count[0], 2)
        np.testing.assert_allclose(environment.actions[1], np.zeros(6))
        np.testing.assert_allclose(environment.actions[2], np.zeros(6))
        self.assertEqual(rollout.dones.tolist(), [1.0, 1.0])

    def test_privileged_recovery_projects_kalman_points_forward_in_space(self):
        from cage_system.learning.privileged_recovery import (
            project_predictions_to_known_trajectory,
            recovery_history_matrix,
        )

        truth = np.column_stack(
            (np.linspace(0.0, 0.20, 21), np.zeros(21), np.full(21, 0.4))
        )
        predictions = truth[[7, 9, 12]] + np.array(
            [[0.001, 0.002, 0.0], [-0.002, 0.001, 0.0], [0.002, -0.001, 0.0]]
        )
        projection = project_predictions_to_known_trajectory(
            predictions,
            truth,
            truth[5],
            lookahead_samples=10,
            maximum_distance_m=0.01,
        )
        self.assertEqual(projection.indices.tolist(), [7, 9, 12])
        self.assertTrue(np.all(np.diff(projection.indices) > 0))
        matrix = recovery_history_matrix(
            truth[[4, 5]], truth[projection.indices], 0.05
        )
        self.assertEqual(matrix.shape, (5, 4))
        self.assertTrue(np.all(np.isfinite(matrix)))

    def test_rollout_marks_privileged_recovery_as_gae_boundary(self):
        try:
            from cage_system.learning.ppo2 import PPO2Trainer
            from cage_system.learning.train_ppo import collect_complete_rollout
        except ImportError as error:
            self.skipTest(str(error))

        class RecoveryEnvironment:
            action_ready = True

            def __init__(self):
                self.phase = 0

            def reset(self):
                self.phase = 0
                return np.zeros(18, dtype=np.float32), {}

            def step(self, action):
                self.phase += 1
                state = np.zeros(18, dtype=np.float32)
                return state, -0.1, False, self.phase >= 2, {
                    "reward_cage_pose6": state[:6].copy(),
                    "next_cage_pose6": state[:6].copy(),
                    "action_state_relative_error6": state[:6].copy(),
                    "next_state_relative_error6": state[:6].copy(),
                    "reward_target_contract": "same_cage_pose_as_action_state",
                    "executed_normalized_action": np.asarray(action).copy(),
                    "rollout_boundary": self.phase == 1,
                }

        config = copy.deepcopy(self.config["ppo"])
        config["hidden_dims"] = [16]
        trainer = PPO2Trainer(config, device="cpu")
        rollout = collect_complete_rollout(
            trainer, RecoveryEnvironment(), minimum_steps=2
        )
        self.assertEqual(rollout.dones.tolist(), [1.0, 1.0])

    def test_c7_reset_contract_rejects_non_c7_trajectory_stage(self):
        from cage_system.learning.expert_training import _assert_c7_reset

        valid = {
            "privileged_reset_window_number": 3,
            "first_policy_window_number": 4,
        }
        _assert_c7_reset(valid, self.config)
        invalid = dict(valid)
        invalid["privileged_reset_window_number"] = 2
        with self.assertRaises(RuntimeError):
            _assert_c7_reset(invalid, self.config)

    def test_random_visualization_segment_is_contiguous_and_reproducible(self):
        from cage_system.system_visualization import (
            _random_contiguous_solution_segment,
            _split_time_contiguous_solutions,
        )

        solutions = [(index, np.full(6, index, dtype=float)) for index in range(20)]
        first = _random_contiguous_solution_segment(
            solutions, 5, np.random.RandomState(73)
        )
        second = _random_contiguous_solution_segment(
            solutions, 5, np.random.RandomState(73)
        )
        self.assertEqual([item[0] for item in first], [item[0] for item in second])
        self.assertEqual(len(first), 5)
        self.assertEqual(
            np.diff([item[0] for item in first]).tolist(),
            [1, 1, 1, 1],
        )
        timestamps = np.array([0.00, 0.05, 0.10, 0.35, 0.40, 0.45])
        timestamped_solutions = [
            (index, np.zeros(6, dtype=float)) for index in range(len(timestamps))
        ]
        runs = _split_time_contiguous_solutions(
            timestamped_solutions, timestamps, maximum_gap_s=0.08
        )
        self.assertEqual([[item[0] for item in run] for run in runs], [[0, 1, 2], [3, 4, 5]])

    def test_mujoco_environment_contract_with_six_joint_fixture(self):
        try:
            from cage_system.learning.ppo2 import PPO2Trainer
            from cage_system.learning.train_ppo import collect_complete_rollout
            from cage_system.sim.mujoco_env import MujocoUR5eCageEnv
            from cage_system.sim.target_playback import TargetSequenceProvider
        except ImportError as error:
            self.skipTest(str(error))
        timestamps = np.array([0.0, 0.1])
        poses = np.array(
            [
                [0.40, 0.0, 0.25, 0.0, 0.0, 0.0],
                [0.60, 0.0, 0.25, 0.0, 0.0, 0.0],
            ]
        )
        sequence = TargetSequenceProvider(timestamps, poses)
        model = Path("tests/assets/six_joint_contract.xml").resolve()
        environment = MujocoUR5eCageEnv(
            str(model), [sequence], self.config, tcp_site_name="attachment_site"
        )
        observation, _ = environment.reset(seed=4)
        self.assertEqual(observation.shape, (18,))
        expert = environment.expert_action()
        self.assertEqual(expert.normalized_action.shape, (6,))
        self.assertTrue(np.all(np.isfinite(expert.normalized_action)))
        self.assertTrue(np.all(np.abs(expert.normalized_action) <= 1.0))
        random_state, random_info = environment.reset_random_local_expert(
            np.random.RandomState(9),
            self.config["expert_training"]["random_demonstration"],
        )
        self.assertEqual(random_info["source"], "random_local_static_cage")
        self.assertEqual(random_state.shape, (18,))
        self.assertTrue(np.all(np.isfinite(random_state)))
        expert = environment.expert_action()
        random_next, _, random_terminated, _, random_step_info = (
            environment.step_fixed_expert_target(
                expert.normalized_action,
                self.config["expert_training"]["random_demonstration"],
            )
        )
        self.assertEqual(random_next.shape, (18,))
        self.assertFalse(random_terminated)
        self.assertEqual(
            random_step_info["target_source"], "random_local_static_cage"
        )
        observation, _ = environment.reset(seed=4)
        next_observation, reward, terminated, truncated, info = environment.step(
            np.zeros(6)
        )
        self.assertAlmostEqual(environment.decision_dt, 0.05)
        self.assertAlmostEqual(environment.elapsed, 0.05)
        self.assertEqual(next_observation.shape, (18,))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertIn("position_error_m", info)
        np.testing.assert_allclose(
            info["action_state_relative_error6"], observation[:6]
        )
        np.testing.assert_allclose(
            info["next_state_relative_error6"], next_observation[:6]
        )
        self.assertEqual(
            info["reward_target_contract"], "same_cage_pose_as_action_state"
        )
        self.assertFalse(
            np.allclose(info["reward_cage_pose6"], info["next_cage_pose6"])
        )
        tcp_position = environment.data.site_xpos[environment.tcp_site_id].copy()
        self.assertAlmostEqual(
            info["position_error_m"],
            np.linalg.norm(tcp_position - info["reward_cage_pose6"][:3]),
        )
        trainer_config = copy.deepcopy(self.config["ppo"])
        trainer_config["hidden_dims"] = [16]
        trainer = PPO2Trainer(trainer_config, device="cpu")
        rollout = collect_complete_rollout(trainer, environment, minimum_steps=1)
        self.assertEqual(rollout.states.shape[1], 18)
        self.assertGreater(trainer.state_normalizer.count, 2.0)

    def test_eye_on_hand_unprojection_uses_camera_minus_z_convention(self):
        intrinsics = CameraIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0)
        depth = np.zeros((3, 3), dtype=float)
        depth[1, 1] = 1.25
        mask = depth > 0.0
        points = unproject_depth_mask(
            depth,
            mask,
            intrinsics,
            np.array([0.2, -0.1, 0.7]),
            np.eye(3),
        )
        np.testing.assert_allclose(points[0], [0.2, -0.1, -0.55])

    def test_rgb_detector_does_not_read_simulation_geometry_ids(self):
        rgb = np.zeros((30, 40, 3), dtype=np.uint8)
        rgb[8:20, 12:27, 0] = 230
        rgb[8:20, 12:27, 1] = 20
        mask, confidence = RedColorMaskDetector().detect(rgb)
        self.assertEqual(int(mask.sum()), 12 * 15)
        self.assertGreater(confidence, 0.5)

    def test_official_ur5e_eye_on_hand_scene_contract(self):
        try:
            import mujoco
        except ImportError as error:
            self.skipTest(str(error))
        path = Path(
            "assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml"
        ).resolve()
        model = mujoco.MjModel.from_xml_path(str(path))
        self.assertEqual(model.nu, 6)
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_on_hand"), 0
        )
        target_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap"
        )
        self.assertGreaterEqual(model.body_mocapid[target_body], 0)

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        link6 = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
        )
        cage = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "cage_gripper"
        )
        tcp = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        camera = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_on_hand"
        )
        self.assertEqual(model.body_parentid[cage], link6)
        self.assertEqual(model.cam_bodyid[camera], cage)
        link6_extension = data.xmat[link6].reshape(3, 3)[:, 1]
        cage_x = data.xmat[cage].reshape(3, 3)[:, 0]
        tcp_x = data.site_xmat[tcp].reshape(3, 3)[:, 0]
        np.testing.assert_allclose(cage_x, link6_extension, atol=1e-8)
        np.testing.assert_allclose(tcp_x, link6_extension, atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(data.xmat[link6].reshape(3, 3)), 1.0)
        self.assertAlmostEqual(np.linalg.det(data.xmat[cage].reshape(3, 3)), 1.0)
        self.assertAlmostEqual(np.linalg.det(data.site_xmat[tcp].reshape(3, 3)), 1.0)
        self.assertAlmostEqual(np.linalg.det(data.cam_xmat[camera].reshape(3, 3)), 1.0)
        self.assertLess(self.config["gripper"]["internal_region_endpoint_frame"]["mouth_x"], 0.060)
        cage_y = data.xmat[cage].reshape(3, 3)[:, 1]
        for geom_name in ("cage_left_finger", "cage_right_finger"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            geom_rotation = data.geom_xmat[geom].reshape(3, 3)
            # The box is thin along local y, so its two broad leaf faces are
            # x-z planes whose normals are parallel to the Cage y axis.
            self.assertLess(model.geom_size[geom, 1], model.geom_size[geom, 0])
            self.assertLess(model.geom_size[geom, 1], model.geom_size[geom, 2])
            self.assertAlmostEqual(abs(np.dot(geom_rotation[:, 1], cage_y)), 1.0)


if __name__ == "__main__":
    unittest.main()
