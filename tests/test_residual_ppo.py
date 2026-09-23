import copy
from types import SimpleNamespace

import numpy as np
import torch

from cage_system.config import load_config
from cage_system.learning.ppo2 import PPO2Trainer
from cage_system.learning.residual_ppo import (
    ResidualPPOTrainer,
    ResidualRolloutBatch,
    load_frozen_control_policy,
)
from cage_system.learning.train_residual_ppo import (
    assert_disjoint_trajectory_sets,
    validation_comparison,
)


def _source_checkpoints(tmp_path, config):
    ppo = copy.deepcopy(config["ppo"])
    ppo["hidden_dims"] = [16]
    trainer = PPO2Trainer(ppo, device="cpu")
    trainer.state_normalizer.update(
        np.linspace(-0.4, 0.7, 18 * 32, dtype=np.float32).reshape(32, 18)
    )
    base = tmp_path / "dagger_04.pt"
    trainer.save(str(base))
    trainer.reset_critic()
    trainer.state_normalizer.update(
        np.linspace(-0.8, 0.9, 18 * 24, dtype=np.float32).reshape(24, 18)
    )
    critic = tmp_path / "critic_warmup.pt"
    trainer.save(str(critic))
    return base, critic


def _residual_config():
    config = load_config("configs/production_curved_residual_ppo.yaml")
    config = copy.deepcopy(config)
    config["ppo"]["hidden_dims"] = [16]
    config["ppo"]["minibatch_size"] = 4
    config["residual_training"]["hidden_dims"] = [16]
    config["residual_training"]["actor_update_epochs"] = 2
    config["residual_training"]["critic_update_epochs"] = 2
    config["residual_training"]["critic_alignment_epochs"] = 2
    config["residual_training"]["critic_alignment_batch_size"] = 4
    return config


def test_zero_residual_preserves_frozen_base_and_checkpoint_is_portable(tmp_path):
    config = _residual_config()
    base, critic = _source_checkpoints(tmp_path, config)
    trainer = ResidualPPOTrainer(config, str(base), str(critic), device="cpu")
    raw_state = np.linspace(-0.2, 0.3, 18, dtype=np.float32)
    decision = trainer.decide(raw_state, deterministic=True)
    np.testing.assert_allclose(decision.residual_action, np.zeros(6), atol=1e-8)
    np.testing.assert_allclose(
        decision.executed_action, decision.base_action, atol=1e-7
    )
    assert all(not item.requires_grad for item in trainer.base_actor.parameters())
    actor_ids = {id(item) for item in trainer.residual_actor.parameters()}
    critic_ids = {id(item) for item in trainer.critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)

    checkpoint = tmp_path / "residual.pt"
    trainer.save(str(checkpoint), include_optimizers=True)
    frozen = load_frozen_control_policy(str(checkpoint), device="cpu")
    with torch.inference_mode():
        action, _, value, _ = frozen.sample(
            torch.from_numpy(raw_state).unsqueeze(0), deterministic=True
        )
    torch.testing.assert_close(
        action[0], torch.from_numpy(decision.executed_action), atol=2e-7, rtol=0.0
    )
    assert np.isfinite(float(value.item()))


def test_actor_kl_stop_does_not_cancel_independent_critic_update(tmp_path):
    config = _residual_config()
    config["ppo"]["target_kl"] = 1e-6
    base, critic = _source_checkpoints(tmp_path, config)
    trainer = ResidualPPOTrainer(config, str(base), str(critic), device="cpu")
    rng = np.random.RandomState(14)
    raw_states = rng.normal(size=(8, 18)).astype(np.float32)
    decisions = [trainer.decide(item, deterministic=False) for item in raw_states]
    critic_before = {
        key: value.detach().clone() for key, value in trainer.critic.state_dict().items()
    }
    batch = ResidualRolloutBatch(
        states=np.asarray([trainer.normalize_rl(item) for item in raw_states]),
        residual_actions=np.asarray([item.residual_action for item in decisions]),
        base_actions=np.asarray([item.base_action for item in decisions]),
        preclip_actions=np.asarray([item.preclip_action for item in decisions]),
        executed_actions=np.asarray([item.executed_action for item in decisions]),
        rewards=rng.normal(size=8).astype(np.float32),
        next_states=np.asarray([trainer.normalize_rl(item) for item in raw_states]),
        dones=np.asarray([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        old_log_densities=np.asarray(
            [item.residual_log_density + 10.0 for item in decisions],
            dtype=np.float32,
        ),
        old_values=np.asarray([item.value for item in decisions], dtype=np.float32),
        next_values=np.zeros(8, dtype=np.float32),
    )
    metrics = trainer.update(batch)
    assert metrics["actor_kl_early_stopped"]
    assert metrics["critic_epochs_completed"] == 2
    assert any(
        not torch.equal(critic_before[key], value)
        for key, value in trainer.critic.state_dict().items()
    )


def test_trajectory_disjointness_uses_curve_content_not_reused_track_ids():
    first = SimpleNamespace(
        timestamps=np.asarray([0.0, 0.05]),
        true_positions=np.asarray([[0.4, 0.0, 0.3], [0.41, 0.0, 0.3]]),
    )
    second = SimpleNamespace(
        timestamps=np.asarray([0.0, 0.05]),
        true_positions=np.asarray([[0.4, 0.0, 0.3], [0.4, 0.01, 0.3]]),
    )
    summary = assert_disjoint_trajectory_sets(
        {"expert": [first], "residual": [second]}
    )
    assert summary["trajectory_sets_disjoint"]
    try:
        assert_disjoint_trajectory_sets(
            {"expert": [first], "residual": [first]}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Exact trajectory reuse was not rejected")


def test_normalized_residual_reward_is_bounded_and_reports_progress_components():
    try:
        from cage_system.sim.mujoco_env import MujocoUR5eCageEnv
        from cage_system.sim.target_playback import TargetSequenceProvider
    except ImportError:
        return
    config = _residual_config()
    sequence = TargetSequenceProvider(
        np.asarray([0.0, 0.1]),
        np.asarray(
            [
                [0.45, 0.0, 0.30, 0.0, 0.0, 0.0],
                [0.46, 0.0, 0.30, 0.0, 0.0, 0.0],
            ]
        ),
    )
    environment = MujocoUR5eCageEnv(
        "tests/assets/six_joint_contract.xml",
        [sequence],
        config,
        tcp_site_name="attachment_site",
    )
    try:
        environment.reset(seed=2)
        _, reward, _, _, info = environment.step(np.zeros(6, dtype=np.float32))
    finally:
        environment.close() if hasattr(environment, "close") else None
    assert np.isfinite(reward)
    assert info["reward_mode"] == "normalized_cage_tracking_v2"
    assert 0.0 <= info["tracking_quality_before"] <= 1.0
    assert 0.0 <= info["tracking_quality_after"] <= 1.0
    assert info["tracking_quality_mapping"] == "inverse_quadratic"
    assert np.isclose(info["tracking_reward"], info["tracking_quality_after"])
    assert "progress_reward" in info
    assert "safety_cost" in info


def test_residual_resume_rejects_changed_reward_specification(tmp_path):
    config = _residual_config()
    base, critic = _source_checkpoints(tmp_path, config)
    trainer = ResidualPPOTrainer(config, str(base), str(critic), device="cpu")
    checkpoint = tmp_path / "residual.pt"
    trainer.save(str(checkpoint), include_optimizers=True)

    changed = copy.deepcopy(config)
    changed["ppo"]["reward_progress_weight"] += 0.25
    changed_trainer = ResidualPPOTrainer(
        changed, str(base), str(critic), device="cpu"
    )
    try:
        changed_trainer.restore(str(checkpoint))
    except ValueError as error:
        assert "reward specification" in str(error)
    else:
        raise AssertionError("Changed residual reward incorrectly resumed")


def test_validation_guard_rejects_reward_gain_that_degrades_pose_accuracy():
    base = {
        "reward_mean": -1.0,
        "position_error_mean_mm": 20.0,
        "orientation_error_mean_deg": 10.0,
        "pose_success_rate": 0.5,
        "visual_escapes": 1,
        "true_visual_escapes": 1,
        "collision_steps": 0,
    }
    residual = {
        **base,
        "reward_mean": -0.9,
        "position_error_mean_mm": 25.0,
    }
    comparison = validation_comparison(base, residual)
    assert comparison["reward_mean_gain"] > 0.0
    assert not comparison["passes_strict_base_nondegradation_guard"]

    comparison = validation_comparison(base, dict(base))
    assert comparison["passes_strict_base_nondegradation_guard"]
