import copy

from pathlib import Path

import numpy as np

import pytest

import torch

from cage_system.config import load_config

from cage_system.learning.train_jailer import _append_movement_trend_bins, _export_movement_trends, _movement_trend_bins, _prepare_success_database

from cage_system.learning.ppo2 import RolloutBatch

from cage_system.learning.jailer_policy import JailerPPO2Trainer


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "tests" / "assets" / "six_joint_contract.xml"

def _final_synthetic_trainer_and_rollout():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_joint_v14_full.yaml")
    )
    settings = copy.deepcopy(config["jailer"])
    settings["ppo"]["hidden_dims"] = [16]
    settings["ppo"]["minibatch_size"] = 8
    settings["ppo"]["actor_update_epochs"] = 1
    settings["ppo"]["critic_update_epochs"] = 1
    settings["ppo"]["target_kl"] = 1.0
    trainer = JailerPPO2Trainer(settings["ppo"], device="cpu")
    stage = settings["curriculum"]["stages"][0]
    trainer.set_curriculum_task(stage["task"])
    trainer.prepare_training_stage(stage, entering=True)

    torch.manual_seed(137)
    rng = np.random.RandomState(137)
    states = rng.normal(size=(8, 18)).astype(np.float32)
    next_states = rng.normal(size=(8, 18)).astype(np.float32)
    with torch.no_grad():
        actions, old_log_densities, old_values, _ = trainer.model.sample(
            torch.from_numpy(states)
        )
        next_values = trainer.model.critic(torch.from_numpy(next_states))
    dones = np.asarray([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
    rollout = RolloutBatch(
        states=states,
        actions=actions.numpy(),
        rewards=np.linspace(-0.8, 1.1, 8, dtype=np.float32),
        next_states=next_states,
        dones=dones,
        old_log_densities=old_log_densities.numpy(),
        old_values=old_values.numpy(),
        next_values=next_values.numpy(),
        bootstrap_dones=dones.copy(),
    )
    return trainer, rollout

def test_final_actor_and_critic_adam_parameters_are_strictly_isolated():
    trainer, _ = _final_synthetic_trainer_and_rollout()
    actor_parameters = {id(item) for item in trainer.model.actor.parameters()}
    critic_parameters = {id(item) for item in trainer.model.critic.parameters()}
    actor_optimizer_parameters = {
        id(item)
        for group in trainer.actor_optimizer.param_groups
        for item in group["params"]
    }
    critic_optimizer_parameters = {
        id(item)
        for group in trainer.critic_optimizer.param_groups
        for item in group["params"]
    }
    assert actor_optimizer_parameters == actor_parameters
    assert critic_optimizer_parameters == critic_parameters
    assert actor_optimizer_parameters.isdisjoint(critic_optimizer_parameters)
    assert [group["lr"] for group in trainer.actor_optimizer.param_groups] == [
        pytest.approx(trainer.config["joint_actor_learning_rate"])
    ]
    assert [group["lr"] for group in trainer.critic_optimizer.param_groups] == [
        pytest.approx(trainer.config["joint_critic_learning_rate"])
    ]
    assert not hasattr(trainer, "optimizer")

def test_final_synthetic_rollout_updates_both_independent_adam_optimizers():
    trainer, rollout = _final_synthetic_trainer_and_rollout()
    actor_before = {
        name: value.detach().clone()
        for name, value in trainer.model.actor.named_parameters()
    }
    critic_before = {
        name: value.detach().clone()
        for name, value in trainer.model.critic.named_parameters()
    }

    metrics = trainer.update(rollout)

    assert any(
        not torch.equal(actor_before[name], value)
        for name, value in trainer.model.actor.named_parameters()
    )
    assert any(
        not torch.equal(critic_before[name], value)
        for name, value in trainer.model.critic.named_parameters()
    )
    assert trainer.actor_optimizer.state
    assert trainer.critic_optimizer.state
    assert set(trainer.actor_optimizer.state) == set(
        trainer.model.actor.parameters()
    )
    assert set(trainer.critic_optimizer.state) == set(
        trainer.model.critic.parameters()
    )
    assert {
        int(state["step"].item())
        for state in trainer.actor_optimizer.state.values()
    } == {1}
    assert {
        int(state["step"].item())
        for state in trainer.critic_optimizer.state.values()
    } == {1}
    assert metrics["actor_epochs_completed"] == 1
    assert metrics["critic_epochs_completed"] == 1
    assert metrics["actor_minibatches_completed"] == 1
    assert metrics["critic_minibatches_completed"] == 1
    assert metrics["optimization_mode"] == "independent_actor_critic_ppo"
    assert not hasattr(trainer, "optimizer")

def test_final_critic_alignment_preserves_actor_and_actor_adam_state():
    trainer, rollout = _final_synthetic_trainer_and_rollout()
    trainer.update(rollout)
    actor_before = {
        name: value.detach().clone()
        for name, value in trainer.model.actor.named_parameters()
    }
    critic_before = {
        name: value.detach().clone()
        for name, value in trainer.model.critic.named_parameters()
    }
    actor_adam_before = {
        parameter: {
            name: value.detach().clone() if torch.is_tensor(value) else value
            for name, value in state.items()
        }
        for parameter, state in trainer.actor_optimizer.state.items()
    }

    metrics = trainer.align_critic(rollout)

    assert metrics["optimization_mode"] == "critic_only_alignment"
    assert metrics["actor_epochs_completed"] == 0
    for name, value in trainer.model.actor.named_parameters():
        assert torch.equal(value, actor_before[name])
    assert set(trainer.actor_optimizer.state) == set(actor_adam_before)
    for parameter, state in trainer.actor_optimizer.state.items():
        expected = actor_adam_before[parameter]
        assert state.keys() == expected.keys()
        for name, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, expected[name])
            else:
                assert value == expected[name]
    assert any(
        not torch.equal(critic_before[name], value)
        for name, value in trainer.model.critic.named_parameters()
    )

def test_movement_trend_bins_export_csv_and_pdf(tmp_path):
    database = tmp_path / "movement.sqlite3"
    connection = _prepare_success_database(database, reset=True)
    stage = {
        "name": "pose_test",
        "task": "pose",
        "curriculum_phase": "pose_bridge",
    }
    infos = [
        {
            "position_error_m": 0.012,
            "orientation_error_deg": 3.0,
            "tcp_50ms_net_displacement_m": 0.004,
            "tcp_100ms_net_displacement_m": 0.007,
        },
        {
            "position_error_m": 0.014,
            "orientation_error_deg": 3.5,
            "tcp_50ms_net_displacement_m": 0.006,
            "tcp_100ms_net_displacement_m": 0.009,
        },
    ]
    rows = _movement_trend_bins(
        infos,
        stage,
        {
            "movement_trend_position_bin_width_m": 0.005,
            "movement_trend_orientation_bin_width_deg": 2.0,
        },
    )
    _append_movement_trend_bins(connection, rows)
    _append_movement_trend_bins(connection, rows)
    csv_path = tmp_path / "movement.csv.gz"
    pdf_path = tmp_path / "movement.pdf"
    _export_movement_trends(connection, csv_path, pdf_path)
    count = connection.execute(
        "SELECT SUM(move50_count) FROM movement_trend_bins "
        "WHERE error_axis='position_m'"
    ).fetchone()[0]
    connection.close()
    assert count == 4
    assert csv_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
