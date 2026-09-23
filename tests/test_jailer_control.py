import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from cage_system.config import load_config
from cage_system.control.constrained_differential_ik import (
    simulate_time_constrained_ik,
)
from cage_system.learning.train_jailer import (
    JAILER_ACTION_CONTRACT,
    JAILER_CHECKPOINT_FORMAT,
    JAILER_LEGACY_POSITION_MASKED_STATE_CONTRACT,
    JAILER_PHASED_STATE_CONTRACT,
    JailerCurriculum,
    _apply_stage_entry_transition,
    _append_movement_trend_bins,
    _append_success_cases,
    _configured_state_contract,
    _export_movement_trends,
    _export_success_csv,
    _movement_trend_bins,
    _plot_success_curves,
    _prepare_pose_transition,
    _prepare_success_database,
    _warm_start_actor,
    collect_logical_rollout,
)
from cage_system.learning.ppo2 import PPO2Trainer, RolloutBatch
from cage_system.learning.jailer_policy import (
    JOINT_PPO_ACTION,
    JOINT_REFERENCE_ACTION,
    SEPARATE_RESIDUAL_ACTION,
    JailerPPO2Trainer,
    joint_action_gain,
    pose_residual_gain,
)
from cage_system.sim.jailer_env import (
    JailerLogicalVectorEnv,
    generate_initial_joint_pool,
)
from cage_system.sim.ur5e_control import UR5eControlModel


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "tests" / "assets" / "six_joint_contract.xml"


def test_jailer_config_is_100hz_40_slot_position_then_pose():
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_curriculum.yaml"))
    jailer = config["jailer"]
    assert jailer["control_rate_hz"] == 100.0
    assert jailer["logical_slots"] == 40
    assert jailer["reward_step_quality_baseline"] == 1.0
    assert jailer["success_bonus"] > 0.0
    assert jailer["reward_position_progress_weight"] == 1.0
    assert jailer["reward_position_progress_reference_m"] == 0.005
    assert jailer["reward_near_target_position_m"] == 0.030
    assert jailer["reward_near_target_orientation_deg"] == 15.0
    tasks = [stage["task"] for stage in jailer["curriculum"]["stages"]]
    assert tasks[:6] == ["position"] * 6
    assert tasks[6:] == ["pose"] * 12
    assert jailer["ppo"]["state_dim"] == 18
    assert jailer["ppo"]["action_dim"] == 6
    assert jailer["ppo"]["updates"] == 4000
    bridge_stages = jailer["curriculum"]["stages"][6:12]
    pose_stages = jailer["curriculum"]["stages"][12:]
    position_stages = jailer["curriculum"]["stages"][:6]
    assert [stage["position_error_range_m"] for stage in position_stages] == [
        [0.006, 0.020],
        [0.006, 0.035],
        [0.006, 0.055],
        [0.006, 0.080],
        [0.006, 0.110],
        [0.006, 0.150],
    ]
    assert [stage["rollout_steps_per_slot"] for stage in position_stages] == [
        60,
        60,
        70,
        80,
        100,
        130,
    ]
    assert [stage["rollout_steps_per_slot"] for stage in pose_stages] == [
        80,
        90,
        110,
        130,
        150,
        180,
    ]
    assert [stage["rollout_steps_per_slot"] for stage in bridge_stages] == [
        80,
        90,
        100,
        110,
        120,
        130,
    ]
    assert [stage["success_position_m"] for stage in bridge_stages] == [
        0.006,
        0.005,
        0.004,
        0.004,
        0.0035,
        0.0028,
    ]
    assert [stage["success_orientation_deg"] for stage in bridge_stages] == [
        12.0,
        8.0,
        4.0,
        3.5,
        3.0,
        2.2,
    ]
    assert [stage["promotion_position_mean_m"] for stage in bridge_stages] == [
        0.006,
        0.0055,
        0.005,
        0.0045,
        0.004,
        0.0032,
    ]
    assert [stage["promotion_orientation_mean_deg"] for stage in bridge_stages] == [
        7.0,
        6.0,
        4.5,
        3.7,
        3.2,
        2.5,
    ]
    assert [stage["action_velocity_scale"] for stage in bridge_stages] == [
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
    ]
    assert [stage["reward_orientation_quality_weight"] for stage in position_stages] == [
        0.0,
    ] * 6
    assert [stage["reward_orientation_quality_weight"] for stage in pose_stages] == [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.50,
    ]
    assert [stage["action_velocity_scale"] for stage in pose_stages] == [
        0.10,
        0.15,
        0.25,
        0.40,
        0.65,
        1.00,
    ]
    expected_position_only_success = [
        0.0015,
        0.002,
        0.0025,
        0.003,
        0.0035,
        0.004,
    ]
    expected_pose_position_success = [
        0.002,
        0.0025,
        0.003,
        0.0035,
        0.004,
        0.0045,
    ]
    expected_position_only_promotion = [
        0.0025,
        0.003,
        0.0035,
        0.0045,
        0.0055,
        0.008,
    ]
    expected_pose_position_promotion = [
        0.0025,
        0.003,
        0.0035,
        0.0045,
        0.0055,
        0.0065,
    ]
    assert [stage["success_position_m"] for stage in position_stages] == (
        expected_position_only_success
    )
    assert [stage["success_position_m"] for stage in pose_stages] == (
        expected_pose_position_success
    )
    assert [
        stage["promotion_position_mean_m"] for stage in position_stages
    ] == expected_position_only_promotion
    assert [
        stage["promotion_position_mean_m"] for stage in pose_stages
    ] == expected_pose_position_promotion
    assert [stage["promotion_success_rate"] for stage in position_stages] == [
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
    ]
    assert position_stages[-1]["promotion_success_rate_strict"] is True
    assert all(stage["promotion_success_rate"] == 0.80 for stage in pose_stages)
    assert all(
        stage["promotion_success_rate_strict"] is True
        for stage in pose_stages
    )
    assert [stage["orientation_error_range_deg"] for stage in pose_stages] == [
        [1.0, 5.0],
        [1.0, 10.0],
        [1.0, 16.0],
        [1.0, 24.0],
        [1.0, 35.0],
        [1.0, 45.0],
    ]
    assert [stage["position_error_range_m"] for stage in pose_stages] == [
        [0.006, 0.025],
        [0.006, 0.040],
        [0.006, 0.060],
        [0.006, 0.090],
        [0.006, 0.120],
        [0.006, 0.150],
    ]
    assert [stage["success_orientation_deg"] for stage in pose_stages] == [
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        7.0,
    ]
    assert [
        stage["promotion_orientation_mean_deg"] for stage in pose_stages
    ] == [
        2.5,
        3.5,
        4.5,
        5.5,
        7.0,
        9.0,
    ]


def test_v9_config_uses_drained_rollouts_and_separate_pose_residual():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_curriculum_v9.yaml")
    )
    jailer = config["jailer"]
    assert jailer["rollout_boundary_mode"] == (
        "stop_spawn_then_drain_active_tasks_v1"
    )
    assert jailer["ppo"]["action_composition_mode"] == SEPARATE_RESIDUAL_ACTION
    assert jailer["ppo"]["pose_residual_gain_orientation_error_deg"] == [
        0.0,
        5.0,
        10.0,
        16.0,
        24.0,
        35.0,
        45.0,
    ]
    pose_stages = jailer["curriculum"]["stages"][12:]
    assert [stage["action_velocity_scale"] for stage in pose_stages] == [
        0.10,
        0.15,
        0.25,
        0.40,
        0.65,
        1.00,
    ]


def test_pose_residual_gain_is_piecewise_linear_and_stage_capped():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_curriculum_v9.yaml")
    )["jailer"]["ppo"]
    values = pose_residual_gain(
        np.asarray([0.0, 5.0, 7.5, 10.0, 20.0, 45.0, 60.0]),
        config,
        stage_cap=1.0,
    )
    assert values == pytest.approx(
        [0.10, 0.10, 0.125, 0.15, 0.325, 1.0, 1.0]
    )
    capped = pose_residual_gain(
        np.asarray([5.0, 24.0, 45.0]), config, stage_cap=0.25
    )
    assert capped == pytest.approx([0.10, 0.25, 0.25])


def test_v9_position_action_is_unscaled_and_only_pose_residual_gets_gain():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_curriculum_v9.yaml")
    )
    trainer = JailerPPO2Trainer(config["jailer"]["ppo"], device="cpu")
    actor = trainer.model.actor
    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.fill_(np.arctanh(0.5))
        actor.pose_residual_head.weight.zero_()
        actor.pose_residual_head.bias.fill_(np.arctanh(0.5))
        actor.log_std.fill_(-5.0)
        actor.pose_residual_log_std.fill_(-5.0)
    normalized = torch.zeros((2, 18), dtype=torch.float32)
    raw = np.zeros((2, 18), dtype=np.float32)
    raw[:, 3] = torch.deg2rad(torch.tensor([5.0, 24.0])).numpy()

    position_stage = config["jailer"]["curriculum"]["stages"][0]
    trainer.set_curriculum_task("position")
    position = trainer.sample_for_environment(
        normalized, raw, position_stage, deterministic=True
    )
    assert position["executed_action"].detach().numpy() == pytest.approx(
        np.full((2, 6), 0.5)
    )

    pose_stage = config["jailer"]["curriculum"]["stages"][15]
    assert pose_stage["action_velocity_scale"] == pytest.approx(0.40)
    trainer.set_curriculum_task("pose")
    pose = trainer.sample_for_environment(
        normalized, raw, pose_stage, deterministic=True
    )
    assert pose["residual_gain"] == pytest.approx([0.10, 0.40])
    assert pose["policy_action"].detach().numpy() == pytest.approx(
        np.full((2, 6), 0.5)
    )
    assert pose["executed_action"].detach().numpy() == pytest.approx(
        np.vstack([np.full(6, 0.55), np.full(6, 0.70)])
    )


def test_ur5e_binding_resolves_named_contract_and_five_physics_steps():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    assert ur5e.contract.joint_names == tuple(config["robot"]["joint_names"])
    assert ur5e.contract.tcp_site_name == "attachment_site"
    assert ur5e.contract.physics_timestep_s == pytest.approx(0.002)
    assert ur5e.contract.physics_steps_per_control == 5


def test_position_and_pose_curricula_share_the_same_full_pose_observation():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool_arrays = generate_initial_joint_pool(ur5e, settings, 2, seed=7)
    environment = JailerLogicalVectorEnv(
        ur5e, settings, pool_arrays["joint_positions_rad"], seed=8
    )
    slot = environment.slots[0]
    current = ur5e.tcp_pose6(slot.data)
    target_rotation = (
        Rotation.from_rotvec(np.array([0.2, 0.0, 0.0]))
        * Rotation.from_rotvec(current[3:])
    )
    slot.target_pose6 = np.concatenate([current[:3], target_rotation.as_rotvec()])
    position_stage = copy.deepcopy(settings["curriculum"]["stages"][0])
    pose_stage = copy.deepcopy(position_stage)
    pose_stage["task"] = "pose"
    position_state = environment._state(slot, position_stage)
    pose_state = environment._state(slot, pose_stage)
    assert np.linalg.norm(position_state[3:6]) > 0.1
    assert np.allclose(position_state, pose_state)


def test_orientation_marker_quality_does_not_double_count_tcp_translation():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=61)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=62)
    stage = settings["curriculum"]["stages"][1]
    slot = environment.slots[0]
    current = ur5e.tcp_pose6(slot.data)
    slot.target_pose6 = current.copy()
    translated = current.copy()
    translated[:3] += np.array([0.05, -0.02, 0.01])
    _, _, same_orientation_quality, marker_mse = environment._pose_quality(
        slot,
        stage,
        translated,
        position_error=float(np.linalg.norm(translated[:3] - current[:3])),
    )
    assert same_orientation_quality == pytest.approx(1.0)
    assert marker_mse == pytest.approx(0.0, abs=1e-12)

    rotated = translated.copy()
    rotated[3:] = (
        Rotation.from_rotvec(np.array([0.2, 0.0, 0.0]))
        * Rotation.from_rotvec(current[3:])
    ).as_rotvec()
    _, _, rotated_quality, rotated_mse = environment._pose_quality(
        slot,
        stage,
        rotated,
        position_error=float(np.linalg.norm(translated[:3] - current[:3])),
    )
    assert rotated_quality < 1.0
    assert rotated_mse > 0.0


def test_legacy_position_actor_warm_start_neutralizes_untrained_orientation(tmp_path):
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=41)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=42)
    source = PPO2Trainer(settings["ppo"], device="cpu")
    first_source_layer = next(
        module
        for module in source.model.actor.features
        if isinstance(module, torch.nn.Linear)
    )
    with torch.no_grad():
        first_source_layer.weight[:, 3:6].fill_(1.0)
    source.state_normalizer.variance[3:6] = 1e-12
    checkpoint_path = tmp_path / "legacy_position.pt"
    torch.save(
        {
            "format": JAILER_CHECKPOINT_FORMAT,
            "state_contract": JAILER_LEGACY_POSITION_MASKED_STATE_CONTRACT,
            "action_contract": JAILER_ACTION_CONTRACT,
            "state_dim": 18,
            "action_dim": 6,
            "model_state": source.model.state_dict(),
            "state_normalizer": source.state_normalizer.state_dict(),
            "completed_updates": 10,
            "ur5e_contract": environment.ur5e.contract_dict(),
        },
        checkpoint_path,
    )
    target = JailerPPO2Trainer(settings["ppo"], device="cpu")
    _warm_start_actor(
        checkpoint_path,
        target,
        environment,
        legacy_orientation_input_std_deg=10.0,
    )
    first_target_layer = next(
        module
        for module in target.model.actor.features
        if isinstance(module, torch.nn.Linear)
    )
    assert torch.all(first_target_layer.weight[:, 3:6] == 0.0)
    assert np.allclose(
        target.state_normalizer.variance[3:6],
        np.deg2rad(10.0) ** 2,
    )
    assert target.jailer_training_initialization["legacy_orientation_adapter"] is True


def test_pose_transition_raises_exploration_without_changing_actor_mean_weights():
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_curriculum.yaml"))
    trainer = JailerPPO2Trainer(config["jailer"]["ppo"], device="cpu")
    with torch.no_grad():
        trainer.model.actor.log_std.fill_(-2.5)
    mean_weights_before = trainer.model.actor.mean_head.weight.detach().clone()
    metrics = _prepare_pose_transition(trainer, minimum_log_std=-1.2)
    assert torch.allclose(
        trainer.model.actor.log_std,
        torch.full_like(trainer.model.actor.log_std, -1.2),
    )
    assert torch.equal(
        trainer.model.actor.mean_head.weight,
        mean_weights_before,
    )
    assert metrics["pose_transition_log_std_before"] == pytest.approx([-2.5] * 6)
    assert metrics["pose_transition_log_std_after"] == pytest.approx([-1.2] * 6)


def test_zero_initialized_pose_residual_preserves_position_actor_at_switch():
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_curriculum.yaml"))
    trainer = JailerPPO2Trainer(config["jailer"]["ppo"], device="cpu")
    states = torch.randn(16, 18)
    trainer.set_curriculum_task("position")
    position_means = trainer.model.actor(states)[0].detach().clone()
    trainer.set_curriculum_task("pose")
    pose_means = trainer.model.actor(states)[0].detach().clone()
    assert torch.equal(position_means, pose_means)
    assert all(
        not parameter.requires_grad
        for parameter in trainer.model.actor.features.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in trainer.model.actor.pose_residual_head.parameters()
    )
    assert [group["lr"] for group in trainer.optimizer.param_groups] == pytest.approx(
        [0.0001, 0.0005, 0.0001]
    )


def test_five_and_ten_control_steps_report_actual_50ms_100ms_fk_motion():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=10)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=11)
    stage = copy.deepcopy(settings["curriculum"]["stages"][0])
    # Keep one task alive for all ten intervals so MOVE100 cannot accidentally
    # join FK samples from two independently reset targets.
    stage["rollout_steps_per_slot"] = 20
    stage["success_position_m"] = 1e-9
    environment.begin_rollout(stage)
    final_infos = None
    action = np.full((2, 6), 0.1, dtype=float)
    for _ in range(10):
        _, _, _, final_infos = environment.step(action, stage)
    assert all(np.isfinite(item["tcp_50ms_net_displacement_m"]) for item in final_infos)
    assert all(item["tcp_50ms_net_displacement_m"] > 0.0 for item in final_infos)
    assert all(
        item["tcp_50ms_path_length_m"]
        >= item["tcp_50ms_net_displacement_m"]
        for item in final_infos
    )
    assert all(np.isfinite(item["tcp_100ms_net_displacement_m"]) for item in final_infos)
    assert all(item["tcp_100ms_net_displacement_m"] > 0.0 for item in final_infos)
    assert all(
        item["tcp_100ms_path_length_m"]
        >= item["tcp_100ms_net_displacement_m"]
        for item in final_infos
    )


def test_pose_stage_scales_physical_joint_velocity_without_changing_policy_action():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=51)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=52)
    pose_stage = settings["curriculum"]["stages"][1]
    environment.begin_rollout(pose_stage)
    action = np.full((2, 6), 0.5, dtype=float)
    _, _, _, infos = environment.step(action, pose_stage)
    expected = (
        0.5
        * environment.action_velocity_limit
        * float(pose_stage["action_velocity_scale"])
    )
    assert all(
        np.allclose(slot.previous_desired_velocity, expected)
        for slot in environment.slots
    )
    assert all(
        item["action_velocity_scale"] == pytest.approx(0.10)
        for item in infos
    )


def test_pose_near_target_progress_rewards_approach_and_penalizes_retreat():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=71)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=72)
    stage = copy.deepcopy(settings["curriculum"]["stages"][1])
    stage["success_position_m"] = 1e-9
    environment.begin_rollout(stage)
    reference = float(settings["reward_position_progress_reference_m"])
    for slot_index, slot in enumerate(environment.slots):
        current = ur5e.tcp_pose6(slot.data)
        target = current.copy()
        target[0] += 0.012
        target[3:] = (
            Rotation.from_rotvec(np.deg2rad([0.0, 0.0, 3.0]))
            * Rotation.from_rotvec(current[3:])
        ).as_rotvec()
        slot.target_pose6 = target
        slot.previous_position_error_m = 0.012 + (
            reference if slot_index == 0 else -reference
        )
        slot.previous_orientation_error_rad = np.deg2rad(3.0)

    _, _, _, infos = environment.step(
        np.zeros((2, 6), dtype=float), stage
    )
    infos = sorted(infos, key=lambda item: item["slot_id"])
    assert infos[0]["near_target_progress_weight"] > 0.0
    assert infos[1]["near_target_progress_weight"] > 0.0
    assert infos[0]["position_progress"] > 0.0
    assert infos[0]["position_progress_reward"] > 0.0
    assert infos[1]["position_progress"] < 0.0
    assert infos[1]["position_progress_reward"] < 0.0


def test_rollout_start_resets_every_slot_to_the_initial_pool():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=12)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=13)
    stage = settings["curriculum"]["stages"][0]

    # Deliberately move both logical chains away from their pool rows.  A new
    # rollout must not inherit either q or qdot from this artificial state.
    for slot in environment.slots:
        moved = np.clip(
            pool[slot.slot_id] + np.full(6, 0.01),
            environment.safe_lower,
            environment.safe_upper,
        )
        ur5e.reset_data(slot.data, moved, np.full(6, 0.2))
        slot.command_q = moved.copy()
        slot.episode_step_count = 19

    environment.begin_rollout(stage)
    matched_pool_rows = []
    for slot in environment.slots:
        joints = ur5e.joints(slot.data)
        distances = np.linalg.norm(pool - joints[None, :], axis=1)
        matched_pool_rows.append(int(np.argmin(distances)))
        assert np.min(distances) < 1e-10
        assert np.allclose(ur5e.joint_velocities(slot.data), 0.0)
        assert slot.episode_step_count == 0
    assert sorted(matched_pool_rows) == [0, 1]


def test_rollout_cut_truncates_partial_task_but_next_rollout_resets_it():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=31)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=32)
    stage = copy.deepcopy(settings["curriculum"]["stages"][0])
    stage["success_position_m"] = 1e-6
    environment.begin_rollout(stage)
    for slot in environment.slots:
        target = ur5e.tcp_pose6(slot.data)
        target[0] += 0.05
        slot.target_pose6 = target
        slot.episode_step_count = 1

    _, _, boundaries, infos = environment.step(
        np.zeros((2, 6), dtype=float), stage, horizon_boundary=True
    )
    assert np.all(boundaries == 1.0)
    assert all(item["rollout_truncated"] for item in infos)
    assert all(not item["episode_done"] for item in infos)
    assert all(item["terminal_reason"] == "rollout_truncated" for item in infos)

    environment.begin_rollout(stage)
    assert all(slot.episode_step_count == 0 for slot in environment.slots)
    assert all(np.allclose(ur5e.joint_velocities(slot.data), 0.0) for slot in environment.slots)


def test_full_stage_budget_is_a_true_task_timeout_not_a_truncation():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=33)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=34)
    stage = copy.deepcopy(settings["curriculum"]["stages"][0])
    stage["success_position_m"] = 1e-6
    environment.begin_rollout(stage)
    for slot in environment.slots:
        target = ur5e.tcp_pose6(slot.data)
        target[0] += 0.05
        slot.target_pose6 = target
        slot.episode_step_count = stage["rollout_steps_per_slot"] - 1

    _, _, boundaries, infos = environment.step(
        np.zeros((2, 6), dtype=float), stage, horizon_boundary=True
    )
    assert np.all(boundaries == 1.0)
    assert all(item["task_timeout"] for item in infos)
    assert all(item["episode_done"] for item in infos)
    assert all(not item["rollout_truncated"] for item in infos)
    assert all(item["terminal_reason"] == "task_timeout" for item in infos)


def test_v9_rollout_stops_spawning_then_drains_every_started_task():
    pytest.importorskip("mujoco")
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_curriculum_v9.yaml")
    )
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=81)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=82)
    trainer = JailerPPO2Trainer(settings["ppo"], device="cpu")
    trainer.set_curriculum_task("pose")
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
        trainer.model.actor.pose_residual_log_std.fill_(-5.0)

    stage = copy.deepcopy(settings["curriculum"]["stages"][12])
    stage["rollout_steps_per_slot"] = 5
    original_step = environment.step
    calls = {"count": 0}

    def first_step_succeeds_then_new_tasks_need_full_budget(
        actions, current_stage, **kwargs
    ):
        effective = copy.deepcopy(current_stage)
        if calls["count"] == 0:
            effective["success_position_m"] = 10.0
            effective["success_orientation_deg"] = 180.0
            effective["success_tcp_speed_mps"] = 10.0
            effective["success_tcp_angular_speed_radps"] = 10.0
        else:
            effective["success_position_m"] = 1e-12
            effective["success_orientation_deg"] = 1e-12
        calls["count"] += 1
        return original_step(actions, effective, **kwargs)

    environment.step = first_step_succeeds_then_new_tasks_need_full_budget
    rollout, metrics, _ = collect_logical_rollout(
        trainer, environment, stage
    )
    # Two first-step successes are followed by two tasks started at step two.
    # Those tasks receive all five of their own steps, requiring one drain
    # iteration beyond the nominal five-step collection horizon.
    assert metrics["nominal_rollout_transitions"] == 10
    assert metrics["drain_transition_count"] == 2
    assert metrics["drain_control_iterations"] == 1
    assert metrics["rollout_transitions"] == 12
    assert metrics["completed_task_count"] == 4
    assert metrics["rollout_truncated_task_count"] == 0
    assert np.sum(rollout.dones) == pytest.approx(4.0)
    assert np.array_equal(rollout.dones, rollout.bootstrap_dones)


def test_v10_expands_two_position_passes_and_pose_pass():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_curriculum_v10_full.yaml")
    )
    settings = config["jailer"]
    stages = settings["curriculum"]["stages"]
    assert len(stages) == 24
    assert [stage["curriculum_phase"] for stage in stages[:6]] == [
        "position_masked"
    ] * 6
    assert [stage["curriculum_phase"] for stage in stages[6:12]] == [
        "position_observed"
    ] * 6
    assert stages[6]["entry_transition"] == "masked_to_observed_new_critic"
    assert stages[12]["entry_transition"] == "preserve_critic_reset_adam_align"
    assert stages[12]["entry_critic_alignment_updates"] == 5
    assert settings["rollout_boundary_mode"] == "stop_spawn_then_drain_active_tasks_v1"
    assert settings["ppo"]["updates"] == 1_000_000
    assert settings["curriculum"]["stop_when_final_stage_mastered"] is True


def test_v11_joint_policy_uses_one_action_and_a_frozen_p6_reference():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_joint_v11.yaml")
    )
    settings = config["jailer"]
    stage = settings["curriculum"]["stages"][0]
    assert settings["ppo"]["action_composition_mode"] == JOINT_REFERENCE_ACTION
    assert all(item["task"] == "pose" for item in settings["curriculum"]["stages"])
    trainer = JailerPPO2Trainer(settings["ppo"], device="cpu")
    trainer.set_curriculum_task("pose")
    trainer.prepare_training_stage(stage, entering=True)
    assert not any(
        parameter.requires_grad
        for parameter in trainer.model.reference_actor.parameters()
    )
    raw = np.zeros((2, 18), dtype=np.float32)
    raw[0, 0] = 0.5 * float(stage["success_position_m"])
    raw[1, 0] = 4.0 * float(stage["success_position_m"])
    raw[:, 3] = np.deg2rad(float(stage["success_orientation_deg"]))
    gains = joint_action_gain(raw, settings["ppo"], stage)
    assert 0.0 < gains[0] < gains[1]
    assert gains[1] == pytest.approx(stage["action_velocity_scale"])


def test_v12_joint_policy_has_no_reference_actor_or_auxiliary_imitation():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_joint_v12.yaml")
    )
    settings = config["jailer"]
    stage = settings["curriculum"]["stages"][0]
    assert settings["ppo"]["action_composition_mode"] == JOINT_PPO_ACTION
    assert not any(
        "position_reference_loss_weight" in item
        for item in settings["curriculum"]["stages"]
    )
    assert not any(
        "reference" in key or "rehearsal" in key
        for key in settings["ppo"]
    )
    trainer = JailerPPO2Trainer(settings["ppo"], device="cpu")
    trainer.set_curriculum_task("pose")
    trainer.prepare_training_stage(stage, entering=True)
    assert not hasattr(trainer.model, "reference_actor")
    auxiliary, metrics = trainer.actor_auxiliary_loss(torch.zeros(4, 18))
    assert auxiliary.item() == pytest.approx(0.0)
    assert metrics["actor_auxiliary_loss"].item() == pytest.approx(0.0)


def _v12_synthetic_trainer_and_rollout():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_joint_v12.yaml")
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


def test_v12_actor_and_critic_adam_parameters_are_strictly_isolated():
    trainer, _ = _v12_synthetic_trainer_and_rollout()
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


def test_v12_synthetic_rollout_updates_both_independent_adam_optimizers():
    trainer, rollout = _v12_synthetic_trainer_and_rollout()
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


def test_v12_entropy_schedule_restarts_within_b1_and_o6():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_joint_v12.yaml")
    )
    settings = copy.deepcopy(config["jailer"])
    settings["ppo"]["hidden_dims"] = [16]
    trainer = JailerPPO2Trainer(settings["ppo"], device="cpu")
    stages = settings["curriculum"]["stages"]
    for stage, expected_start, expected_end in (
        (stages[0], 0.0030, 0.0025),
        (stages[-1], 0.0030, 0.0018),
    ):
        trainer.prepare_training_stage(stage, entering=True)
        assert trainer.effective_entropy_coefficient() == pytest.approx(
            expected_start
        )
        trainer.prepare_training_stage(
            stage,
            entering=False,
            updates_in_stage=int(stage["minimum_updates"]),
        )
        assert trainer.effective_entropy_coefficient() == pytest.approx(
            expected_end
        )


def test_v12_critic_alignment_preserves_actor_and_actor_adam_state():
    trainer, rollout = _v12_synthetic_trainer_and_rollout()
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


def test_v12_multiscale_position_quality_resolves_millimeter_errors():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_joint_v12.yaml")
    )
    settings = config["jailer"]
    environment = object.__new__(JailerLogicalVectorEnv)
    environment.config = settings
    stage = settings["curriculum"]["stages"][5]
    at_threshold = environment._position_quality(stage, 0.0025)
    at_3p5mm = environment._position_quality(stage, 0.0035)
    at_stage_max = environment._position_quality(stage, 0.025)
    assert at_threshold == pytest.approx(
        [0.64702970297, 0.5, 0.99009900990]
    )
    assert at_3p5mm[0] == pytest.approx(0.47415295223)
    assert at_threshold[0] - at_3p5mm[0] > 0.17
    assert at_stage_max[0] == pytest.approx(0.15)


def test_v12_achieved_component_keeps_quality_weight_but_no_positive_progress():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_joint_v12.yaml")
    )
    settings = config["jailer"]
    environment = object.__new__(JailerLogicalVectorEnv)
    environment.config = settings
    stage = settings["curriculum"]["stages"][5]
    position_weight, orientation_weight = environment._reward_weights(
        stage,
        position_error=0.002,
        orientation_error=np.deg2rad(3.0),
    )
    assert position_weight == pytest.approx(0.15)
    assert orientation_weight == pytest.approx(0.85)
    mode = settings["reward_component_progress_after_success_mode"]
    assert environment._component_progress(0.4, True, mode) == 0.0
    assert environment._component_progress(-0.4, True, mode) == -0.4
    assert environment._component_progress(0.4, False, mode) == 0.4


def test_v11_dynamic_reward_prioritizes_the_larger_normalized_error():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_joint_v11.yaml")
    )
    settings = config["jailer"]
    settings["logical_slots"] = 1
    environment = object.__new__(JailerLogicalVectorEnv)
    environment.config = settings
    stage = settings["curriculum"]["stages"][2]
    position_weight, orientation_weight = environment._reward_weights(
        stage,
        position_error=0.5 * float(stage["success_position_m"]),
        orientation_error=np.deg2rad(
            2.0 * float(stage["success_orientation_deg"])
        ),
    )
    assert orientation_weight > position_weight
    assert position_weight >= settings["dynamic_reward_min_weight"]


def test_v10_stage_boundaries_reset_only_the_agreed_training_state():
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_curriculum_v10_full.yaml")
    )
    settings = config["jailer"]
    stages = settings["curriculum"]["stages"]
    trainer = JailerPPO2Trainer(settings["ppo"], device="cpu")
    trainer.jailer_training_initialization = {"mode": "random"}
    curriculum = JailerCurriculum(settings["curriculum"])
    first_layer = next(
        module
        for module in trainer.model.actor.features
        if isinstance(module, torch.nn.Linear)
    )
    with torch.no_grad():
        first_layer.weight[:, 3:6].fill_(0.25)
        for parameter in trainer.model.critic.parameters():
            parameter.fill_(0.5)
    actor_before = {
        name: value.detach().clone()
        for name, value in trainer.model.actor.named_parameters()
    }
    transition = _apply_stage_entry_transition(
        trainer, curriculum, stages[5], stages[6], settings
    )
    assert transition["stage_transition_critic_weights_preserved"] is False
    assert torch.count_nonzero(first_layer.weight[:, 3:6]) == 0
    for name, value in trainer.model.actor.named_parameters():
        if name == "features.0.weight":
            assert torch.equal(value[:, :3], actor_before[name][:, :3])
            assert torch.equal(value[:, 6:], actor_before[name][:, 6:])
        else:
            assert torch.equal(value, actor_before[name])
    assert any(
        not torch.allclose(parameter, torch.full_like(parameter, 0.5))
        for parameter in trainer.model.critic.parameters()
    )

    critic_before = {
        name: value.detach().clone()
        for name, value in trainer.model.critic.named_parameters()
    }
    actor_before = {
        name: value.detach().clone()
        for name, value in trainer.model.actor.named_parameters()
    }
    curriculum.state.stage_index = 12
    transition = _apply_stage_entry_transition(
        trainer, curriculum, stages[11], stages[12], settings
    )
    assert transition["stage_transition_critic_weights_preserved"] is True
    assert transition["stage_transition_optimizer_state_preserved"] is False
    assert curriculum.state.critic_alignment_updates_remaining == 5
    assert not trainer.optimizer.state
    for name, value in trainer.model.critic.named_parameters():
        assert torch.equal(value, critic_before[name])
    for name, value in trainer.model.actor.named_parameters():
        assert torch.equal(value, actor_before[name])


@pytest.mark.parametrize("stage_index", [0, 12, 18])
def test_v10_drained_boundary_has_true_outcomes_in_p_b_o(stage_index):
    pytest.importorskip("mujoco")
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_curriculum_v10_full.yaml")
    )
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=91 + stage_index)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(
        ur5e, settings, pool, seed=92 + stage_index
    )
    assert _configured_state_contract(environment) == JAILER_PHASED_STATE_CONTRACT
    trainer = JailerPPO2Trainer(settings["ppo"], device="cpu")
    stage = copy.deepcopy(settings["curriculum"]["stages"][stage_index])
    trainer.set_curriculum_task(stage["task"])
    stage["rollout_steps_per_slot"] = 3
    stage["success_position_m"] = 1e-12
    if stage["task"] == "pose":
        stage["success_orientation_deg"] = 1e-12
    rollout, metrics, _ = collect_logical_rollout(trainer, environment, stage)
    assert metrics["rollout_truncated_task_count"] == 0
    assert metrics["completed_task_count"] >= 2
    assert np.array_equal(rollout.dones, rollout.bootstrap_dones)


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


def test_critic_only_alignment_does_not_update_actor():
    pytest.importorskip("mujoco")
    config = load_config(
        str(ROOT / "configs" / "ur5e_jailer_curriculum_v10_full.yaml")
    )
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=121)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=122)
    trainer = JailerPPO2Trainer(settings["ppo"], device="cpu")
    stage = copy.deepcopy(settings["curriculum"]["stages"][12])
    stage["rollout_steps_per_slot"] = 3
    trainer.set_curriculum_task("pose")
    rollout, _, _ = collect_logical_rollout(trainer, environment, stage)
    actor_before = {
        name: value.detach().clone()
        for name, value in trainer.model.actor.named_parameters()
    }
    critic_before = {
        name: value.detach().clone()
        for name, value in trainer.model.critic.named_parameters()
    }
    metrics = trainer.align_critic(rollout)
    assert metrics["optimization_mode"] == "critic_only_alignment"
    for name, value in trainer.model.actor.named_parameters():
        assert torch.equal(value, actor_before[name])
    assert any(
        not torch.equal(value, critic_before[name])
        for name, value in trainer.model.critic.named_parameters()
    )


def test_reaching_target_terminates_task_and_recovers_slot():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_smoke.yaml"))
    settings = copy.deepcopy(config["jailer"])
    settings["logical_slots"] = 2
    settings["initial_pose_pool"]["minimum_pairwise_joint_distance_rad"] = 0.01
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    pool = generate_initial_joint_pool(ur5e, settings, 2, seed=14)[
        "joint_positions_rad"
    ]
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed=15)
    stage = settings["curriculum"]["stages"][0]
    environment.begin_rollout(stage)
    for slot in environment.slots:
        slot.target_pose6 = ur5e.tcp_pose6(slot.data)

    _, rewards, dones, infos = environment.step(
        np.zeros((2, 6), dtype=float), stage
    )
    assert np.all(dones == 1.0)
    assert np.all(rewards > 0.0)
    assert all(item["task_success"] for item in infos)
    assert all(item["terminal_reason"] == "success" for item in infos)
    assert all(item["episode_step_count"] == 1 for item in infos)
    assert all(item["initial_position_error_m"] > 0.0 for item in infos)
    # Recovery starts a fresh task; it does not keep sampling around the
    # already reached target.
    assert all(slot.episode_step_count == 0 for slot in environment.slots)


def test_success_cases_export_csv_and_pdf(tmp_path):
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_curriculum.yaml"))
    settings = config["jailer"]
    database_path = tmp_path / "success.sqlite3"
    csv_path = tmp_path / "success.csv.gz"
    pdf_path = tmp_path / "success.pdf"
    connection = _prepare_success_database(database_path, reset=True)
    stage = settings["curriculum"]["stages"][0]
    _append_success_cases(
        connection,
        update_index=1,
        stage=stage,
        cases=[
            {
                "slot_id": 0,
                "task_sequence": 1,
                "initial_position_error_m": 0.010,
                "initial_orientation_error_deg": 0.0,
                "steps": 4,
            },
            {
                "slot_id": 1,
                "task_sequence": 1,
                "initial_position_error_m": 0.018,
                "initial_orientation_error_deg": 0.0,
                "steps": 7,
            },
        ],
        control_rate_hz=100.0,
    )
    _export_success_csv(connection, csv_path)
    _plot_success_curves(connection, pdf_path, settings)
    connection.close()
    assert database_path.stat().st_size > 0
    assert csv_path.stat().st_size > 0
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_constrained_differential_ik_reduces_tcp_error_and_audits_time():
    pytest.importorskip("mujoco")
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_curriculum.yaml"))
    ur5e = UR5eControlModel(str(MODEL), config["robot"], control_rate_hz=100.0)
    start = np.asarray(config["robot"]["nominal_joint_state_rad"], dtype=float)
    goal_q = start + np.array([0.08, 0.05, -0.05, 0.03, 0.04, 0.05])
    goal_data = ur5e.new_data()
    ur5e.reset_data(goal_data, goal_q)
    goal_pose = ur5e.tcp_pose6(goal_data)
    settings = copy.deepcopy(config["constrained_differential_ik"])
    settings["settling_time_s"] = 0.10
    report = simulate_time_constrained_ik(
        ur5e, settings, start, goal_pose, duration_s=0.30
    )
    initial_data = ur5e.new_data()
    ur5e.reset_data(initial_data, start)
    initial_error = np.linalg.norm(goal_pose[:3] - ur5e.tcp_pose6(initial_data)[:3])
    assert report["final_state"]["position_error_m"] < initial_error
    assert report["qp_solve_time_mean_ms"] >= 0.0
    assert np.isfinite(report["tcp_50ms_net_displacement_mean_m"])
    assert np.isfinite(report["tcp_100ms_net_displacement_mean_m"])
    assert "success_at_prescribed_time" in report


def test_curriculum_requires_window_then_advances_position_to_pose():
    curriculum = JailerCurriculum(
        {
            "promotion_window_updates": 2,
            "stages": [
                {
                    "name": "position",
                    "task": "position",
                    "minimum_updates": 2,
                    "promotion_position_mean_m": 0.01,
                    "promotion_orientation_mean_deg": 180.0,
                    "promotion_success_rate": 0.8,
                },
                {
                    "name": "pose",
                    "task": "pose",
                    "minimum_updates": 2,
                    "promotion_position_mean_m": 0.01,
                    "promotion_orientation_mean_deg": 5.0,
                    "promotion_success_rate": 0.8,
                },
            ],
        }
    )
    good_position = {
        "final_position_error_mean_m": 0.005,
        "final_orientation_error_mean_deg": 90.0,
        "terminal_success_rate": 1.0,
    }
    assert curriculum.observe(good_position) is None
    promotion = curriculum.observe(good_position)
    assert promotion["next"] == "pose"
    good_pose = {
        "final_position_error_mean_m": 0.005,
        "final_orientation_error_mean_deg": 2.0,
        "terminal_success_rate": 1.0,
    }
    assert curriculum.observe(good_pose) is None
    promotion = curriculum.observe(good_pose)
    assert promotion["final_mastered"] is True


def test_actor_warm_start_can_begin_at_an_explicit_pose_stage():
    config = load_config(str(ROOT / "configs" / "ur5e_jailer_curriculum.yaml"))
    curriculum = JailerCurriculum(config["jailer"]["curriculum"])
    stage_name = "pose_o01_006_025mm_01_05deg"
    curriculum.start_at_stage(stage_name)
    assert curriculum.stage["name"] == stage_name
    assert curriculum.state.updates_in_stage == 0
    assert list(curriculum.state.promotion_window) == []


def test_pose_curriculum_can_require_strictly_more_than_eighty_percent():
    curriculum = JailerCurriculum(
        {
            "promotion_window_updates": 1,
            "stages": [
                {
                    "name": "pose_strict",
                    "task": "pose",
                    "minimum_updates": 1,
                    "promotion_position_mean_m": 0.01,
                    "promotion_orientation_mean_deg": 5.0,
                    "promotion_success_rate": 0.80,
                    "promotion_success_rate_strict": True,
                }
            ],
        }
    )
    metrics = {
        "final_position_error_mean_m": 0.005,
        "final_orientation_error_mean_deg": 2.0,
        "terminal_success_rate": 0.80,
    }
    assert curriculum.observe(metrics) is None
    metrics["terminal_success_rate"] = 0.81
    promotion = curriculum.observe(metrics)
    assert promotion["final_mastered"] is True
