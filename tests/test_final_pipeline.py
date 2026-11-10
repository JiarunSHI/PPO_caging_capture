"""Small regressions for the final entry points and causal execution contract."""
from pathlib import Path

import numpy as np

from cage_system.cli import build_parser
from cage_system.config import load_config
from cage_system.learning.ppo2 import compute_gae
from cage_system.sim.ur5e_control import UR5eControlModel
from cage_system.trajectory.online import OnlineTrajectoryEstimator
from cage_system.types import PointObservation

ROOT = Path(__file__).resolve().parents[1]


def test_final_cli_keeps_existing_training_and_visualization_flags():
    args = build_parser().parse_args([
        'evaluate-dynamic-cage', '--config', 'config.yaml',
        '--checkpoint', 'best.pt', '--mujoco-model', 'scene.xml',
        '--output', 'report.json', '--show-gripper',
        '--show-window', '--realtime-display',
    ])
    assert args.show_gripper and args.show_window and args.realtime_display
    assert args.function.__name__ == 'command_evaluate_dynamic_cage'


def test_final_curriculum_has_fifty_slots_and_three_ten_stage_passes():
    config = load_config(str(ROOT / 'configs/ur5e_jailer_joint_v14_full.yaml'))
    settings = config['jailer']
    assert settings['logical_slots'] == 50
    assert settings['control_rate_hz'] == 100.0
    stages = settings['curriculum']['stages']
    assert len(stages) == 30
    assert [s['task'] for s in stages] == ['position'] * 20 + ['pose'] * 10
    assert settings['curriculum']['stop_when_final_stage_mastered']


def test_first_four_visual_points_do_not_initialize_kalman():
    config = load_config(str(ROOT / 'configs/ur5e_dynamic_cage_inference.yaml'))
    estimator = OnlineTrajectoryEstimator(config['trajectory'], config['kalman'], config['timing'])
    for i in range(5):
        timestamp = i * 0.1
        point = np.array([0.5 + 0.12 * timestamp, 0.03 * timestamp ** 2, 0.4])
        estimator.update(PointObservation(timestamp, point))
        if i < 4:
            assert estimator.filter.timestamp is None
    assert np.isclose(estimator.filter.timestamp, 0.4)
    estimator.update(PointObservation(0.5, np.array([0.56, 0.0075, 0.4])))
    assert np.isclose(estimator.filter.timestamp, 0.5)


def test_timeout_keeps_bootstrap_without_leaking_next_task_advantage():
    advantages = compute_gae(
        rewards=np.array([1., 100.]), values=np.array([2., 0.]),
        next_values=np.array([3., 0.]), dones=np.array([1., 1.]),
        gamma=0.9, gae_lambda=0.95, bootstrap_dones=np.array([0., 1.]),
    )
    np.testing.assert_allclose(advantages, [1.7, 100.])
    np.testing.assert_allclose(advantages + np.array([2., 0.]), [3.7, 100.])


def test_named_ur5e_binding_keeps_five_physics_steps_per_control():
    config = load_config(str(ROOT / 'configs/ur5e_jailer_joint_v14_full.yaml'))
    robot = UR5eControlModel(str(ROOT / 'tests/assets/six_joint_contract.xml'), config['robot'], tcp_site_name='attachment_site', control_rate_hz=100.0)
    contract = robot.contract_dict()
    assert len(contract['joint_names']) == 6
    assert contract['physics_steps_per_control'] == 5
