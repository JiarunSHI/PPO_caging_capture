"""Command line entry points for the independent 100 Hz control branches."""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from cage_system.config import load_config
from cage_system.control.constrained_differential_ik import (
    simulate_time_constrained_ik,
)
from cage_system.sim.jailer_env import (
    generate_initial_joint_pool,
    save_initial_joint_pool,
)
from cage_system.sim.ur5e_control import UR5eControlModel


def _write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def command_generate_initial_pool(args):
    config = load_config(args.config)
    settings = config["jailer"]
    ur5e = UR5eControlModel(
        args.mujoco_model,
        config["robot"],
        tcp_site_name=args.tcp_site,
        control_rate_hz=float(settings["control_rate_hz"]),
    )
    arrays = generate_initial_joint_pool(
        ur5e,
        settings,
        int(args.count),
        int(args.seed if args.seed is not None else settings["seed"]),
    )
    save_initial_joint_pool(args.output, arrays)
    joints = arrays["joint_positions_rad"]
    tcp = arrays["tcp_pose6"]
    report = {
        "output": str(args.output),
        "count": int(len(joints)),
        "joint_min_rad": np.min(joints, axis=0).tolist(),
        "joint_max_rad": np.max(joints, axis=0).tolist(),
        "tcp_position_min_m": np.min(tcp[:, :3], axis=0).tolist(),
        "tcp_position_max_m": np.max(tcp[:, :3], axis=0).tolist(),
        "ur5e_contract": ur5e.contract_dict(),
    }
    print(json.dumps(report, ensure_ascii=False))


def command_train(args):
    config = load_config(args.config)
    from cage_system.learning.train_jailer import train_jailer_curriculum

    report = train_jailer_curriculum(
        config,
        args.mujoco_model,
        args.tcp_site,
        args.output,
        device=args.device,
        initial_pool_path=args.initial_pose_pool,
        resume_checkpoint=args.resume_checkpoint,
        resume_start_stage=args.resume_start_stage,
        actor_warm_start_checkpoint=args.actor_warm_start_checkpoint,
        actor_warm_start_stage=args.actor_warm_start_stage,
    )
    print(json.dumps(report, ensure_ascii=False))


def command_render_video(args):
    config = load_config(args.config)
    from cage_system.jailer_visualization import render_jailer_video

    report = render_jailer_video(
        config=config,
        checkpoint_path=args.checkpoint,
        model_path=args.mujoco_model,
        tcp_site_name=args.tcp_site,
        initial_pool_path=args.initial_pose_pool,
        stage_name=args.stage,
        slot_id=args.slot_id,
        seed=args.seed,
        control_steps=args.control_steps,
        video_fps=args.video_fps,
        output_path=args.output,
        device=args.device,
    )
    summary = {key: value for key, value in report.items() if key != "trace"}
    print(json.dumps(summary, ensure_ascii=False))


def command_run_jacobian(args):
    config = load_config(args.config)
    settings = copy.deepcopy(config["constrained_differential_ik"])
    if args.position_only:
        settings["position_only"] = True
    if float(settings["control_rate_hz"]) != 100.0:
        raise ValueError("Constrained differential IK is fixed at 100 Hz")
    ur5e = UR5eControlModel(
        args.mujoco_model,
        config["robot"],
        tcp_site_name=args.tcp_site,
        control_rate_hz=float(settings["control_rate_hz"]),
    )
    start_joints = (
        np.asarray(config["robot"]["nominal_joint_state_rad"], dtype=float)
        if args.start_joints is None
        else np.asarray(args.start_joints, dtype=float)
    )
    if args.goal_pose is not None:
        goal_pose = np.asarray(args.goal_pose, dtype=float)
        goal_source = "explicit_base_tcp_pose6"
    else:
        goal_data = ur5e.new_data()
        ur5e.reset_data(goal_data, np.asarray(args.goal_joints, dtype=float))
        goal_pose = ur5e.tcp_pose6(goal_data)
        goal_source = "fk_of_explicit_goal_q6"
    report = simulate_time_constrained_ik(
        ur5e,
        settings,
        start_joints,
        goal_pose,
        float(args.duration),
    )
    report["goal_source"] = goal_source
    _write_json(args.output, report)
    summary = {key: value for key, value in report.items() if key != "trace"}
    summary["output"] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False))


def command_evaluate_dynamic_cage(args):
    config = load_config(args.config)
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    from cage_system.dynamic_cage_inference import evaluate_dynamic_cage_tracking

    report = evaluate_dynamic_cage_tracking(
        config=config,
        checkpoint_path=args.checkpoint,
        model_path=args.mujoco_model,
        tcp_site_name=args.tcp_site,
        episodes=args.episodes,
        seed=args.seed,
        output_path=args.output,
        trajectory_output=args.trajectory_output,
        stage_name=args.stage,
        video_output=args.video_output,
        video_fps=args.video_fps,
        show_window=args.show_window,
        realtime_display=args.realtime_display,
        show_gripper=args.show_gripper,
        device=args.device,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "episodes"},
            ensure_ascii=False,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="UR5e JAiLeR curriculum and constrained Jacobian branches"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pool = subparsers.add_parser("generate-initial-pool")
    pool.add_argument("--config", required=True)
    pool.add_argument("--mujoco-model", required=True)
    pool.add_argument("--tcp-site", default="attachment_site")
    pool.add_argument("--count", type=int, default=40)
    pool.add_argument("--seed", type=int)
    pool.add_argument("--output", required=True)
    pool.set_defaults(function=command_generate_initial_pool)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--mujoco-model", required=True)
    train.add_argument("--tcp-site", default="attachment_site")
    train.add_argument("--initial-pose-pool")
    continuation = train.add_mutually_exclusive_group()
    continuation.add_argument("--resume-checkpoint")
    continuation.add_argument("--actor-warm-start-checkpoint")
    train.add_argument(
        "--resume-start-stage",
        help=(
            "explicit curriculum stage for a full-state resume; preserves "
            "Actor, Critic, optimizer, and normalization statistics"
        ),
    )
    train.add_argument(
        "--actor-warm-start-stage",
        help=(
            "start this curriculum stage after an Actor-only warm start; "
            "invalid with full resume"
        ),
    )
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="auto")
    train.set_defaults(function=command_train)

    jacobian = subparsers.add_parser("run-constrained-jacobian")
    jacobian.add_argument("--config", required=True)
    jacobian.add_argument("--mujoco-model", required=True)
    jacobian.add_argument("--tcp-site", default="attachment_site")
    jacobian.add_argument("--start-joints", nargs=6, type=float)
    goal = jacobian.add_mutually_exclusive_group(required=True)
    goal.add_argument("--goal-joints", nargs=6, type=float)
    goal.add_argument(
        "--goal-pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Base-frame target TCP xyz+rotation-vector",
    )
    jacobian.add_argument("--duration", type=float, required=True)
    jacobian.add_argument("--position-only", action="store_true")
    jacobian.add_argument("--output", required=True)
    jacobian.set_defaults(function=command_run_jacobian)

    dynamic = subparsers.add_parser("evaluate-dynamic-cage")
    dynamic.add_argument("--config", required=True)
    dynamic.add_argument("--checkpoint", required=True)
    dynamic.add_argument("--mujoco-model", required=True)
    dynamic.add_argument("--tcp-site", default="attachment_site")
    dynamic.add_argument("--stage")
    dynamic.add_argument("--episodes", type=int, default=20)
    dynamic.add_argument("--seed", type=int, default=20260911)
    dynamic.add_argument("--video-output")
    dynamic.add_argument("--trajectory-output")
    dynamic.add_argument("--video-fps", type=float, default=25.0)
    dynamic.add_argument("--show-window", action="store_true")
    dynamic.add_argument("--realtime-display", action="store_true")
    dynamic.add_argument("--output", required=True)
    dynamic.add_argument("--device", default="auto")
    dynamic.set_defaults(function=command_evaluate_dynamic_cage)
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
