"""Commands for offline EST experience, final PPO training, and dynamic tracking."""
import argparse
import json
import numpy as np
from cage_system.config import load_config
from cage_system.io import load_generated_trajectories, save_generated_trajectories, validate_trajectory_detection_period
from cage_system.trajectory.generator import PlanarTrajectoryGenerator, trajectory_diagnostics


def command_generate_trajectories(args):
    config = load_config(args.config)
    generator = PlanarTrajectoryGenerator(
        config["trajectory"], seed=int(config["system"]["seed"])
    )
    trajectories = [generator.generate(index) for index in range(int(args.count))]
    save_generated_trajectories(trajectories, args.output)
    diagnostics = [
        trajectory_diagnostics(
            item.timestamps,
            item.true_positions,
            int(config["trajectory"]["curvature_history_frames"]),
        )
        for item in trajectories
    ]
    family_counts = {}
    for item in trajectories:
        family_counts[item.family] = family_counts.get(item.family, 0) + 1
    print(
        json.dumps(
            {
                "trajectory_count": len(trajectories),
                "families": family_counts,
                "validation": {
                    "curvature_max_m_inv": max(
                        item.curvature_max_m_inv for item in diagnostics
                    ),
                    "five_frame_curvature_range_min_m_inv": min(
                        item.five_frame_curvature_range_min_m_inv
                        for item in diagnostics
                    ),
                    "five_frame_curvature_range_max_m_inv": max(
                        item.five_frame_curvature_range_max_m_inv
                        for item in diagnostics
                    ),
                    "local_plane_total_variation_min_deg": min(
                        item.local_plane_total_variation_deg for item in diagnostics
                    ),
                    "local_plane_total_variation_max_deg": max(
                        item.local_plane_total_variation_deg for item in diagnostics
                    ),
                    "local_plane_max_step_deg": max(
                        item.local_plane_max_step_deg for item in diagnostics
                    ),
                    "five_frame_turn_max_deg": max(
                        item.five_frame_turn_max_deg for item in diagnostics
                    ),
                },
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )

def command_generate_labels(args):
    config = load_config(args.config)
    trajectories = load_generated_trajectories(args.input)
    validate_trajectory_detection_period(
        trajectories, 1.0 / float(config["timing"]["detection_rate_hz"])
    )
    from cage_system.parallel import generate_labels_parallel

    arrays, _ = generate_labels_parallel(
        config,
        trajectories,
        args.output,
        args.audit,
        args.workers,
        resume=args.resume,
        keep_parts=args.keep_parts,
    )
    print(
        json.dumps(
            {
                "valid_label_count": len(arrays.inputs),
                "trajectory_count": len(trajectories),
                "output": args.output,
                "audit": args.audit,
            },
            ensure_ascii=False,
        )
    )

def build_parser():
    parser = argparse.ArgumentParser(description="Final EST experience, UR5e PPO training, and dynamic Cage tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-trajectories")
    generate.add_argument("--config", required=True)
    generate.add_argument("--count", type=int, required=True)
    generate.add_argument("--output", required=True)
    generate.set_defaults(function=command_generate_trajectories)
    labels = subparsers.add_parser("generate-labels")
    labels.add_argument("--config", required=True)
    labels.add_argument("--input", required=True)
    labels.add_argument("--output", required=True)
    labels.add_argument("--audit", required=True)
    labels.add_argument("--workers", type=int, default=1)
    labels.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="discard per-track checkpoints and recompute every track",
    )
    labels.add_argument(
        "--keep-parts",
        action="store_true",
        help="keep per-track checkpoints after the final merged files are written",
    )
    labels.set_defaults(resume=True, keep_parts=False)
    labels.set_defaults(function=command_generate_labels)
    from cage_system.jailer_cli import (
        command_generate_initial_pool as command_generate_jailer_initial_pool,
        command_train as command_train_jailer,
        command_render_video as command_render_jailer_video,
        command_evaluate_dynamic_cage,
        command_run_jacobian as command_run_constrained_jacobian,
    )
    jailer_pool = subparsers.add_parser("generate-jailer-initial-pool")
    jailer_pool.add_argument("--config", required=True)
    jailer_pool.add_argument("--mujoco-model", required=True)
    jailer_pool.add_argument("--tcp-site", default="attachment_site")
    jailer_pool.add_argument("--count", type=int, default=50)
    jailer_pool.add_argument("--seed", type=int)
    jailer_pool.add_argument("--output", required=True)
    jailer_pool.set_defaults(function=command_generate_jailer_initial_pool)
    jailer_train = subparsers.add_parser("train-jailer")
    jailer_train.add_argument("--config", required=True)
    jailer_train.add_argument("--mujoco-model", required=True)
    jailer_train.add_argument("--tcp-site", default="attachment_site")
    jailer_train.add_argument("--initial-pose-pool")
    jailer_continuation = jailer_train.add_mutually_exclusive_group()
    jailer_continuation.add_argument("--resume-checkpoint")
    jailer_continuation.add_argument("--actor-warm-start-checkpoint")
    jailer_train.add_argument(
        "--resume-start-stage",
        help=(
            "explicit curriculum stage for a full-state resume; preserves "
            "Actor, Critic, optimizer, and normalization statistics"
        ),
    )
    jailer_train.add_argument(
        "--actor-warm-start-stage",
        help=(
            "start this curriculum stage after an Actor-only warm start; "
            "invalid with full resume"
        ),
    )
    jailer_train.add_argument("--output", required=True)
    jailer_train.add_argument("--device", default="auto")
    jailer_train.set_defaults(function=command_train_jailer)
    jailer_video = subparsers.add_parser("render-jailer-video")
    jailer_video.add_argument("--config", required=True)
    jailer_video.add_argument("--checkpoint", required=True)
    jailer_video.add_argument("--mujoco-model", required=True)
    jailer_video.add_argument("--tcp-site", default="attachment_site")
    jailer_video.add_argument("--initial-pose-pool", required=True)
    jailer_video.add_argument("--stage", required=True)
    jailer_video.add_argument("--slot-id", type=int, default=0)
    jailer_video.add_argument("--seed", type=int, default=20260901)
    jailer_video.add_argument("--control-steps", type=int, default=600)
    jailer_video.add_argument("--video-fps", type=float, default=25.0)
    jailer_video.add_argument("--output", required=True)
    jailer_video.add_argument("--device", default="auto")
    jailer_video.set_defaults(function=command_render_jailer_video)
    dynamic_cage = subparsers.add_parser("evaluate-dynamic-cage")
    dynamic_cage.add_argument("--config", required=True)
    dynamic_cage.add_argument("--checkpoint", required=True)
    dynamic_cage.add_argument("--mujoco-model", required=True)
    dynamic_cage.add_argument("--tcp-site", default="attachment_site")
    dynamic_cage.add_argument("--stage")
    dynamic_cage.add_argument("--episodes", type=int, default=20)
    dynamic_cage.add_argument("--seed", type=int, default=20260911)
    dynamic_cage.add_argument("--video-output")
    dynamic_cage.add_argument("--trajectory-output")
    dynamic_cage.add_argument("--video-fps", type=float, default=25.0)
    dynamic_cage.add_argument("--show-window", action="store_true")
    dynamic_cage.add_argument("--realtime-display", action="store_true")
    gripper_visibility = dynamic_cage.add_mutually_exclusive_group()
    gripper_visibility.add_argument(
        "--show-gripper",
        dest="show_gripper",
        action="store_true",
        help="show the physical gripper and Cage region in visualization (default)",
    )
    gripper_visibility.add_argument(
        "--hide-gripper",
        dest="show_gripper",
        action="store_false",
        help="hide gripper/Cage rendering only; physics and control remain unchanged",
    )
    dynamic_cage.add_argument("--output", required=True)
    dynamic_cage.add_argument("--device", default="auto")
    dynamic_cage.set_defaults(
        function=command_evaluate_dynamic_cage,
        show_gripper=True,
    )
    constrained_jacobian = subparsers.add_parser(
        "run-constrained-jacobian"
    )
    constrained_jacobian.add_argument("--config", required=True)
    constrained_jacobian.add_argument("--mujoco-model", required=True)
    constrained_jacobian.add_argument(
        "--tcp-site", default="attachment_site"
    )
    constrained_jacobian.add_argument("--start-joints", nargs=6, type=float)
    constrained_goal = constrained_jacobian.add_mutually_exclusive_group(
        required=True
    )
    constrained_goal.add_argument("--goal-joints", nargs=6, type=float)
    constrained_goal.add_argument(
        "--goal-pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Base-frame attachment_site xyz plus rotation-vector",
    )
    constrained_jacobian.add_argument(
        "--duration", type=float, required=True
    )
    constrained_jacobian.add_argument(
        "--position-only", action="store_true"
    )
    constrained_jacobian.add_argument("--output", required=True)
    constrained_jacobian.set_defaults(
        function=command_run_constrained_jacobian
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)

if __name__ == "__main__":
    main()
