import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from cage_system.config import load_config
from cage_system.io import (
    load_generated_trajectories,
    save_generated_trajectories,
    validate_trajectory_detection_period,
)
from cage_system.pipeline import LabelGenerator
from cage_system.trajectory.generator import PlanarTrajectoryGenerator, trajectory_diagnostics


def make_label_backend(config):
    backend_name = config["labeling"]["feasibility_backend"]
    if backend_name == "workspace":
        return None
    if backend_name == "moveit":
        import rospy
        from cage_system.ros1.moveit_backend import MoveItFeasibilityBackend

        if not rospy.core.is_initialized():
            rospy.init_node("cage_offline_label_generator", anonymous=True, disable_signals=True)
        return MoveItFeasibilityBackend(
            config["robot"], config["labeling"], config["system"]["base_frame"]
        )
    if backend_name == "mujoco":
        from cage_system.sim.mujoco_feasibility import MujocoFeasibilityBackend

        model_path = config["labeling"]["mujoco_model_path"]
        if not model_path:
            raise RuntimeError("labeling.mujoco_model_path is required")
        return MujocoFeasibilityBackend(
            model_path, config["labeling"]["mujoco_tcp_site"], config["robot"]
        )
    raise ValueError("Unknown feasibility backend '{}'".format(backend_name))


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


def command_generate_vision_dataset(args):
    config = load_config(args.config)
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    from cage_system.learning.dataset import CageDatasetArrays
    from cage_system.perception.vision_dataset import VisionCageDatasetBuilder

    trajectories = load_generated_trajectories(args.trajectories)
    geometric_labels = CageDatasetArrays.load(args.labels)
    builder = VisionCageDatasetBuilder(
        config, args.mujoco_model, detector_device=args.device
    )
    try:
        arrays = builder.build(trajectories, geometric_labels, args.audit)
    finally:
        builder.close()
    arrays.save(args.output)
    print(
        json.dumps(
            {
                "geometric_label_count": len(geometric_labels.inputs),
                "vision_label_count": len(arrays.inputs),
                "input_shape": list(arrays.inputs.shape),
                "input_source": arrays.input_source,
                "output": args.output,
                "audit": args.audit,
            },
            ensure_ascii=False,
        )
    )


def command_filter_cage_labels(args):
    from cage_system.learning.dataset import CageDatasetArrays
    from cage_system.learning.dataset_filter import (
        filter_cage_labels_by_history_quality,
    )

    config = load_config(args.config)
    trajectories = load_generated_trajectories(args.trajectories)
    arrays = CageDatasetArrays.load(args.input, allow_unfiltered_legacy=True)
    filtered, summary = filter_cage_labels_by_history_quality(
        config, trajectories, arrays
    )
    filtered.save(args.output)
    print(
        json.dumps(
            {**summary, "output": args.output},
            ensure_ascii=False,
        )
    )
def command_filter_est_empty_labels(args):
    from cage_system.learning.est_label_filter import filter_all_zero_est_labels

    summary = filter_all_zero_est_labels(args.input, args.audit, args.output)
    print(json.dumps({**summary, "output": args.output}, ensure_ascii=False))


def command_train_cage(args):
    config = load_config(args.config)
    try:
        from cage_system.learning.cage_network import train_cage_network
    except ImportError as error:
        raise SystemExit("PyTorch is not installed; install the learning extra") from error
    metrics = train_cage_network(config, args.dataset, args.output, device=args.device)
    print(json.dumps({**metrics, "output": args.output}))


def command_train_ppo(args):
    config = load_config(args.config)
    if config.get("vision", {}).get("enabled", False):
        os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    try:
        from cage_system.learning.cage_network import FrozenCagePredictor
        from cage_system.learning.expert_training import train_expert_guided_ppo
        from cage_system.learning.train_ppo import train_ppo2
        from cage_system.sim.mujoco_env import MujocoUR5eCageEnv, MujocoVisionUR5eCageEnv
    except ImportError as error:
        raise SystemExit("PyTorch/MuJoCo dependencies are missing") from error
    trajectories = load_generated_trajectories(args.trajectories)
    cage = FrozenCagePredictor(args.cage_checkpoint, device=args.device)
    if config.get("vision", {}).get("enabled", False):
        environment = MujocoVisionUR5eCageEnv(
            args.mujoco_model,
            trajectories,
            cage,
            config,
            tcp_site_name=args.tcp_site,
            detector_device=str(cage.device),
        )
    else:
        from cage_system.sim.target_playback import build_frozen_cage_sequences

        target_sequences = build_frozen_cage_sequences(config, trajectories, cage)
        environment = MujocoUR5eCageEnv(
            args.mujoco_model, target_sequences, config, tcp_site_name=args.tcp_site
        )
    try:
        if bool(config.get("expert_training", {}).get("enabled", False)):
            metrics = train_expert_guided_ppo(
                config,
                environment,
                args.output,
                device=args.device,
                resume_checkpoint=args.resume_checkpoint,
            )
        else:
            if args.resume_checkpoint is not None:
                raise ValueError(
                    "--resume-checkpoint currently requires expert_training.enabled"
                )
            metrics = train_ppo2(
                config, environment, args.output, device=args.device
            )
    finally:
        environment.close()
    print(json.dumps({"output": args.output, "last_update": metrics}))


def _assert_residual_validation_config(training_config, validation_config):
    """Allow only dataset/noise seed and declared trajectory count to differ."""
    training_contract = copy.deepcopy(training_config)
    validation_contract = copy.deepcopy(validation_config)
    training_seed = int(training_contract["system"].pop("seed"))
    validation_seed = int(validation_contract["system"].pop("seed"))
    training_contract["trajectory"].pop("count", None)
    validation_contract["trajectory"].pop("count", None)
    if training_seed == validation_seed:
        raise ValueError(
            "Residual validation must use a system.seed different from training"
        )
    if training_contract != validation_contract:
        raise ValueError(
            "Residual training and validation configs may differ only in "
            "system.seed and trajectory.count"
        )
    return {
        "residual_training_seed": training_seed,
        "residual_validation_seed": validation_seed,
        "validation_config_contract_equal": True,
    }


def _fixed_validation_subset(trajectories, episode_count):
    """Spread checkpoint-selection episodes over the complete held-out bank."""
    if not trajectories:
        raise ValueError("Residual validation trajectory archive is empty")
    count = min(int(episode_count), len(trajectories))
    if count == len(trajectories):
        return list(trajectories)
    indices = [
        int(index)
        for index in np.linspace(
            0, len(trajectories) - 1, num=count, dtype=int
        )
    ]
    return [trajectories[index] for index in indices]


def command_train_residual_ppo(args):
    config = load_config(args.config)
    validation_config = load_config(args.validation_config)
    if not bool(config.get("residual_training", {}).get("enabled", False)):
        raise ValueError("The selected config does not enable residual_training")
    if not bool(config.get("vision", {}).get("enabled", False)):
        raise ValueError("Residual PPO requires the causal eye-on-hand environment")
    validation_config_audit = _assert_residual_validation_config(
        config, validation_config
    )
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    try:
        from cage_system.learning.cage_network import FrozenCagePredictor
        from cage_system.learning.expert_training import _assert_c7_reset
        from cage_system.learning.train_residual_ppo import (
            assert_disjoint_trajectory_sets,
            train_residual_ppo,
        )
        from cage_system.sim.mujoco_env import MujocoVisionUR5eCageEnv
    except ImportError as error:
        raise SystemExit("PyTorch/MuJoCo dependencies are missing") from error

    expert_trajectories = load_generated_trajectories(args.expert_trajectories)
    training_trajectories = load_generated_trajectories(args.trajectories)
    validation_trajectories = load_generated_trajectories(
        args.validation_trajectories
    )
    validation_evaluation_trajectories = _fixed_validation_subset(
        validation_trajectories,
        int(config["residual_training"]["validation_episodes"]),
    )
    dataset_audit = assert_disjoint_trajectory_sets(
        {
            "expert_actor": expert_trajectories,
            "residual_train": training_trajectories,
            "residual_validation": validation_trajectories,
        }
    )
    dataset_audit.update(
        {
            **validation_config_audit,
            "residual_training_config": str(Path(args.config).resolve()),
            "residual_validation_config": str(
                Path(args.validation_config).resolve()
            ),
            "expert_trajectory_archive": str(
                Path(args.expert_trajectories).resolve()
            ),
            "residual_training_archive": str(Path(args.trajectories).resolve()),
            "residual_validation_archive": str(
                Path(args.validation_trajectories).resolve()
            ),
            "residual_validation_selection_track_ids": [
                int(item.track_id) for item in validation_evaluation_trajectories
            ],
        }
    )
    cage = FrozenCagePredictor(args.cage_checkpoint, device=args.device)

    def make_environment(trajectories, environment_config):
        return MujocoVisionUR5eCageEnv(
            args.mujoco_model,
            trajectories,
            cage,
            environment_config,
            tcp_site_name=args.tcp_site,
            detector_device=str(cage.device),
        )

    environment = make_environment(training_trajectories, config)
    try:
        metrics = train_residual_ppo(
            config,
            environment,
            lambda: make_environment(
                validation_evaluation_trajectories, validation_config
            ),
            args.base_actor_checkpoint,
            args.critic_checkpoint,
            args.output,
            device=args.device,
            resume_checkpoint=args.resume_checkpoint,
            reset_validator=lambda info: _assert_c7_reset(info, config),
            dataset_audit=dataset_audit,
        )
    finally:
        environment.close()
    print(json.dumps({"output": args.output, "last_update": metrics}))


def command_evaluate_residual_checkpoints(args):
    config = load_config(args.config)
    if not bool(config.get("residual_training", {}).get("enabled", False)):
        raise ValueError("The selected config does not enable residual_training")
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    from cage_system.learning.cage_network import FrozenCagePredictor
    from cage_system.learning.train_residual_ppo import (
        evaluate_residual_checkpoints,
    )
    from cage_system.sim.mujoco_env import MujocoVisionUR5eCageEnv

    trajectories = load_generated_trajectories(args.validation_trajectories)
    trajectories = _fixed_validation_subset(trajectories, int(args.episodes))
    cage = FrozenCagePredictor(args.cage_checkpoint, device=args.device)

    def make_environment():
        return MujocoVisionUR5eCageEnv(
            args.mujoco_model,
            trajectories,
            cage,
            config,
            tcp_site_name=args.tcp_site,
            detector_device=str(cage.device),
        )

    report = evaluate_residual_checkpoints(
        args.checkpoints,
        make_environment,
        int(args.episodes),
        device=args.device,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checkpoint_count": len(report["checkpoints"]),
                "validation_episodes": int(args.episodes),
                "best_reward_checkpoint": report["best_reward_checkpoint"],
                "best_nondegrading_checkpoint": report[
                    "best_nondegrading_checkpoint"
                ],
                "output": str(destination),
            },
            ensure_ascii=False,
        )
    )


def command_infer_visualize(args):
    config = load_config(args.config)
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    from cage_system.inference_visualization import run_inference_visualization

    result = run_inference_visualization(
        config,
        args.mujoco_model,
        args.trajectories,
        args.cage_checkpoint,
        args.ppo_checkpoint,
        args.output,
        tcp_site=args.tcp_site,
        device=args.device,
        track_id=args.track_id,
        snapshot_count=args.snapshot_count,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, ensure_ascii=False))


def command_render_eye(args):
    config = load_config(args.config)
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np

    from cage_system.perception.eye_on_hand import MujocoEyeOnHandCamera

    model = mujoco.MjModel.from_xml_path(args.mujoco_model)
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in config["robot"]["joint_names"]
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise RuntimeError("The rendered model does not contain all configured UR5e joints")
    qpos_addresses = model.jnt_qposadr[np.asarray(joint_ids)]
    data.qpos[qpos_addresses] = np.asarray(
        config["robot"]["nominal_joint_state_rad"], dtype=float
    )
    mujoco.mj_forward(model, data)
    vision_config = copy.deepcopy(config["vision"])
    vision_config["frame_dropout_probability"] = 0.0
    vision_config["depth_pixel_dropout_probability"] = 0.0
    vision_config["depth_noise_std_m"] = 0.0
    camera = MujocoEyeOnHandCamera(model, vision_config, seed=0, detector_device=args.device)
    try:
        detection = camera.capture(data, 0.0)
    finally:
        camera.close()
    target_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, vision_config["target_mocap_body"]
    )
    mocap_id = model.body_mocapid[target_body]
    target_truth = data.mocap_pos[mocap_id].copy()
    error = (
        float(np.linalg.norm(detection.observation.position - target_truth))
        if detection.detected
        else float("inf")
    )
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(detection.rgb)
    axes[0].set_title("eye-on-hand RGB")
    axes[1].imshow(detection.depth_m, cmap="viridis")
    axes[1].set_title("depth [m]")
    axes[2].imshow(detection.detection_mask, cmap="gray")
    axes[2].set_title("detector mask")
    truth_mask = detection.simulation_target_mask
    axes[3].imshow(
        np.zeros_like(detection.detection_mask) if truth_mask is None else truth_mask,
        cmap="gray",
    )
    axes[3].set_title("simulation audit mask")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(destination), dpi=170)
    plt.close(figure)
    print(
        json.dumps(
            {
                "detected": detection.detected,
                "mask_pixels": detection.mask_pixels,
                "estimated_center_base": detection.observation.position.tolist(),
                "true_center_base_for_audit_only": target_truth.tolist(),
                "center_error_m": error,
                "output": str(destination),
            }
        )
    )


def command_visualize_system(args):
    config = load_config(args.config)
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    from cage_system.system_visualization import render_complete_system

    result = render_complete_system(
        config,
        args.mujoco_model,
        args.trajectories,
        args.labels,
        args.output,
        sample_index=args.sample_index,
        detector_device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False))


def command_visualize_trajectory_cages(args):
    config = load_config(args.config)
    from cage_system.trajectory_cage_visualization import visualize_trajectory_cages

    result = visualize_trajectory_cages(
        config,
        args.trajectories,
        args.labels,
        args.output,
        track_id=args.track_id,
        pose_count=args.pose_count,
    )
    print(json.dumps(result, ensure_ascii=False))


def command_visualize_trajectory_robot_cages(args):
    config = load_config(args.config)
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    from cage_system.system_visualization import render_trajectory_robot_cages

    result = render_trajectory_robot_cages(
        config,
        args.mujoco_model,
        args.trajectories,
        args.labels,
        args.output,
        track_id=args.track_id,
        pose_count=args.pose_count,
    )
    print(json.dumps(result, ensure_ascii=False))


def command_visualize_random_cage_segment(args):
    config = load_config(args.config)
    os.environ.setdefault("MUJOCO_GL", str(config["vision"].get("mujoco_gl", "egl")))
    from cage_system.system_visualization import render_trajectory_robot_cages

    result = render_trajectory_robot_cages(
        config,
        args.mujoco_model,
        args.trajectories,
        args.labels,
        args.output,
        track_id=args.track_id,
        pose_count=args.pose_count,
        random_segment=True,
        random_seed=args.seed,
        include_eye_on_hand=True,
    )
    print(json.dumps(result, ensure_ascii=False))


def command_smoke(args):
    config = load_config(args.config)
    smoke = copy.deepcopy(config)
    smoke["trajectory"]["duration_s"] = [2.5, 2.5]
    smoke["trajectory"]["dropout_probability"] = 0.0
    smoke["trajectory"]["outlier_probability"] = 0.0
    smoke["est"]["node_count"] = min(int(smoke["est"]["node_count"]), 80)
    smoke["candidate"]["nominal_samples_per_arc"] = 5
    smoke["candidate"]["orientation_branches"] = [1]
    generator = PlanarTrajectoryGenerator(smoke["trajectory"], seed=11)
    trajectories = [generator.generate(0)]
    temporary = Path(args.output_dir)
    temporary.mkdir(parents=True, exist_ok=True)
    trajectory_path = str(temporary / "smoke_trajectories.npz")
    label_path = str(temporary / "smoke_labels.npz")
    audit_path = str(temporary / "smoke_audit.jsonl")
    save_generated_trajectories(trajectories, trajectory_path)
    # Keep smoke behavior identical to the normal label command: production
    # configurations must exercise their configured IK/collision backend rather
    # than silently falling back to the workspace approximation.
    arrays = LabelGenerator(smoke, make_label_backend(smoke)).generate(
        trajectories, audit_path
    )
    arrays.save(label_path)
    result = {
        "trajectory_count": 1,
        "valid_label_count": len(arrays.inputs),
        "input_shape": list(arrays.inputs.shape),
        "label_shape": list(arrays.labels.shape),
        "history_excludes_current": True,
        "object_model": "point_mass",
        "artifacts": [trajectory_path, label_path, audit_path],
    }
    print(json.dumps(result, ensure_ascii=False))
    if len(arrays.inputs) == 0:
        raise SystemExit("Smoke test generated no valid labels; inspect the audit")


def command_visualize(args):
    config = load_config(args.config)
    from cage_system.visualization import load_audit_record, visualize_audit_record

    record = load_audit_record(args.audit, args.track_id, args.sample_index)
    visualize_audit_record(record, config, args.output)


def build_parser():
    parser = argparse.ArgumentParser(description="Dynamic Cage and PPO2 pipeline")
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
    vision_labels = subparsers.add_parser("generate-vision-dataset")
    vision_labels.add_argument("--config", required=True)
    vision_labels.add_argument("--trajectories", required=True)
    vision_labels.add_argument("--labels", required=True)
    vision_labels.add_argument("--mujoco-model", required=True)
    vision_labels.add_argument("--output", required=True)
    vision_labels.add_argument("--audit", required=True)
    vision_labels.add_argument("--device", default="auto")
    vision_labels.set_defaults(function=command_generate_vision_dataset)
    filter_labels = subparsers.add_parser("filter-cage-labels")
    filter_labels.add_argument("--config", required=True)
    filter_labels.add_argument("--trajectories", required=True)
    filter_labels.add_argument("--input", required=True)
    filter_labels.add_argument("--output", required=True)
    filter_labels.set_defaults(function=command_filter_cage_labels)
    filter_est = subparsers.add_parser("filter-est-empty-labels")
    filter_est.add_argument("--input", required=True)
    filter_est.add_argument("--audit", required=True)
    filter_est.add_argument("--output", required=True)
    filter_est.set_defaults(function=command_filter_est_empty_labels)
    cage = subparsers.add_parser("train-cage")
    cage.add_argument("--config", required=True)
    cage.add_argument("--dataset", required=True)
    cage.add_argument("--output", required=True)
    cage.add_argument("--device", default="auto")
    cage.set_defaults(function=command_train_cage)
    ppo = subparsers.add_parser("train-ppo")
    ppo.add_argument("--config", required=True)
    ppo.add_argument("--cage-checkpoint", required=True)
    ppo.add_argument("--trajectories", required=True)
    ppo.add_argument("--mujoco-model", required=True)
    ppo.add_argument("--tcp-site", default="attachment_site")
    ppo.add_argument("--device", default="auto")
    ppo.add_argument(
        "--resume-checkpoint",
        help=(
            "Resume formal PPO from critic_warmup.pt or a PPO latest/update "
            "checkpoint; completed expert stages are not repeated"
        ),
    )
    ppo.add_argument("--output", required=True)
    ppo.set_defaults(function=command_train_ppo)
    from cage_system.jailer_cli import (
        command_evaluate_dynamic_cage,
        command_generate_initial_pool as command_generate_jailer_initial_pool,
        command_render_video as command_render_jailer_video,
        command_run_jacobian as command_run_constrained_jacobian,
        command_train as command_train_jailer,
    )

    jailer_pool = subparsers.add_parser("generate-jailer-initial-pool")
    jailer_pool.add_argument("--config", required=True)
    jailer_pool.add_argument("--mujoco-model", required=True)
    jailer_pool.add_argument("--tcp-site", default="attachment_site")
    jailer_pool.add_argument("--count", type=int, default=40)
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
    residual_ppo = subparsers.add_parser("train-residual-ppo")
    residual_ppo.add_argument("--config", required=True)
    residual_ppo.add_argument("--validation-config", required=True)
    residual_ppo.add_argument("--cage-checkpoint", required=True)
    residual_ppo.add_argument("--base-actor-checkpoint", required=True)
    residual_ppo.add_argument("--critic-checkpoint", required=True)
    residual_ppo.add_argument(
        "--expert-trajectories",
        required=True,
        help="trajectory archive used by final DAgger; exact reuse is rejected",
    )
    residual_ppo.add_argument("--trajectories", required=True)
    residual_ppo.add_argument("--validation-trajectories", required=True)
    residual_ppo.add_argument("--mujoco-model", required=True)
    residual_ppo.add_argument("--tcp-site", default="attachment_site")
    residual_ppo.add_argument("--device", default="auto")
    residual_ppo.add_argument("--resume-checkpoint")
    residual_ppo.add_argument("--output", required=True)
    residual_ppo.set_defaults(function=command_train_residual_ppo)
    residual_eval = subparsers.add_parser("evaluate-residual-checkpoints")
    residual_eval.add_argument("--config", required=True)
    residual_eval.add_argument("--cage-checkpoint", required=True)
    residual_eval.add_argument("--validation-trajectories", required=True)
    residual_eval.add_argument("--mujoco-model", required=True)
    residual_eval.add_argument("--tcp-site", default="attachment_site")
    residual_eval.add_argument("--device", default="auto")
    residual_eval.add_argument("--episodes", type=int, default=20)
    residual_eval.add_argument("--checkpoints", nargs="+", required=True)
    residual_eval.add_argument("--output", required=True)
    residual_eval.set_defaults(function=command_evaluate_residual_checkpoints)
    inference = subparsers.add_parser("infer-visualize")
    inference.add_argument("--config", required=True)
    inference.add_argument("--cage-checkpoint", required=True)
    inference.add_argument("--ppo-checkpoint", required=True)
    inference.add_argument("--trajectories", required=True)
    inference.add_argument("--mujoco-model", required=True)
    inference.add_argument("--tcp-site", default="attachment_site")
    inference.add_argument("--device", default="auto")
    inference.add_argument(
        "--track-id",
        type=int,
        default=-1,
        help="-1 selects the first feasible track; otherwise require this track",
    )
    inference.add_argument("--snapshot-count", type=int, default=6)
    inference.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 runs the complete selected trajectory",
    )
    inference.add_argument("--output", required=True)
    inference.set_defaults(function=command_infer_visualize)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--output-dir", default="artifacts/smoke")
    smoke.set_defaults(function=command_smoke)
    visualize = subparsers.add_parser("visualize")
    visualize.add_argument("--config", required=True)
    visualize.add_argument("--audit", required=True)
    visualize.add_argument("--track-id", type=int, required=True)
    visualize.add_argument("--sample-index", type=int, required=True)
    visualize.add_argument("--output")
    visualize.set_defaults(function=command_visualize)
    render_eye = subparsers.add_parser("render-eye")
    render_eye.add_argument("--config", required=True)
    render_eye.add_argument("--mujoco-model", required=True)
    render_eye.add_argument("--device", default="auto")
    render_eye.add_argument("--output", required=True)
    render_eye.set_defaults(function=command_render_eye)
    visualize_system = subparsers.add_parser("visualize-system")
    visualize_system.add_argument("--config", required=True)
    visualize_system.add_argument("--trajectories", required=True)
    visualize_system.add_argument("--labels", required=True)
    visualize_system.add_argument("--mujoco-model", required=True)
    visualize_system.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="dataset row; -1 automatically chooses the first collision-free label",
    )
    visualize_system.add_argument("--device", default="auto")
    visualize_system.add_argument("--output", required=True)
    visualize_system.set_defaults(function=command_visualize_system)
    trajectory_cages = subparsers.add_parser("visualize-trajectory-cages")
    trajectory_cages.add_argument("--config", required=True)
    trajectory_cages.add_argument("--trajectories", required=True)
    trajectory_cages.add_argument("--labels", required=True)
    trajectory_cages.add_argument("--track-id", type=int, default=-1)
    trajectory_cages.add_argument("--pose-count", type=int, default=8)
    trajectory_cages.add_argument("--output", required=True)
    trajectory_cages.set_defaults(function=command_visualize_trajectory_cages)
    trajectory_robot_cages = subparsers.add_parser(
        "visualize-trajectory-robot-cages"
    )
    trajectory_robot_cages.add_argument("--config", required=True)
    trajectory_robot_cages.add_argument("--trajectories", required=True)
    trajectory_robot_cages.add_argument("--labels", required=True)
    trajectory_robot_cages.add_argument("--mujoco-model", required=True)
    trajectory_robot_cages.add_argument("--track-id", type=int, default=-1)
    trajectory_robot_cages.add_argument("--pose-count", type=int, default=6)
    trajectory_robot_cages.add_argument("--output", required=True)
    trajectory_robot_cages.set_defaults(
        function=command_visualize_trajectory_robot_cages
    )
    random_cage_segment = subparsers.add_parser(
        "visualize-random-cage-segment"
    )
    random_cage_segment.add_argument("--config", required=True)
    random_cage_segment.add_argument("--trajectories", required=True)
    random_cage_segment.add_argument("--labels", required=True)
    random_cage_segment.add_argument("--mujoco-model", required=True)
    random_cage_segment.add_argument(
        "--track-id",
        type=int,
        default=-1,
        help="-1 chooses a random feasible track; nonnegative fixes the track",
    )
    random_cage_segment.add_argument("--pose-count", type=int, default=4)
    random_cage_segment.add_argument("--seed", type=int, default=42)
    random_cage_segment.add_argument("--output", required=True)
    random_cage_segment.set_defaults(
        function=command_visualize_random_cage_segment
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
