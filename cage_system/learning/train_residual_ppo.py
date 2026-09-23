"""Rollout, validation and checkpoint orchestration for residual PPO."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from cage_system.learning.ppo2 import PPO_REWARD_TARGET_CONTRACT
from cage_system.learning.residual_ppo import (
    FrozenResidualPolicy,
    ResidualPPOTrainer,
    ResidualRolloutBatch,
    load_frozen_residual_ppo,
)


def _trajectory_fingerprint(trajectory):
    digest = hashlib.sha256()
    digest.update(np.asarray(trajectory.timestamps, dtype="<f8").tobytes())
    digest.update(np.asarray(trajectory.true_positions, dtype="<f8").tobytes())
    return digest.hexdigest()


def assert_disjoint_trajectory_sets(named_trajectory_sets):
    """Reject exact curve reuse even though independent archives reuse track ids."""
    fingerprints = {}
    summary = {}
    for name, trajectories in named_trajectory_sets.items():
        current = {_trajectory_fingerprint(item) for item in trajectories}
        if len(current) != len(trajectories):
            raise ValueError("Trajectory set '{}' contains duplicate curves".format(name))
        fingerprints[name] = current
        summary[name] = int(len(current))
    names = list(fingerprints)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            overlap = fingerprints[first].intersection(fingerprints[second])
            if overlap:
                raise ValueError(
                    "Trajectory sets '{}' and '{}' share {} exact curves".format(
                        first, second, len(overlap)
                    )
                )
    return {
        "trajectory_sets_disjoint": True,
        "trajectory_set_counts": summary,
        "disjointness_contract": "sha256(true_timestamps,true_positions)_v1",
    }


def _validate_same_cage_transition(raw_state, raw_next_state, info):
    reward_pose = np.asarray(info.get("reward_cage_pose6", []), dtype=float)
    next_pose = np.asarray(info.get("next_cage_pose6", []), dtype=float)
    action_relative = np.asarray(
        info.get("action_state_relative_error6", []), dtype=float
    )
    next_relative = np.asarray(
        info.get("next_state_relative_error6", []), dtype=float
    )
    if (
        info.get("reward_target_contract") != PPO_REWARD_TARGET_CONTRACT
        or reward_pose.shape != (6,)
        or next_pose.shape != (6,)
        or action_relative.shape != (6,)
        or next_relative.shape != (6,)
        or not np.array_equal(
            action_relative.astype(np.float32),
            np.asarray(raw_state[:6], dtype=np.float32),
        )
        or not np.array_equal(
            next_relative.astype(np.float32),
            np.asarray(raw_next_state[:6], dtype=np.float32),
        )
    ):
        raise RuntimeError(
            "Residual transition violated the same-Cage reward contract"
        )


def collect_residual_rollout(
    trainer,
    environment,
    minimum_steps,
    progress_bar=None,
    reset_validator=None,
):
    states, residual_actions, base_actions = [], [], []
    preclip_actions, executed_actions = [], []
    rewards, next_states, dones = [], [], []
    old_logs, old_values, next_values = [], [], []
    while len(states) < int(minimum_steps):
        raw_state, reset_info = environment.reset()
        if reset_validator is not None:
            reset_validator(reset_info)
        done = False
        while not done:
            policy_decision = bool(getattr(environment, "action_ready", True))
            decision = (
                trainer.decide(raw_state, deterministic=False)
                if policy_decision
                else None
            )
            action = (
                decision.executed_action
                if decision is not None
                else np.zeros(6, dtype=np.float32)
            )
            raw_next_state, reward, terminated, truncated, info = environment.step(
                action
            )
            done = bool(terminated or truncated)
            next_ready = bool(getattr(environment, "action_ready", True))
            if policy_decision:
                actual = np.asarray(
                    info.get("executed_normalized_action", action), dtype=np.float32
                )
                if actual.shape != (6,) or not np.allclose(
                    actual, decision.executed_action, atol=1e-7, rtol=0.0
                ):
                    raise RuntimeError(
                        "The environment changed the composed residual action"
                    )
                _validate_same_cage_transition(raw_state, raw_next_state, info)
                training_boundary = bool(
                    done
                    or not next_ready
                    or info.get("rollout_boundary", False)
                )
                next_normalized = (
                    trainer.normalize_rl(raw_next_state)
                    if next_ready
                    else np.zeros(18, dtype=np.float32)
                )
                states.append(trainer.normalize_rl(raw_state))
                residual_actions.append(decision.residual_action)
                base_actions.append(decision.base_action)
                preclip_actions.append(decision.preclip_action)
                executed_actions.append(decision.executed_action)
                rewards.append(float(reward))
                next_states.append(next_normalized)
                dones.append(float(training_boundary))
                old_logs.append(decision.residual_log_density)
                old_values.append(decision.value)
                next_values.append(
                    0.0 if training_boundary else trainer.value(raw_next_state)
                )
                if progress_bar is not None and progress_bar.n < int(minimum_steps):
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        accepted=len(states),
                        reward=f"{float(reward):.3f}",
                        refresh=False,
                    )
            raw_state = raw_next_state
    return ResidualRolloutBatch(
        states=np.asarray(states, dtype=np.float32),
        residual_actions=np.asarray(residual_actions, dtype=np.float32),
        base_actions=np.asarray(base_actions, dtype=np.float32),
        preclip_actions=np.asarray(preclip_actions, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        old_log_densities=np.asarray(old_logs, dtype=np.float32),
        old_values=np.asarray(old_values, dtype=np.float32),
        next_values=np.asarray(next_values, dtype=np.float32),
    )


def _deterministic_components(policy, raw_state):
    if isinstance(policy, ResidualPPOTrainer):
        decision = policy.decide(raw_state, deterministic=True)
        return (
            decision.base_action,
            decision.residual_action,
            decision.preclip_action,
            decision.executed_action,
        )
    if isinstance(policy, FrozenResidualPolicy):
        state = torch.as_tensor(
            raw_state, dtype=torch.float32, device=policy.device
        ).unsqueeze(0)
        with torch.inference_mode():
            executed, _, _, _, base, residual, preclip = policy.decompose(
                state, deterministic=True
            )
        return tuple(
            item.cpu().numpy()[0].astype(np.float32)
            for item in (base, residual, preclip, executed)
        )
    raise TypeError("Validation requires a residual trainer or frozen residual policy")


def evaluate_residual_policy(policy, environment, episodes, mode):
    if mode not in {"base_only", "base_plus_residual"}:
        raise ValueError("Unknown residual validation mode '{}'".format(mode))
    rewards, positions, orientations = [], [], []
    actions, residuals, composition_clips = [], [], []
    successes, detections, valid_histories = [], [], []
    collisions = 0
    visual_escapes = 0
    true_visual_escapes = 0
    policy_decisions = 0
    total_steps = 0
    track_ids = []
    for _ in range(int(episodes)):
        raw_state, reset_info = environment.reset()
        track_ids.append(int(reset_info.get("trajectory_id", -1)))
        done = False
        guard = 0
        while not done:
            guard += 1
            if guard > 10000:
                raise RuntimeError("Validation episode exceeded its safety step limit")
            ready = bool(getattr(environment, "action_ready", True))
            if ready:
                base, residual, preclip, composite = _deterministic_components(
                    policy, raw_state
                )
                if mode == "base_only":
                    action = base
                    preclip_for_metric = base
                    residual_for_metric = np.zeros(6, dtype=np.float32)
                else:
                    action = composite
                    preclip_for_metric = preclip
                    residual_for_metric = residual
                policy_decisions += 1
            else:
                action = np.zeros(6, dtype=np.float32)
                preclip_for_metric = action
                residual_for_metric = action
            raw_state, reward, terminated, truncated, info = environment.step(action)
            done = bool(terminated or truncated)
            total_steps += 1
            rewards.append(float(reward))
            positions.append(float(info["position_error_m"]))
            orientations.append(float(np.degrees(info["orientation_error_rad"])))
            successes.append(
                bool(
                    positions[-1]
                    <= float(environment.ppo["success_position_m"])
                    and orientations[-1]
                    <= float(environment.ppo["success_orientation_deg"])
                )
            )
            actions.append(float(np.max(np.abs(action))))
            residuals.append(float(np.mean(np.abs(residual_for_metric))))
            composition_clips.append(bool(np.any(np.abs(preclip_for_metric) > 1.0)))
            detections.append(bool(info.get("vision_detected", True)))
            valid_histories.append(bool(info.get("history_quality_valid", True)))
            collisions += int(bool(info.get("contact_collision", False)))
            visual_escapes += int(bool(info.get("visual_escape", False)))
            true_visual_escapes += int(
                bool(info.get("true_visual_escape", False))
            )
    if not rewards:
        raise RuntimeError("Residual validation produced no transitions")
    return {
        "mode": mode,
        "episodes": int(episodes),
        "steps": int(total_steps),
        "track_ids": track_ids,
        "reward_mean": float(np.mean(rewards)),
        "reward_sum": float(np.sum(rewards)),
        "position_error_mean_mm": float(1000.0 * np.mean(positions)),
        "position_error_p95_mm": float(1000.0 * np.percentile(positions, 95.0)),
        "orientation_error_mean_deg": float(np.mean(orientations)),
        "orientation_error_p95_deg": float(np.percentile(orientations, 95.0)),
        "pose_success_rate": float(np.mean(successes)),
        "vision_detection_rate": float(np.mean(detections)),
        "valid_history_rate": float(np.mean(valid_histories)),
        "policy_inference_rate": float(policy_decisions / max(total_steps, 1)),
        "collision_steps": int(collisions),
        "visual_escapes": int(visual_escapes),
        "true_visual_escapes": int(true_visual_escapes),
        "max_action_mean": float(np.mean(actions)),
        "residual_mean_abs": float(np.mean(residuals)),
        "composition_clip_rate": float(np.mean(composition_clips)),
    }


def validation_comparison(base_metrics, residual_metrics):
    comparison = {
        "reward_mean_gain": float(
            residual_metrics["reward_mean"] - base_metrics["reward_mean"]
        ),
        "position_mean_improvement_mm": float(
            base_metrics["position_error_mean_mm"]
            - residual_metrics["position_error_mean_mm"]
        ),
        "orientation_mean_improvement_deg": float(
            base_metrics["orientation_error_mean_deg"]
            - residual_metrics["orientation_error_mean_deg"]
        ),
        "pose_success_rate_gain": float(
            residual_metrics["pose_success_rate"]
            - base_metrics["pose_success_rate"]
        ),
        "visual_escape_reduction": int(
            base_metrics["visual_escapes"] - residual_metrics["visual_escapes"]
        ),
        "true_visual_escape_reduction": int(
            base_metrics["true_visual_escapes"]
            - residual_metrics["true_visual_escapes"]
        ),
        "collision_step_reduction": int(
            base_metrics["collision_steps"] - residual_metrics["collision_steps"]
        ),
    }
    comparison["passes_strict_base_nondegradation_guard"] = bool(
        comparison["reward_mean_gain"] >= -1e-9
        and comparison["position_mean_improvement_mm"] >= -1e-6
        and comparison["orientation_mean_improvement_deg"] >= -1e-6
        and comparison["pose_success_rate_gain"] >= -1e-9
        and comparison["true_visual_escape_reduction"] >= 0
        and comparison["collision_step_reduction"] >= 0
    )
    return comparison


def _append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _rewind_jsonl_for_resume(path, maximum_update):
    """Keep history consistent when resuming from an older checkpoint."""
    path = Path(path)
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    records = [json.loads(line) for line in lines]
    retained = [item for item in records if int(item.get("update", -1)) <= maximum_update]
    if len(retained) == len(records):
        return None
    suffix = ".before_resume_from_{:04d}.jsonl".format(int(maximum_update))
    archive = path.with_name(path.stem + suffix)
    counter = 1
    while archive.exists():
        archive = path.with_name(
            path.stem
            + ".before_resume_from_{:04d}_{:02d}.jsonl".format(
                int(maximum_update), counter
            )
        )
        counter += 1
    path.replace(archive)
    serialized = "".join(
        json.dumps(item, sort_keys=True) + "\n" for item in retained
    )
    path.write_text(serialized, encoding="utf-8")
    return str(archive)


def train_residual_ppo(
    config,
    environment,
    validation_environment_factory,
    base_checkpoint,
    critic_checkpoint,
    output_path,
    device="auto",
    resume_checkpoint=None,
    reset_validator=None,
    dataset_audit=None,
):
    residual = config["residual_training"]
    trainer = ResidualPPOTrainer(
        config, base_checkpoint, critic_checkpoint, device=device
    )
    destination = Path(output_path)
    checkpoint_directory = destination.parent / (
        destination.stem + "_checkpoints"
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    training_history = checkpoint_directory / "training_history.jsonl"
    validation_history = checkpoint_directory / "validation_history.jsonl"
    base_validation_path = checkpoint_directory / "base_validation.json"
    start_update = 0
    initialization = dict(dataset_audit or {})
    if resume_checkpoint is None:
        for path in (training_history, validation_history):
            if path.exists():
                path.unlink()
        alignment_progress = tqdm(
            total=int(residual["critic_alignment_steps"]),
            desc="Residual Critic alignment rollout",
            unit="transition",
            dynamic_ncols=True,
        )
        try:
            alignment_rollout = collect_residual_rollout(
                trainer,
                environment,
                int(residual["critic_alignment_steps"]),
                progress_bar=alignment_progress,
                reset_validator=reset_validator,
            )
        finally:
            alignment_progress.close()
        alignment_epoch_progress = tqdm(
            total=int(residual["critic_alignment_epochs"]),
            desc="Residual Critic alignment fit",
            unit="epoch",
            dynamic_ncols=True,
        )
        try:
            alignment_metrics = trainer.align_critic(
                alignment_rollout, progress_bar=alignment_epoch_progress
            )
        finally:
            alignment_epoch_progress.close()
        trainer.critic_optimizer = torch.optim.Adam(
            trainer.critic.parameters(),
            lr=float(residual["critic_learning_rate"]),
        )
        initialization.update(alignment_metrics)
        trainer.save(
            str(checkpoint_directory / "critic_aligned.pt"),
            training_metadata={"stage": "critic_alignment", **initialization},
            include_optimizers=True,
        )
    else:
        metadata, optimizer_restored = trainer.restore(resume_checkpoint)
        if not optimizer_restored:
            raise ValueError(
                "Residual PPO resume requires a checkpoint with both optimizer states"
            )
        start_update = int(metadata.get("update", 0))
        initialization.update(metadata)
        initialization["resume_checkpoint"] = str(Path(resume_checkpoint).resolve())
        archived_histories = {}
        for name, path in (
            ("training", training_history),
            ("validation", validation_history),
        ):
            archive = _rewind_jsonl_for_resume(path, start_update)
            if archive is not None:
                archived_histories[name] = archive
        if archived_histories:
            initialization["resume_archived_histories"] = archived_histories

    validation_episodes = int(residual["validation_episodes"])
    base_environment = validation_environment_factory()
    try:
        base_metrics = evaluate_residual_policy(
            trainer, base_environment, validation_episodes, "base_only"
        )
    finally:
        base_environment.close()
    base_validation_path.write_text(
        json.dumps(base_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_score = float("-inf")
    best_update = -1
    if resume_checkpoint is None:
        initial_environment = validation_environment_factory()
        try:
            initial_residual = evaluate_residual_policy(
                trainer,
                initial_environment,
                validation_episodes,
                "base_plus_residual",
            )
        finally:
            initial_environment.close()
        initial_validation = {
            "update": 0,
            "checkpoint": str(checkpoint_directory / "critic_aligned.pt"),
            "base": base_metrics,
            "base_plus_residual": initial_residual,
            "improvement": validation_comparison(base_metrics, initial_residual),
        }
        _append_jsonl(validation_history, initial_validation)
        best_score = float(initial_residual["reward_mean"])
        best_update = 0
        trainer.save(
            str(checkpoint_directory / "best_validation.pt"),
            training_metadata={
                "update": 0,
                "validation": initial_validation,
                **initialization,
            },
            include_optimizers=True,
        )
    elif validation_history.exists():
        records = [
            json.loads(line)
            for line in validation_history.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if records:
            for record in records:
                record["improvement"] = validation_comparison(
                    record["base"], record["base_plus_residual"]
                )
            eligible_records = [
                record
                for record in records
                if record["improvement"][
                    "passes_strict_base_nondegradation_guard"
                ]
            ]
            if eligible_records:
                best_record = max(
                    eligible_records,
                    key=lambda item: item["base_plus_residual"]["reward_mean"],
                )
                best_score = float(
                    best_record["base_plus_residual"]["reward_mean"]
                )
                best_update = int(best_record["update"])

    updates = int(config["ppo"]["updates"])
    rollout_steps = int(config["ppo"]["rollout_steps"])
    checkpoint_interval = int(residual["checkpoint_interval_updates"])
    validation_interval = int(residual["validation_interval_updates"])
    progress = tqdm(
        range(start_update, updates),
        total=updates,
        initial=start_update,
        desc="Residual PPO updates",
        unit="update",
        dynamic_ncols=True,
    )
    last_metrics = {}
    for update_index in progress:
        rollout_progress = tqdm(
            total=rollout_steps,
            desc="residual rollout {:04d}/{:04d}".format(
                update_index + 1, updates
            ),
            unit="transition",
            leave=False,
            dynamic_ncols=True,
        )
        try:
            rollout = collect_residual_rollout(
                trainer,
                environment,
                rollout_steps,
                progress_bar=rollout_progress,
                reset_validator=reset_validator,
            )
        finally:
            rollout_progress.close()
        last_metrics = trainer.update(rollout)
        last_metrics.update(
            {
                "update": int(update_index + 1),
                "rollout_samples": int(len(rollout.states)),
                "rollout_reward_mean": float(np.mean(rollout.rewards)),
                "rollout_reward_sum": float(np.sum(rollout.rewards)),
            }
        )
        metadata = {**initialization, **last_metrics}
        trainer.save(
            str(checkpoint_directory / "latest.pt"),
            training_metadata=metadata,
            include_optimizers=True,
        )
        should_checkpoint = (
            (update_index + 1) % checkpoint_interval == 0
            or update_index + 1 == updates
        )
        checkpoint_path = checkpoint_directory / "update_{:04d}.pt".format(
            update_index + 1
        )
        if should_checkpoint:
            trainer.save(
                str(checkpoint_path),
                training_metadata=metadata,
                include_optimizers=True,
            )
        if (
            (update_index + 1) % validation_interval == 0
            or update_index + 1 == updates
        ):
            validation_environment = validation_environment_factory()
            try:
                residual_metrics = evaluate_residual_policy(
                    trainer,
                    validation_environment,
                    validation_episodes,
                    "base_plus_residual",
                )
            finally:
                validation_environment.close()
            validation_record = {
                "update": int(update_index + 1),
                "checkpoint": str(
                    checkpoint_path
                    if should_checkpoint
                    else checkpoint_directory / "latest.pt"
                ),
                "base": base_metrics,
                "base_plus_residual": residual_metrics,
                "improvement": validation_comparison(base_metrics, residual_metrics),
            }
            _append_jsonl(validation_history, validation_record)
            last_metrics["validation"] = validation_record
            validation_metadata = {
                **metadata,
                "validation": validation_record,
            }
            trainer.save(
                str(checkpoint_directory / "latest.pt"),
                training_metadata=validation_metadata,
                include_optimizers=True,
            )
            if should_checkpoint:
                trainer.save(
                    str(checkpoint_path),
                    training_metadata=validation_metadata,
                    include_optimizers=True,
                )
            score = float(residual_metrics["reward_mean"])
            if (
                validation_record["improvement"][
                    "passes_strict_base_nondegradation_guard"
                ]
                and score > best_score
            ):
                best_score = score
                best_update = int(update_index + 1)
                trainer.save(
                    str(checkpoint_directory / "best_validation.pt"),
                    training_metadata=validation_metadata,
                    include_optimizers=True,
                )
        _append_jsonl(training_history, last_metrics)
        progress.set_postfix(
            reward=f"{last_metrics['rollout_reward_mean']:.3f}",
            kl=f"{last_metrics['approx_kl_max']:.4f}",
            value_ev=f"{last_metrics['explained_variance']:.3f}",
            best=best_update,
        )
    progress.close()
    trainer.save(
        str(destination),
        training_metadata={
            **initialization,
            **last_metrics,
            "best_validation_update": int(best_update),
            "best_validation_reward_mean": float(best_score),
        },
        include_optimizers=True,
    )
    return {
        **last_metrics,
        "output": str(destination),
        "checkpoint_directory": str(checkpoint_directory),
        "training_history": str(training_history),
        "validation_history": str(validation_history),
        "base_validation": str(base_validation_path),
        "best_validation_checkpoint": str(
            checkpoint_directory / "best_validation.pt"
        ),
        "best_validation_update": int(best_update),
        "best_validation_reward_mean": float(best_score),
        "device": str(trainer.device),
    }


def evaluate_residual_checkpoints(
    checkpoint_paths,
    validation_environment_factory,
    episodes,
    device="auto",
):
    results = []
    base_metrics = None
    base_fingerprint = None
    for checkpoint_path in checkpoint_paths:
        policy = load_frozen_residual_ppo(checkpoint_path, device=device)
        if base_metrics is None:
            base_fingerprint = policy.base_fingerprint
            base_environment = validation_environment_factory()
            try:
                base_metrics = evaluate_residual_policy(
                    policy, base_environment, episodes, "base_only"
                )
            finally:
                base_environment.close()
        elif policy.base_fingerprint != base_fingerprint:
            raise ValueError(
                "All compared residual checkpoints must contain the identical "
                "frozen base Actor and base normalization statistics"
            )
        environment = validation_environment_factory()
        try:
            residual_metrics = evaluate_residual_policy(
                policy, environment, episodes, "base_plus_residual"
            )
        finally:
            environment.close()
        results.append(
            {
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "update": int(policy.training_metadata.get("update", 0)),
                "base": base_metrics,
                "base_plus_residual": residual_metrics,
                "improvement": validation_comparison(
                    base_metrics, residual_metrics
                ),
            }
        )
    ranked = sorted(
        results,
        key=lambda item: item["base_plus_residual"]["reward_mean"],
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["validation_rank"] = int(rank)
    best_by_reward = ranked[0]
    guarded = [
        item
        for item in ranked
        if item["improvement"]["passes_strict_base_nondegradation_guard"]
    ]
    best_guarded = guarded[0] if guarded else None
    return {
        "comparison_contract": (
            "paired_deterministic_base_only_vs_base_plus_residual_on_fixed_"
            "held_out_visual_trajectories_v1"
        ),
        "episodes": int(episodes),
        "base": base_metrics,
        "checkpoints": results,
        "best_reward_checkpoint": best_by_reward["checkpoint"],
        "best_reward_update": int(best_by_reward["update"]),
        "best_reward_mean": float(
            best_by_reward["base_plus_residual"]["reward_mean"]
        ),
        "best_nondegrading_checkpoint": (
            None if best_guarded is None else best_guarded["checkpoint"]
        ),
        "best_nondegrading_update": (
            None if best_guarded is None else int(best_guarded["update"])
        ),
    }
