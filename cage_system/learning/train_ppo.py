import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from cage_system.learning.ppo2 import (
    PPO_REWARD_TARGET_CONTRACT,
    PPO2Trainer,
    RolloutBatch,
)


def _prepare_training_history(history_path: Path, start_update: int, resume: bool):
    """Keep history aligned with the exact model update in a resume checkpoint."""
    if not history_path.exists():
        return None
    if not resume:
        history_path.unlink()
        return None
    lines = history_path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if "update" not in record:
            raise ValueError("PPO training history contains a row without update")
        records.append((int(record["update"]), line))
    if not records or max(update for update, _ in records) <= int(start_update):
        return None

    # Preserve records produced after an older checkpoint, but do not append to
    # them as though their unavailable model updates had been restored.
    suffix = 1
    while True:
        archive = history_path.with_name(
            "{}.pre_resume_from_{:04d}.{:02d}.jsonl".format(
                history_path.stem, int(start_update), suffix
            )
        )
        if not archive.exists():
            break
        suffix += 1
    history_path.replace(archive)
    retained = [
        line for update, line in records if update <= int(start_update)
    ]
    history_path.write_text(
        "" if not retained else "\n".join(retained) + "\n",
        encoding="utf-8",
    )
    return archive


def collect_complete_rollout(
    trainer: PPO2Trainer,
    environment,
    minimum_steps: int,
    progress_bar=None,
    reset_validator=None,
):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    old_logs, old_values, next_values = [], [], []
    while len(states) < minimum_steps:
        raw_state, reset_info = environment.reset()
        if reset_validator is not None:
            reset_validator(reset_info)
        policy_state = (
            trainer.normalize_state(raw_state, update=True)
            if bool(getattr(environment, "action_ready", True))
            else None
        )
        done = False
        while not done:
            policy_decision = bool(getattr(environment, "action_ready", True))
            if policy_decision:
                if policy_state is None:
                    policy_state = trainer.normalize_state(raw_state, update=True)
                state_tensor = torch.as_tensor(
                    policy_state, dtype=torch.float32, device=trainer.device
                ).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, log_density, value, _ = trainer.model.sample(state_tensor)
                action = action_tensor.cpu().numpy()[0]
            else:
                # Advance the visual/dynamics clock without invoking PPO.  The
                # environment interprets zero as a hold-at-current-joints command.
                action = np.zeros(6, dtype=np.float32)
            raw_next_state, reward, terminated, truncated, transition_info = (
                environment.step(action)
            )
            done = bool(terminated or truncated)
            next_ready = bool(getattr(environment, "action_ready", True))
            policy_next_state = (
                trainer.normalize_state(raw_next_state, update=True)
                if next_ready
                else None
            )
            if policy_decision:
                executed_action = np.asarray(
                    transition_info.get("executed_normalized_action", action),
                    dtype=np.float32,
                )
                if executed_action.shape != (6,) or not np.allclose(
                    executed_action, action, atol=1e-7, rtol=0.0
                ):
                    raise RuntimeError(
                        "A sampled PPO action was suppressed or changed after policy "
                        "selection; it must not be stored as an on-policy transition"
                    )
                reward_pose = np.asarray(
                    transition_info.get("reward_cage_pose6", []), dtype=float
                )
                next_pose = np.asarray(
                    transition_info.get("next_cage_pose6", []), dtype=float
                )
                action_relative = np.asarray(
                    transition_info.get("action_state_relative_error6", []),
                    dtype=float,
                )
                next_relative = np.asarray(
                    transition_info.get("next_state_relative_error6", []),
                    dtype=float,
                )
                if (
                    transition_info.get("reward_target_contract")
                    != PPO_REWARD_TARGET_CONTRACT
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
                        "PPO transition violated the same-Cage reward contract: "
                        "a_t and r_t must use C_t while state[:6] is the "
                        "corresponding TCP-to-Cage relative error"
                    )
                training_boundary = bool(
                    done
                    or not next_ready
                    or transition_info.get("rollout_boundary", False)
                )
                if training_boundary:
                    next_value_scalar = 0.0
                else:
                    next_tensor = torch.as_tensor(
                        policy_next_state,
                        dtype=torch.float32,
                        device=trainer.device,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        _, _, next_value = trainer.model(next_tensor)
                    next_value_scalar = float(next_value.item())
                states.append(policy_state)
                actions.append(executed_action)
                rewards.append(reward)
                next_states.append(
                    policy_next_state
                    if policy_next_state is not None
                    else np.zeros_like(policy_state)
                )
                dones.append(float(training_boundary))
                old_logs.append(float(log_density.item()))
                old_values.append(float(value.item()))
                next_values.append(next_value_scalar)
                if progress_bar is not None and progress_bar.n < minimum_steps:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        accepted=len(states),
                        reward=f"{float(reward):.3f}",
                        refresh=False,
                    )
            raw_state = raw_next_state
            policy_state = policy_next_state
    return RolloutBatch(
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
        np.asarray(dones, dtype=np.float32),
        np.asarray(old_logs, dtype=np.float32),
        np.asarray(old_values, dtype=np.float32),
        np.asarray(next_values, dtype=np.float32),
    )


def train_ppo2(
    config,
    environment,
    output_path: str,
    device: str = "auto",
    trainer=None,
    initialization_metadata=None,
    rollout_reset_validator=None,
    start_update: int = 0,
    resume: bool = False,
):
    trainer = trainer or PPO2Trainer(config["ppo"], device=device)
    last_metrics = {}
    update_count = int(config["ppo"]["updates"])
    start_update = int(start_update)
    if start_update < 0 or start_update > update_count:
        raise ValueError(
            "Resume update {} is outside configured total [0, {}]".format(
                start_update, update_count
            )
        )
    progress_interval = max(1, update_count // 100)
    rollout_steps = int(config["ppo"]["rollout_steps"])
    checkpoint_interval = int(
        config.get("ppo_training", {}).get("checkpoint_interval_updates", 50)
    )
    destination = Path(output_path)
    checkpoint_directory = destination.parent / (
        destination.stem + "_checkpoints"
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_directory / "training_history.jsonl"
    archived_history = _prepare_training_history(
        history_path, start_update, bool(resume)
    )
    update_progress = tqdm(
        range(start_update, update_count),
        total=update_count,
        initial=start_update,
        desc="PPO updates",
        unit="update",
        dynamic_ncols=True,
    )
    for update_index in update_progress:
        rollout_progress = tqdm(
            total=rollout_steps,
            desc="rollout {:04d}/{:04d}".format(update_index + 1, update_count),
            unit="transition",
            leave=False,
            dynamic_ncols=True,
        )
        try:
            rollout = collect_complete_rollout(
                trainer,
                environment,
                rollout_steps,
                progress_bar=rollout_progress,
                reset_validator=rollout_reset_validator,
            )
        finally:
            rollout_progress.close()
        last_metrics = trainer.update(rollout)
        last_metrics["update"] = update_index + 1
        last_metrics["rollout_samples"] = int(len(rollout.states))
        last_metrics["rollout_reward_mean"] = float(np.mean(rollout.rewards))
        last_metrics["rollout_reward_sum"] = float(np.sum(rollout.rewards))
        checkpoint_metadata = {
            **dict(initialization_metadata or {}),
            **last_metrics,
        }
        # The rolling checkpoint is written after every completed optimizer
        # update. An interrupted rollout is intentionally discarded on resume.
        trainer.save(
            str(checkpoint_directory / "latest.pt"),
            training_metadata=checkpoint_metadata,
            include_optimizer=True,
        )
        if (
            (update_index + 1) % checkpoint_interval == 0
            or update_index + 1 == update_count
        ):
            trainer.save(
                str(
                    checkpoint_directory
                    / "update_{:04d}.pt".format(update_index + 1)
                ),
                training_metadata=checkpoint_metadata,
                include_optimizer=True,
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(last_metrics, sort_keys=True) + "\n")
        update_progress.set_postfix(
            reward=f"{last_metrics['rollout_reward_mean']:.3f}",
            kl=f"{last_metrics['approx_kl_max']:.4f}",
            policy_epochs=last_metrics["epochs_completed"],
            value_ev=f"{last_metrics['explained_variance']:.3f}",
        )
        if (
            update_index == 0
            or update_index + 1 == update_count
            or (update_index + 1) % progress_interval == 0
        ):
            tqdm.write("[ppo] " + json.dumps(last_metrics, sort_keys=True))
    update_progress.close()
    trainer.save(
        output_path,
        training_metadata={
            **dict(initialization_metadata or {}),
            **last_metrics,
        },
        include_optimizer=True,
    )
    last_metrics["device"] = str(trainer.device)
    last_metrics["checkpoint_directory"] = str(checkpoint_directory)
    last_metrics["training_history"] = str(history_path)
    last_metrics["resumed_from_update"] = int(start_update)
    if archived_history is not None:
        last_metrics["archived_unrecoverable_history"] = str(archived_history)
    return last_metrics
