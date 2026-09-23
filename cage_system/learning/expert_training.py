"""Random local Jacobian BC, C7-aligned DAgger and critic warm-up."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from cage_system.learning.ppo2 import (
    PPO_EXPERT_INITIALIZATION_CONTRACT,
    PPO2Trainer,
    load_trainable_ppo,
)
from cage_system.learning.train_ppo import collect_complete_rollout, train_ppo2


def _assert_c7_reset(reset_info, config):
    training = config["ppo_training"]
    expected_reset = int(training["privileged_reset_window_number"])
    expected_policy = int(training["first_policy_window_number"])
    if expected_reset != 3 or expected_policy != 4:
        raise ValueError(
            "Expert-guided training is fixed to third-window C7 IK reset and "
            "fourth valid five-frame policy start"
        )
    if int(reset_info.get("privileged_reset_window_number", -1)) != expected_reset:
        raise RuntimeError("Environment did not reset the robot from C7")
    if int(reset_info.get("first_policy_window_number", -1)) != expected_policy:
        raise RuntimeError("Environment did not begin control from the fourth window")


def collect_random_local_demonstrations(environment, config, minimum_steps):
    """Collect trajectory-free expert labels from diverse local pose errors."""
    settings = config["expert_training"]["random_demonstration"]
    rng = np.random.RandomState(int(config["system"]["seed"]) + 419)
    states, labels, episode_ids = [], [], []
    episodes = 0
    successful_episodes = 0
    collisions = 0
    expert_saturation = 0
    total_reset_attempts = 0
    horizon = int(settings["episode_horizon_steps"])
    progress = tqdm(
        total=int(minimum_steps),
        desc="Random local Jacobian demonstrations",
        unit="label",
        dynamic_ncols=True,
    )
    try:
        while len(states) < int(minimum_steps):
            raw_state, reset_info = environment.reset_random_local_expert(
                rng, settings
            )
            episodes += 1
            # Negative ids cannot collide with stored positive trajectory ids.
            episode_id = -episodes
            total_reset_attempts += int(reset_info["reset_attempts"])
            for _ in range(horizon):
                expert = environment.expert_action()
                if not np.all(np.isfinite(expert.normalized_action)):
                    raise RuntimeError("Jacobian expert generated a non-finite action")
                states.append(np.asarray(raw_state, dtype=np.float32).copy())
                labels.append(expert.normalized_action.copy())
                episode_ids.append(episode_id)
                expert_saturation += int(expert.saturated)
                raw_state, _, terminated, _, info = (
                    environment.step_fixed_expert_target(
                        expert.normalized_action, settings
                    )
                )
                progress.update(1)
                progress.set_postfix(
                    episodes=episodes,
                    success=successful_episodes,
                    refresh=False,
                )
                if bool(info["local_target_success"]):
                    successful_episodes += 1
                    break
                if terminated:
                    collisions += int(bool(info["contact_collision"]))
                    break
                if len(states) >= int(minimum_steps):
                    break
    finally:
        progress.close()
    arrays = (
        np.asarray(states, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(episode_ids, dtype=np.int64),
    )
    metrics = {
        "samples": int(len(states)),
        "episodes": int(episodes),
        "successful_episodes": int(successful_episodes),
        "contact_terminated_episodes": int(collisions),
        "mean_reset_attempts": float(total_reset_attempts / max(episodes, 1)),
        "expert_saturation_fraction": float(expert_saturation / max(len(states), 1)),
        "demonstration_source": "random_local_static_cage",
        "uses_trajectory": False,
        "uses_c7_reset": False,
        "uses_visual_observation": False,
        "uses_cage_network": False,
    }
    return arrays, metrics


def collect_dagger_samples(
    trainer,
    environment,
    config,
    minimum_steps,
    expert_weight,
    description,
):
    """Collect pure expert labels while executing a smooth expert/Actor mix."""
    weight = float(expert_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("DAgger expert weight must be inside [0,1]")
    states, labels, track_ids = [], [], []
    episodes = 0
    visual_escapes = 0
    recoveries = 0
    expert_saturation = 0
    actor_fraction = 1.0 - weight
    progress = tqdm(
        total=int(minimum_steps),
        desc=description,
        unit="label",
        dynamic_ncols=True,
    )
    try:
        while len(states) < int(minimum_steps):
            raw_state, reset_info = environment.reset()
            _assert_c7_reset(reset_info, config)
            trajectory_id = int(reset_info["trajectory_id"])
            episodes += 1
            done = False
            while not done and len(states) < int(minimum_steps):
                if not bool(getattr(environment, "action_ready", True)):
                    raw_state, _, terminated, truncated, _ = environment.step(
                        np.zeros(6, dtype=np.float32)
                    )
                    done = bool(terminated or truncated)
                    continue

                expert = environment.expert_action()
                if not np.all(np.isfinite(expert.normalized_action)):
                    raise RuntimeError("Jacobian expert generated a non-finite action")
                if weight < 1.0:
                    normalized = trainer.normalize_state(raw_state, update=False)
                    state_tensor = torch.as_tensor(
                        normalized, dtype=torch.float32, device=trainer.device
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        actor_action, _, _, _ = trainer.model.sample(
                            state_tensor, deterministic=True
                        )
                    actor_action = actor_action.cpu().numpy()[0]
                else:
                    actor_action = np.zeros(6, dtype=np.float32)
                executed = np.clip(
                    weight * expert.normalized_action
                    + actor_fraction * actor_action,
                    -1.0,
                    1.0,
                ).astype(np.float32)
                raw_next_state, _, terminated, truncated, info = environment.step(
                    executed
                )
                actual = np.asarray(
                    info.get("executed_normalized_action", executed), dtype=np.float32
                )
                if actual.shape != (6,) or not np.allclose(
                    actual, executed, atol=1e-7, rtol=0.0
                ):
                    raise RuntimeError(
                        "DAgger action was changed after expert/Actor mixing"
                    )
                # Supervision is always the pure current-state expert action;
                # the mixed action is only a safe data-collection policy.
                states.append(np.asarray(raw_state, dtype=np.float32).copy())
                labels.append(expert.normalized_action.copy())
                track_ids.append(trajectory_id)
                expert_saturation += int(expert.saturated)
                visual_escapes += int(bool(info.get("visual_escape", False)))
                recoveries += int(
                    bool(info.get("privileged_recovery_succeeded", False))
                )
                progress.update(1)
                progress.set_postfix(
                    expert=f"{weight:.2f}",
                    escapes=visual_escapes,
                    refresh=False,
                )
                raw_state = raw_next_state
                done = bool(terminated or truncated)
    finally:
        progress.close()
    arrays = (
        np.asarray(states, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(track_ids, dtype=np.int64),
    )
    metrics = {
        "samples": int(len(states)),
        "episodes": int(episodes),
        "expert_weight": weight,
        "actor_weight": actor_fraction,
        "collection_source": "c7_visual_trajectory",
        "uses_trajectory": True,
        "uses_c7_reset": True,
        "uses_visual_observation": True,
        "uses_cage_network": True,
        "visual_escapes": int(visual_escapes),
        "privileged_recoveries": int(recoveries),
        "expert_saturation_fraction": float(expert_saturation / max(len(states), 1)),
    }
    return arrays, metrics


def train_actor_supervised(
    trainer,
    raw_states,
    expert_actions,
    track_ids,
    config,
    epochs,
    description,
):
    expert_config = config["expert_training"]
    states = trainer.state_normalizer.normalize(raw_states)
    actions = np.asarray(expert_actions, dtype=np.float32)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    if states.shape[0] != actions.shape[0] or actions.shape[1:] != (6,):
        raise ValueError("Expert dataset must be [N,18] states and [N,6] actions")
    if track_ids.shape != (len(states),):
        raise ValueError("Expert dataset needs one trajectory id per state")
    rng = np.random.RandomState(int(config["system"]["seed"]) + 731)
    unique_tracks = np.unique(track_ids)
    if len(unique_tracks) >= 2:
        validation_track_count = max(
            1,
            int(round(len(unique_tracks) * float(expert_config["validation_fraction"]))),
        )
        validation_tracks = set(
            rng.permutation(unique_tracks)[:validation_track_count].tolist()
        )
        validation_mask = np.asarray(
            [int(item) in validation_tracks for item in track_ids], dtype=bool
        )
        validation_indices = np.flatnonzero(validation_mask)
        training_indices = np.flatnonzero(~validation_mask)
    else:
        # Small smoke tests may collect less than one complete track. Formal
        # production runs span many tracks and therefore take the branch above.
        permutation = rng.permutation(len(states))
        validation_count = max(
            1,
            int(round(len(states) * float(expert_config["validation_fraction"]))),
        )
        validation_indices = permutation[:validation_count]
        training_indices = permutation[validation_count:]
    if not len(training_indices):
        raise ValueError("Expert dataset is too small for train/validation split")

    actor = trainer.model.actor
    actor.log_std.requires_grad_(False)
    parameters = list(actor.features.parameters()) + list(actor.mean_head.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=float(expert_config["supervised_learning_rate"])
    )
    weights = torch.as_tensor(
        expert_config["joint_loss_weights"],
        dtype=torch.float32,
        device=trainer.device,
    )
    batch_size = int(expert_config["supervised_batch_size"])
    patience = int(expert_config["supervised_early_stopping_patience"])
    best_validation = float("inf")
    best_actor = None
    stale_epochs = 0
    last_metrics = {}
    progress = tqdm(
        range(int(epochs)),
        total=int(epochs),
        desc=description,
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        actor.train()
        epoch_losses = []
        shuffled = rng.permutation(training_indices)
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            state_tensor = torch.as_tensor(
                states[indices], dtype=torch.float32, device=trainer.device
            )
            label_tensor = torch.as_tensor(
                actions[indices], dtype=torch.float32, device=trainer.device
            )
            means, _ = actor(state_tensor)
            predicted = torch.tanh(means)
            per_joint = F.smooth_l1_loss(
                predicted, label_tensor, reduction="none"
            )
            loss = torch.mean(per_joint * weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config["ppo"]["max_grad_norm"])
            )
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        actor.eval()
        with torch.no_grad():
            validation_states = torch.as_tensor(
                states[validation_indices],
                dtype=torch.float32,
                device=trainer.device,
            )
            validation_labels = torch.as_tensor(
                actions[validation_indices],
                dtype=torch.float32,
                device=trainer.device,
            )
            validation_means, _ = actor(validation_states)
            validation_predictions = torch.tanh(validation_means)
            validation_loss = torch.mean(
                F.smooth_l1_loss(
                    validation_predictions,
                    validation_labels,
                    reduction="none",
                )
                * weights
            )
            mae = torch.mean(torch.abs(validation_predictions - validation_labels))
            sign_agreement = torch.mean(
                (
                    torch.sign(validation_predictions)
                    == torch.sign(validation_labels)
                ).float()
            )
        last_metrics = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": float(validation_loss.item()),
            "validation_action_mae": float(mae.item()),
            "validation_sign_agreement": float(sign_agreement.item()),
            "training_samples": int(len(training_indices)),
            "validation_samples": int(len(validation_indices)),
        }
        progress.set_postfix(
            val=f"{last_metrics['validation_loss']:.4f}",
            mae=f"{last_metrics['validation_action_mae']:.3f}",
            sign=f"{100.0 * last_metrics['validation_sign_agreement']:.1f}%",
        )
        if last_metrics["validation_loss"] < best_validation - float(
            expert_config["supervised_min_delta"]
        ):
            best_validation = last_metrics["validation_loss"]
            best_actor = {
                key: value.detach().cpu().clone()
                for key, value in actor.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    progress.close()
    if best_actor is None:
        raise RuntimeError("Actor supervised training did not produce a checkpoint")
    actor.load_state_dict(best_actor)
    actor.log_std.requires_grad_(True)
    with torch.no_grad():
        actor.log_std.fill_(float(expert_config["ppo_initial_log_std"]))
    last_metrics.update(
        {
            "best_validation_loss": float(best_validation),
            "epochs_completed": int(last_metrics.get("epoch", 0)),
        }
    )
    return last_metrics


def _discounted_returns(rewards, dones, gamma):
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    returns = np.zeros_like(rewards)
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        accumulator = rewards[index] + float(gamma) * (
            1.0 - dones[index]
        ) * accumulator
        returns[index] = accumulator
    return returns.astype(np.float32)


def warmup_critic(trainer, rollout, config):
    expert_config = config["expert_training"]
    returns = _discounted_returns(
        rollout.rewards, rollout.dones, config["ppo"]["gamma"]
    )
    states = torch.as_tensor(
        rollout.states, dtype=torch.float32, device=trainer.device
    )
    targets = torch.as_tensor(
        returns, dtype=torch.float32, device=trainer.device
    )
    critic = trainer.model.critic
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(expert_config["critic_warmup_learning_rate"])
    )
    rng = np.random.RandomState(int(config["system"]["seed"]) + 991)
    batch_size = int(expert_config["critic_warmup_batch_size"])
    epochs = int(expert_config["critic_warmup_epochs"])
    progress = tqdm(
        range(epochs),
        total=epochs,
        desc="Critic warm-up",
        unit="epoch",
        dynamic_ncols=True,
    )
    loss_value = float("inf")
    for _ in progress:
        losses = []
        shuffled = rng.permutation(len(states))
        for start in range(0, len(states), batch_size):
            indices = shuffled[start : start + batch_size]
            index_tensor = torch.as_tensor(indices, device=trainer.device)
            predicted = critic(states[index_tensor])
            loss = 0.5 * torch.mean(
                (predicted - targets[index_tensor]) ** 2
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), float(config["ppo"]["max_grad_norm"])
            )
            optimizer.step()
            losses.append(float(loss.item()))
        loss_value = float(np.mean(losses))
        progress.set_postfix(loss=f"{loss_value:.4f}")
    progress.close()
    with torch.no_grad():
        predictions = critic(states).cpu().numpy()
    variance = float(np.var(returns))
    explained = (
        0.0
        if variance < 1e-12
        else 1.0 - float(np.var(returns - predictions)) / variance
    )
    return {
        "critic_warmup_loss": loss_value,
        "critic_warmup_explained_variance": explained,
        "critic_warmup_samples": int(len(states)),
    }


def train_expert_guided_ppo(
    config,
    environment,
    output_path,
    device="auto",
    resume_checkpoint=None,
):
    """Run trajectory-free BC, then C7 trajectory-aligned DAgger and PPO."""
    expert_config = config["expert_training"]
    destination = Path(output_path)
    stage_directory = destination.parent / (destination.stem + "_expert_stages")

    if resume_checkpoint is not None:
        trainer, checkpoint_metadata, optimizer_restored = load_trainable_ppo(
            str(resume_checkpoint), config["ppo"], device=device
        )
        if trainer.initialization_contract != PPO_EXPERT_INITIALIZATION_CONTRACT:
            raise ValueError(
                "Expert-guided PPO can only resume its agreed expert initialization"
            )
        start_update = int(checkpoint_metadata.get("update", 0))
        if (
            start_update == 0
            and not optimizer_restored
            and "critic_warmup_samples" not in checkpoint_metadata
        ):
            raise ValueError(
                "The supplied expert checkpoint predates Critic warm-up and cannot "
                "be used as the formal PPO starting point"
            )
        initialization = {
            **checkpoint_metadata,
            "expert_guided": True,
            "resume_checkpoint": str(Path(resume_checkpoint)),
            "resume_optimizer_restored": bool(optimizer_restored),
        }
        result = train_ppo2(
            config,
            environment,
            output_path,
            device=device,
            trainer=trainer,
            initialization_metadata=initialization,
            rollout_reset_validator=lambda info: _assert_c7_reset(info, config),
            start_update=start_update,
            resume=True,
        )
        result["resume_checkpoint"] = str(Path(resume_checkpoint))
        result["resume_optimizer_restored"] = bool(optimizer_restored)
        return result

    trainer = PPO2Trainer(config["ppo"], device=device)
    trainer.initialization_contract = PPO_EXPERT_INITIALIZATION_CONTRACT
    stage_directory.mkdir(parents=True, exist_ok=True)
    history_path = stage_directory / "expert_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    (states, actions, track_ids), demonstration_metrics = (
        collect_random_local_demonstrations(
            environment,
            config,
            int(expert_config["demonstration_steps"]),
        )
    )
    trainer.state_normalizer.update(states)
    bc_metrics = train_actor_supervised(
        trainer,
        states,
        actions,
        track_ids,
        config,
        int(expert_config["behavior_cloning_epochs"]),
        "Actor behavior cloning",
    )
    trainer.save(
        str(stage_directory / "actor_bc.pt"),
        training_metadata={**demonstration_metrics, **bc_metrics},
    )
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": "behavior_cloning", **demonstration_metrics, **bc_metrics},
                sort_keys=True,
            )
            + "\n"
        )

    dagger_history = []
    weights = [float(item) for item in expert_config["dagger_expert_weights"]]
    for round_index, weight in enumerate(weights, start=1):
        (new_states, new_actions, new_track_ids), collection_metrics = collect_dagger_samples(
            trainer,
            environment,
            config,
            int(expert_config["dagger_steps_per_round"]),
            weight,
            "DAgger round {}/{} from C7".format(round_index, len(weights)),
        )
        trainer.state_normalizer.update(new_states)
        states = np.concatenate([states, new_states], axis=0)
        actions = np.concatenate([actions, new_actions], axis=0)
        track_ids = np.concatenate([track_ids, new_track_ids], axis=0)
        supervised_metrics = train_actor_supervised(
            trainer,
            states,
            actions,
            track_ids,
            config,
            int(expert_config["dagger_supervised_epochs"]),
            "DAgger actor update {}/{}".format(round_index, len(weights)),
        )
        round_metrics = {
            "round": int(round_index),
            "aggregate_samples": int(len(states)),
            **collection_metrics,
            **supervised_metrics,
        }
        dagger_history.append(round_metrics)
        trainer.save(
            str(stage_directory / "dagger_{:02d}.pt".format(round_index)),
            training_metadata=round_metrics,
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"stage": "dagger", **round_metrics}, sort_keys=True
                )
                + "\n"
            )

    np.savez_compressed(
        stage_directory / "aggregated_expert_dataset.npz",
        states=states,
        expert_actions=actions,
        track_ids=track_ids,
        sample_group_ids=track_ids,
        sample_sources=np.where(
            track_ids < 0, "random_local_static_cage", "c7_visual_trajectory"
        ),
        group_id_contract=np.asarray(
            "negative_random_episode_id_nonnegative_stored_trajectory_id_v1"
        ),
    )

    # The critic is deliberately created only after Actor supervision ends.
    trainer.reset_critic()
    warmup_rollout = collect_complete_rollout(
        trainer,
        environment,
        int(expert_config["critic_warmup_steps"]),
        reset_validator=lambda info: _assert_c7_reset(info, config),
    )
    critic_metrics = warmup_critic(trainer, warmup_rollout, config)
    critic_metrics.update(
        {
            "collection_source": "c7_visual_trajectory_actor_rollout",
            "uses_trajectory": True,
            "uses_c7_reset": True,
            "uses_visual_observation": True,
            "uses_cage_network": True,
        }
    )
    trainer.save(
        str(stage_directory / "critic_warmup.pt"),
        training_metadata=critic_metrics,
    )
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": "critic_warmup", **critic_metrics}, sort_keys=True
            )
            + "\n"
        )

    # PPO must start with fresh Adam moments. Expert actions are never inserted
    # into the on-policy rollout collected below.
    trainer.rebuild_optimizer()
    initialization = {
        "expert_guided": True,
        "initial_demonstrations": "random_local_static_cage",
        "initial_demonstrations_use_c7": False,
        "dagger_uses_c7": True,
        "c7_reset": True,
        "demonstration_samples": int(demonstration_metrics["samples"]),
        "aggregate_expert_samples": int(len(states)),
        "dagger_rounds": int(len(weights)),
        "dagger_final_expert_weight": float(weights[-1]),
        **critic_metrics,
    }
    result = train_ppo2(
        config,
        environment,
        output_path,
        device=device,
        trainer=trainer,
        initialization_metadata=initialization,
        rollout_reset_validator=lambda info: _assert_c7_reset(info, config),
    )
    result["expert_stage_directory"] = str(stage_directory)
    result["expert_history"] = str(history_path)
    return result
