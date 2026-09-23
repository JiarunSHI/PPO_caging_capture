"""Frozen DAgger base policy plus an independently optimized PPO residual."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from cage_system.learning.device import resolve_torch_device
from cage_system.learning.ppo2 import (
    ActorNetwork,
    CriticNetwork,
    PPO_ACTION_DISTRIBUTION_CONTRACT,
    PPO_CRITIC_TARGET_CONTRACT,
    PPO_ENTROPY_CONTRACT,
    PPO_REWARD_TARGET_CONTRACT,
    PPO_STATE_CONTRACT,
    RunningStateNormalizer,
    _validate_ppo_checkpoint,
    compute_gae,
    load_frozen_ppo,
)


RESIDUAL_PPO_CHECKPOINT_FORMAT_VERSION = 1
RESIDUAL_POLICY_CONTRACT = (
    "frozen_dagger_base_plus_tanh_gaussian_ppo_residual_v1"
)
RESIDUAL_ACTION_CONTRACT = (
    "clip(base_deterministic_action+scale*tanh_gaussian_residual,-1,1)"
)
RESIDUAL_OPTIMIZER_CONTRACT = "independent_residual_actor_and_critic_adam_v1"
RESIDUAL_NORMALIZATION_CONTRACT = (
    "frozen_base_and_residual_critic_checkpoint_statistics_v1"
)
RESIDUAL_REWARD_SPEC_KEYS = (
    "reward_mode",
    "reward_position_reference_m",
    "reward_orientation_reference_deg",
    "reward_position_quality_weight",
    "reward_orientation_quality_weight",
    "reward_progress_weight",
    "reward_success_bonus",
    "reward_joint_limit_weight",
    "reward_velocity_weight",
    "reward_action_weight",
    "visual_escape_penalty",
    "collision_penalty",
    "terminate_on_contact",
    "success_position_m",
    "success_orientation_deg",
)


def _reward_spec(config):
    result = {}
    for key in RESIDUAL_REWARD_SPEC_KEYS:
        value = config[key]
        result[key] = value if isinstance(value, (str, bool)) else float(value)
    return result


def _component_state(model_state, prefix: str):
    result = {
        name[len(prefix) :]: value
        for name, value in model_state.items()
        if name.startswith(prefix)
    }
    if not result:
        raise ValueError("Checkpoint does not contain component '{}'".format(prefix))
    return result


def _normalizer(config: Dict, state: Dict):
    normalizer = RunningStateNormalizer(
        int(config["state_dim"]),
        float(config["state_normalization_clip"]),
        float(config["state_normalization_epsilon"]),
    )
    normalizer.load_state_dict(state)
    return normalizer


def _file_sha256(path: str):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_base_fingerprint(actor_state, normalizer_state):
    """Identify the actual frozen base, not only its source file name."""
    digest = hashlib.sha256()
    for name in sorted(actor_state):
        tensor = actor_state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    for name in ("mean", "variance"):
        values = np.asarray(normalizer_state[name], dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(values.tobytes())
    for name in ("clip", "epsilon"):
        digest.update(name.encode("ascii"))
        digest.update(np.asarray([normalizer_state[name]], dtype="<f8").tobytes())
    return digest.hexdigest()


def _move_optimizer_state(optimizer, device):
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def _residual_scale(value):
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 0:
        values = np.repeat(values, 6)
    if values.shape != (6,) or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("Residual action scale must be one scalar or six values in (0,1]")
    return values


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


@dataclass
class ResidualDecision:
    base_action: np.ndarray
    residual_action: np.ndarray
    preclip_action: np.ndarray
    executed_action: np.ndarray
    residual_log_density: float
    value: float
    entropy: float


@dataclass
class ResidualRolloutBatch:
    states: np.ndarray
    residual_actions: np.ndarray
    base_actions: np.ndarray
    preclip_actions: np.ndarray
    executed_actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    old_log_densities: np.ndarray
    old_values: np.ndarray
    next_values: np.ndarray


class ResidualPPOTrainer:
    """Train only a small residual while preserving a validated DAgger policy."""

    def __init__(
        self,
        config: Dict,
        base_checkpoint_path: str,
        critic_checkpoint_path: str,
        device: str = "auto",
    ):
        self.full_config = config
        self.config = config["ppo"]
        self.residual_config = config["residual_training"]
        self.device = resolve_torch_device(device)

        base_checkpoint = torch.load(
            base_checkpoint_path, map_location="cpu", weights_only=False
        )
        critic_checkpoint = torch.load(
            critic_checkpoint_path, map_location="cpu", weights_only=False
        )
        _validate_ppo_checkpoint(base_checkpoint)
        _validate_ppo_checkpoint(critic_checkpoint)
        self._validate_source_checkpoints(base_checkpoint, critic_checkpoint)

        base_hidden = [int(item) for item in base_checkpoint["hidden_dims"]]
        critic_hidden = [int(item) for item in critic_checkpoint["hidden_dims"]]
        residual_hidden = [
            int(item)
            for item in self.residual_config.get(
                "hidden_dims", self.config["hidden_dims"]
            )
        ]
        self.base_actor = ActorNetwork(18, 6, base_hidden).to(self.device)
        self.base_actor.load_state_dict(
            _component_state(base_checkpoint["model_state"], "actor.")
        )
        self.base_actor.eval()
        for parameter in self.base_actor.parameters():
            parameter.requires_grad_(False)

        self.residual_actor = ActorNetwork(18, 6, residual_hidden).to(self.device)
        with torch.no_grad():
            nn.init.zeros_(self.residual_actor.mean_head.weight)
            nn.init.zeros_(self.residual_actor.mean_head.bias)
            self.residual_actor.log_std.fill_(
                float(self.residual_config["initial_log_std"])
            )

        self.critic = CriticNetwork(18, critic_hidden).to(self.device)
        self.critic.load_state_dict(
            _component_state(critic_checkpoint["model_state"], "critic.")
        )
        self.base_hidden_dims = base_hidden
        self.residual_hidden_dims = residual_hidden
        self.critic_hidden_dims = critic_hidden
        self.base_normalizer = _normalizer(
            self.config, base_checkpoint["state_normalizer"]
        )
        self.rl_normalizer = _normalizer(
            self.config, critic_checkpoint["state_normalizer"]
        )
        self.scale = _residual_scale(self.residual_config["action_scale"])

        self.actor_optimizer = torch.optim.Adam(
            self.residual_actor.parameters(),
            lr=float(self.residual_config["actor_learning_rate"]),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.residual_config["critic_learning_rate"]),
        )
        self.base_source = {
            "path": str(Path(base_checkpoint_path).resolve()),
            "sha256": _file_sha256(base_checkpoint_path),
        }
        self.critic_source = {
            "path": str(Path(critic_checkpoint_path).resolve()),
            "sha256": _file_sha256(critic_checkpoint_path),
        }

    @staticmethod
    def _validate_source_checkpoints(base_checkpoint, critic_checkpoint):
        for checkpoint in (base_checkpoint, critic_checkpoint):
            if (
                int(checkpoint.get("state_dim", -1)) != 18
                or int(checkpoint.get("action_dim", -1)) != 6
                or checkpoint.get("state_contract") != PPO_STATE_CONTRACT
                or checkpoint.get("reward_target_contract")
                != PPO_REWARD_TARGET_CONTRACT
            ):
                raise ValueError("Residual PPO requires compatible 18-D expert checkpoints")
        base_actor = _component_state(base_checkpoint["model_state"], "actor.")
        critic_actor = _component_state(critic_checkpoint["model_state"], "actor.")
        if set(base_actor) != set(critic_actor) or any(
            not torch.equal(base_actor[name], critic_actor[name])
            for name in base_actor
        ):
            raise ValueError(
                "Base Actor and Critic warm-up checkpoint must originate from the "
                "same final DAgger Actor"
            )

    def normalize_base(self, raw_state):
        return self.base_normalizer.normalize(raw_state)

    def normalize_rl(self, raw_state):
        return self.rl_normalizer.normalize(raw_state)

    def _single_state_tensors(self, raw_state):
        base = torch.as_tensor(
            self.normalize_base(raw_state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        rl = torch.as_tensor(
            self.normalize_rl(raw_state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return base, rl

    def decide(self, raw_state, deterministic: bool = False):
        base_state, rl_state = self._single_state_tensors(raw_state)
        with torch.no_grad():
            base_action, _, _ = self.base_actor.sample(
                base_state, deterministic=True
            )
            residual_action, log_density, entropy = self.residual_actor.sample(
                rl_state, deterministic=deterministic
            )
            value = self.critic(rl_state)
        base_numpy = base_action.cpu().numpy()[0]
        residual_numpy = residual_action.cpu().numpy()[0]
        preclip = base_numpy + self.scale * residual_numpy
        executed = np.clip(preclip, -1.0, 1.0).astype(np.float32)
        return ResidualDecision(
            base_numpy.astype(np.float32),
            residual_numpy.astype(np.float32),
            preclip.astype(np.float32),
            executed,
            float(log_density.item()),
            float(value.item()),
            float(entropy.item()),
        )

    def value(self, raw_state):
        state = torch.as_tensor(
            self.normalize_rl(raw_state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return float(self.critic(state).item())

    def update(self, rollout: ResidualRolloutBatch):
        raw_advantages = compute_gae(
            rollout.rewards,
            rollout.old_values,
            rollout.next_values,
            rollout.dones,
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        critic_targets = raw_advantages + np.asarray(
            rollout.old_values, dtype=np.float64
        )
        advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std() + 1e-8
        )
        tensors = {
            "states": torch.as_tensor(
                rollout.states, dtype=torch.float32, device=self.device
            ),
            "residual_actions": torch.as_tensor(
                rollout.residual_actions, dtype=torch.float32, device=self.device
            ),
            "old_log": torch.as_tensor(
                rollout.old_log_densities, dtype=torch.float32, device=self.device
            ),
            "advantages": torch.as_tensor(
                advantages, dtype=torch.float32, device=self.device
            ),
            "returns": torch.as_tensor(
                critic_targets, dtype=torch.float32, device=self.device
            ),
        }
        actor_metrics = self._update_actor(tensors)
        critic_metrics = self._update_critic(tensors)
        executed_saturation = float(
            np.mean(
                np.abs(rollout.executed_actions)
                >= float(self.config["action_saturation_threshold"])
            )
        )
        composition_clip = float(
            np.mean(np.abs(rollout.preclip_actions) > 1.0)
        )
        residual_saturation = float(
            np.mean(
                np.abs(rollout.residual_actions)
                >= float(self.config["action_saturation_threshold"])
            )
        )
        return {
            **actor_metrics,
            **critic_metrics,
            "residual_action_saturation_fraction": residual_saturation,
            "executed_action_saturation_fraction": executed_saturation,
            "composition_clip_fraction": composition_clip,
            "base_action_mean_abs": float(np.mean(np.abs(rollout.base_actions))),
            "residual_action_mean_abs": float(
                np.mean(np.abs(rollout.residual_actions))
            ),
            "scaled_residual_mean_abs": float(
                np.mean(np.abs(rollout.residual_actions * self.scale[None, :]))
            ),
        }

    def _update_actor(self, tensors):
        sample_count = len(tensors["states"])
        minibatch = int(self.config["minibatch_size"])
        clip_epsilon = float(self.config["clip_epsilon"])
        kl_stop = float(self.config["target_kl"]) * float(
            self.config["kl_stop_multiplier"]
        )
        history = {
            "residual_actor_loss": [],
            "residual_entropy": [],
            "density_ratio_mean": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        early_stopped = False
        completed_epochs = 0
        self.residual_actor.train()
        for epoch in range(int(self.residual_config["actor_update_epochs"])):
            permutation = np.random.permutation(sample_count)
            for start in range(0, sample_count, minibatch):
                indices = torch.as_tensor(
                    permutation[start : start + minibatch], device=self.device
                )
                log_density, entropy = self.residual_actor.evaluate_actions(
                    tensors["states"][indices],
                    tensors["residual_actions"][indices],
                )
                log_ratio = log_density - tensors["old_log"][indices]
                ratio = torch.exp(log_ratio)
                approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > clip_epsilon).float()
                )
                advantages = tensors["advantages"][indices]
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                ) * advantages
                actor_loss = -torch.mean(torch.minimum(unclipped, clipped))
                entropy_mean = torch.mean(entropy)
                for key, value in (
                    ("residual_actor_loss", actor_loss),
                    ("residual_entropy", entropy_mean),
                    ("density_ratio_mean", ratio.mean()),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                ):
                    history[key].append(float(value.item()))
                if float(approx_kl.item()) > kl_stop:
                    early_stopped = True
                    break
                loss = actor_loss - float(
                    self.config["entropy_coefficient"]
                ) * entropy_mean
                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.residual_actor.parameters(),
                    float(self.config["max_grad_norm"]),
                )
                self.actor_optimizer.step()
            if early_stopped:
                break
            completed_epochs = epoch + 1
        metrics = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in history.items()
        }
        metrics.update({
            "approx_kl_max": float(max(history["approx_kl"], default=0.0)),
            "target_kl": float(self.config["target_kl"]),
            "actor_kl_early_stopped": bool(early_stopped),
            "actor_epochs_completed": int(completed_epochs),
        })
        return metrics

    def _update_critic(self, tensors):
        sample_count = len(tensors["states"])
        minibatch = int(self.config["minibatch_size"])
        losses = []
        self.critic.train()
        for _ in range(int(self.residual_config["critic_update_epochs"])):
            permutation = np.random.permutation(sample_count)
            for start in range(0, sample_count, minibatch):
                indices = torch.as_tensor(
                    permutation[start : start + minibatch], device=self.device
                )
                values = self.critic(tensors["states"][indices])
                loss = 0.5 * torch.mean(
                    (values - tensors["returns"][indices]) ** 2
                )
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(self.config["max_grad_norm"])
                )
                self.critic_optimizer.step()
                losses.append(float(loss.item()))
        self.critic.eval()
        with torch.no_grad():
            predictions = self.critic(tensors["states"]).cpu().numpy()
        targets = tensors["returns"].detach().cpu().numpy()
        variance = float(np.var(targets))
        explained = (
            0.0
            if variance < 1e-12
            else 1.0 - float(np.var(targets - predictions)) / variance
        )
        return {
            "value_loss": float(np.mean(losses)) if losses else 0.0,
            "critic_epochs_completed": int(
                self.residual_config["critic_update_epochs"]
            ),
            "explained_variance": float(explained),
        }

    def align_critic(self, rollout: ResidualRolloutBatch, progress_bar=None):
        targets_numpy = _discounted_returns(
            rollout.rewards, rollout.dones, float(self.config["gamma"])
        )
        states = torch.as_tensor(
            rollout.states, dtype=torch.float32, device=self.device
        )
        targets = torch.as_tensor(
            targets_numpy, dtype=torch.float32, device=self.device
        )
        batch_size = int(self.residual_config["critic_alignment_batch_size"])
        epochs = int(self.residual_config["critic_alignment_epochs"])
        rng = np.random.RandomState(
            int(self.full_config["system"]["seed"]) + 1709
        )
        losses = []
        self.critic.train()
        for epoch in range(epochs):
            permutation = rng.permutation(len(states))
            epoch_losses = []
            for start in range(0, len(states), batch_size):
                indices = permutation[start : start + batch_size]
                index_tensor = torch.as_tensor(indices, device=self.device)
                predicted = self.critic(states[index_tensor])
                loss = 0.5 * torch.mean(
                    (predicted - targets[index_tensor]) ** 2
                )
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(self.config["max_grad_norm"])
                )
                self.critic_optimizer.step()
                losses.append(float(loss.item()))
                epoch_losses.append(float(loss.item()))
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss="{:.4f}".format(float(np.mean(epoch_losses))),
                    epoch=epoch + 1,
                    refresh=False,
                )
        self.critic.eval()
        with torch.no_grad():
            predictions = self.critic(states).cpu().numpy()
        variance = float(np.var(targets_numpy))
        explained = (
            0.0
            if variance < 1e-12
            else 1.0 - float(np.var(targets_numpy - predictions)) / variance
        )
        return {
            "critic_alignment_samples": int(len(states)),
            "critic_alignment_epochs": int(epochs),
            "critic_alignment_loss": float(np.mean(losses)) if losses else 0.0,
            "critic_alignment_explained_variance": float(explained),
        }

    def save(
        self,
        path: str,
        training_metadata: Optional[Dict] = None,
        include_optimizers: bool = False,
    ):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": RESIDUAL_PPO_CHECKPOINT_FORMAT_VERSION,
            "policy_type": "residual_ppo",
            "policy_contract": RESIDUAL_POLICY_CONTRACT,
            "action_contract": RESIDUAL_ACTION_CONTRACT,
            "optimizer_contract": RESIDUAL_OPTIMIZER_CONTRACT,
            "normalization_contract": RESIDUAL_NORMALIZATION_CONTRACT,
            "state_dim": 18,
            "action_dim": 6,
            "state_contract": PPO_STATE_CONTRACT,
            "action_distribution": PPO_ACTION_DISTRIBUTION_CONTRACT,
            "entropy_estimator": PPO_ENTROPY_CONTRACT,
            "critic_target": PPO_CRITIC_TARGET_CONTRACT,
            "reward_target_contract": PPO_REWARD_TARGET_CONTRACT,
            "reward_mode": self.config.get("reward_mode", "legacy_squared"),
            "reward_spec": _reward_spec(self.config),
            "base_hidden_dims": self.base_hidden_dims,
            "residual_hidden_dims": self.residual_hidden_dims,
            "critic_hidden_dims": self.critic_hidden_dims,
            "base_actor_state": {
                name: value.detach().cpu()
                for name, value in self.base_actor.state_dict().items()
            },
            "residual_actor_state": {
                name: value.detach().cpu()
                for name, value in self.residual_actor.state_dict().items()
            },
            "critic_state": {
                name: value.detach().cpu()
                for name, value in self.critic.state_dict().items()
            },
            "base_normalizer": self.base_normalizer.state_dict(),
            "rl_normalizer": self.rl_normalizer.state_dict(),
            "residual_scale": self.scale.copy(),
            "base_source": dict(self.base_source),
            "critic_source": dict(self.critic_source),
            "training_metadata": dict(training_metadata or {}),
        }
        if include_optimizers:
            checkpoint["actor_optimizer_state"] = self.actor_optimizer.state_dict()
            checkpoint["critic_optimizer_state"] = self.critic_optimizer.state_dict()
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(destination)

    def restore(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        _validate_residual_checkpoint(checkpoint)
        if (
            checkpoint["base_source"]["sha256"] != self.base_source["sha256"]
            or checkpoint["critic_source"]["sha256"]
            != self.critic_source["sha256"]
            or not np.array_equal(
                np.asarray(checkpoint["residual_scale"], dtype=np.float32),
                self.scale,
            )
            or checkpoint.get("reward_spec") != _reward_spec(self.config)
        ):
            raise ValueError(
                "Resume checkpoint source policies, residual scale or reward "
                "specification do not match"
            )
        self.base_actor.load_state_dict(checkpoint["base_actor_state"])
        self.residual_actor.load_state_dict(checkpoint["residual_actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        self.base_normalizer.load_state_dict(checkpoint["base_normalizer"])
        self.rl_normalizer.load_state_dict(checkpoint["rl_normalizer"])
        optimizer_restored = False
        if (
            "actor_optimizer_state" in checkpoint
            and "critic_optimizer_state" in checkpoint
        ):
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state"]
            )
            _move_optimizer_state(self.actor_optimizer, self.device)
            _move_optimizer_state(self.critic_optimizer, self.device)
            optimizer_restored = True
        return dict(checkpoint.get("training_metadata", {})), optimizer_restored


def _validate_residual_checkpoint(checkpoint):
    if (
        int(checkpoint.get("format_version", -1))
        != RESIDUAL_PPO_CHECKPOINT_FORMAT_VERSION
        or checkpoint.get("policy_type") != "residual_ppo"
        or checkpoint.get("policy_contract") != RESIDUAL_POLICY_CONTRACT
        or checkpoint.get("action_contract") != RESIDUAL_ACTION_CONTRACT
        or checkpoint.get("optimizer_contract") != RESIDUAL_OPTIMIZER_CONTRACT
        or checkpoint.get("normalization_contract")
        != RESIDUAL_NORMALIZATION_CONTRACT
        or checkpoint.get("state_contract") != PPO_STATE_CONTRACT
        or checkpoint.get("reward_target_contract")
        != PPO_REWARD_TARGET_CONTRACT
        or checkpoint.get("action_distribution")
        != PPO_ACTION_DISTRIBUTION_CONTRACT
        or checkpoint.get("entropy_estimator") != PPO_ENTROPY_CONTRACT
        or checkpoint.get("critic_target") != PPO_CRITIC_TARGET_CONTRACT
    ):
        raise ValueError("Unsupported residual PPO checkpoint")
    for key in (
        "base_actor_state",
        "residual_actor_state",
        "critic_state",
        "base_normalizer",
        "rl_normalizer",
        "residual_scale",
        "base_source",
        "critic_source",
    ):
        if key not in checkpoint:
            raise ValueError("Residual PPO checkpoint is missing '{}'".format(key))


class FrozenResidualPolicy:
    def __init__(self, checkpoint, device):
        _validate_residual_checkpoint(checkpoint)
        self.device = device
        self.base_actor = ActorNetwork(
            18, 6, checkpoint["base_hidden_dims"]
        ).to(device)
        self.residual_actor = ActorNetwork(
            18, 6, checkpoint["residual_hidden_dims"]
        ).to(device)
        self.critic = CriticNetwork(18, checkpoint["critic_hidden_dims"]).to(device)
        self.base_actor.load_state_dict(checkpoint["base_actor_state"])
        self.residual_actor.load_state_dict(checkpoint["residual_actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        for model in (self.base_actor, self.residual_actor, self.critic):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.base_normalizer = checkpoint["base_normalizer"]
        self.rl_normalizer = checkpoint["rl_normalizer"]
        self.scale = torch.as_tensor(
            checkpoint["residual_scale"], dtype=torch.float32, device=device
        )
        self.base_source = dict(checkpoint["base_source"])
        self.base_fingerprint = _frozen_base_fingerprint(
            checkpoint["base_actor_state"], checkpoint["base_normalizer"]
        )
        self.training_metadata = dict(checkpoint.get("training_metadata", {}))

    @staticmethod
    def _normalize(states, normalizer):
        mean = torch.as_tensor(
            normalizer["mean"], dtype=states.dtype, device=states.device
        )
        variance = torch.as_tensor(
            normalizer["variance"], dtype=states.dtype, device=states.device
        )
        return torch.clamp(
            (states - mean)
            / torch.sqrt(variance + float(normalizer["epsilon"])),
            -float(normalizer["clip"]),
            float(normalizer["clip"]),
        )

    def decompose(self, raw_states, deterministic=True):
        raw_states = raw_states.to(self.device, dtype=torch.float32)
        base_states = self._normalize(raw_states, self.base_normalizer)
        rl_states = self._normalize(raw_states, self.rl_normalizer)
        base, _, _ = self.base_actor.sample(base_states, deterministic=True)
        residual, log_density, entropy = self.residual_actor.sample(
            rl_states, deterministic=deterministic
        )
        preclip = base + self.scale * residual
        executed = torch.clamp(preclip, -1.0, 1.0)
        value = self.critic(rl_states)
        return executed, log_density, value, entropy, base, residual, preclip

    def sample(self, raw_states, deterministic=True):
        executed, log_density, value, entropy, _, _, _ = self.decompose(
            raw_states, deterministic=deterministic
        )
        return executed, log_density, value, entropy


def load_frozen_residual_ppo(path: str, device: str = "auto"):
    resolved_device = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return FrozenResidualPolicy(checkpoint, resolved_device)


def load_frozen_control_policy(path: str, device: str = "auto"):
    """Load either the legacy direct PPO format or the residual PPO format."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type") == "residual_ppo":
        return FrozenResidualPolicy(checkpoint, resolve_torch_device(device))
    return load_frozen_ppo(path, device=device)
