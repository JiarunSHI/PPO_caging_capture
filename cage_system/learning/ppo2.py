from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from scipy.spatial.transform import Rotation

from cage_system.geometry import rotation_as_matrix, rotation_from_matrix
from cage_system.learning.dataset import CAGE_GEOMETRY_CONTRACT
from cage_system.learning.history_quality import HISTORY_QUALITY_CONTRACT

from cage_system.learning.device import resolve_torch_device


PPO_CHECKPOINT_FORMAT_VERSION = 11
PPO_ACTION_DISTRIBUTION_CONTRACT = (
    "tanh_transformed_diagonal_gaussian_joint_density"
)
PPO_REWARD_TARGET_CONTRACT = "same_cage_pose_as_action_state"
PPO_STATE_NORMALIZATION_CONTRACT = "running_mean_variance_frozen_at_inference"
PPO_ENTROPY_CONTRACT = "monte_carlo_tanh_transformed_policy_entropy"
PPO_CRITIC_TARGET_CONTRACT = "gae_lambda_return"
PPO_HISTORY_SUPPRESSION_CONTRACT = (
    "actual_executed_action_only_privileged_recovery_is_gae_boundary_v2"
)
PPO_TRAINING_RESET_CONTRACT = (
    "known_track_third_window_ik_fourth_visual_window_policy_start_v1"
)
PPO_STATE_CONTRACT = "base_tcp_to_cage_relative_xyz_rotvec_q_qdot_v1"
PPO_NETWORK_CONTRACT = "independent_actor_and_critic_v1"
PPO_EXPERT_INITIALIZATION_CONTRACT = (
    "random_local_jacobian_bc_c7_track_mixed_dagger_critic_warmup_v2"
)
PPO_ALLOWED_INITIALIZATION_CONTRACTS = {
    "random_actor_critic_v1",
    PPO_EXPERT_INITIALIZATION_CONTRACT,
}


class RunningStateNormalizer:
    """Numerically stable running mean/variance for the fixed 18-D state."""

    def __init__(self, dimension: int, clip: float, epsilon: float):
        self.dimension = int(dimension)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.variance = np.ones(self.dimension, dtype=np.float64)
        self.count = 1e-4

    def update(self, observations):
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                "PPO state normalizer expected [N,{}]".format(self.dimension)
            )
        batch_count = len(values)
        if batch_count == 0:
            return
        batch_mean = np.mean(values, axis=0)
        batch_variance = np.var(values, axis=0)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        old_m2 = self.variance * self.count
        batch_m2 = batch_variance * batch_count
        correction = delta ** 2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.variance = np.maximum(
            (old_m2 + batch_m2 + correction) / total_count, 0.0
        )
        self.count = float(total_count)

    def normalize(self, observations):
        values = np.asarray(observations, dtype=np.float64)
        normalized = (values - self.mean) / np.sqrt(
            self.variance + self.epsilon
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
            "count": float(self.count),
            "clip": float(self.clip),
            "epsilon": float(self.epsilon),
        }

    def load_state_dict(self, state):
        required = {"mean", "variance", "count", "clip", "epsilon"}
        if not required.issubset(state):
            missing = sorted(required.difference(state))
            raise ValueError(
                "PPO state normalizer checkpoint is incomplete: {}".format(
                    ", ".join(missing)
                )
            )
        mean = np.asarray(state["mean"], dtype=np.float64)
        variance = np.asarray(state["variance"], dtype=np.float64)
        if mean.shape != (self.dimension,) or variance.shape != (self.dimension,):
            raise ValueError(
                "PPO state normalizer must contain {}-D statistics".format(
                    self.dimension
                )
            )
        count = float(state["count"])
        if count <= 0.0 or not np.all(np.isfinite(mean)) or not np.all(
            np.isfinite(variance)
        ):
            raise ValueError("PPO state normalizer contains invalid statistics")
        self.mean = mean.copy()
        self.variance = np.maximum(variance, 0.0).copy()
        self.count = count
        self.clip = float(state["clip"])
        self.epsilon = float(state["epsilon"])


def _mlp(input_dim: int, hidden_dims: Sequence[int]):
    layers = []
    current = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, int(hidden)), nn.Tanh()])
        current = int(hidden)
    return nn.Sequential(*layers), current


class ActorNetwork(nn.Module):
    """Tanh-squashed diagonal Gaussian actor used by BC, DAgger and PPO."""

    def __init__(self, state_dim: int = 18, action_dim: int = 6, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        if state_dim != 18 or action_dim != 6:
            raise ValueError("The agreed PPO2 contract is fixed at state=18 and action=6")
        self.features, feature_dim = _mlp(state_dim, hidden_dims)
        self.mean_head = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def forward(self, states):
        features = self.features(states)
        means = self.mean_head(features)
        log_std = torch.clamp(self.log_std, -5.0, 1.0).expand_as(means)
        return means, log_std

    @staticmethod
    def _squashed_log_density(distribution, raw_actions):
        # Joint conditional log-density: sum all six diagonal-Gaussian terms,
        # then include the tanh change-of-variables correction.  The softplus
        # identity is stable even when tanh is close to +/-1.
        gaussian_joint = distribution.log_prob(raw_actions).sum(dim=-1)
        correction = (
            2.0
            * (np.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions))
        ).sum(dim=-1)
        return gaussian_joint - correction

    @classmethod
    def _transformed_entropy_sample(cls, distribution):
        # Differential entropy of tanh(Normal) has no simple closed form.
        # A fresh reparameterized action gives an unbiased Monte Carlo sample
        # of -log pi(a|s), with gradients through both Normal and tanh.
        raw_actions = distribution.rsample()
        return -cls._squashed_log_density(distribution, raw_actions)

    def sample(self, states, deterministic=False):
        means, log_std = self(states)
        distribution = Normal(means, torch.exp(log_std))
        raw_actions = means if deterministic else distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_density = self._squashed_log_density(distribution, raw_actions)
        entropy = (
            -log_density
            if not deterministic
            else torch.zeros_like(log_density)
        )
        return actions, log_density, entropy

    def evaluate_actions(self, states, actions):
        means, log_std = self(states)
        distribution = Normal(means, torch.exp(log_std))
        bounded = torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6)
        raw_actions = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
        log_density = self._squashed_log_density(distribution, raw_actions)
        entropy = self._transformed_entropy_sample(distribution)
        return log_density, entropy


class CriticNetwork(nn.Module):
    """Independent value network so BC/DAgger cannot alter critic features."""

    def __init__(self, state_dim: int = 18, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        self.features, feature_dim = _mlp(state_dim, hidden_dims)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, states):
        return self.value_head(self.features(states)).squeeze(-1)


class SharedActorCritic(nn.Module):
    """Compatibility name for an actor/critic with deliberately separate trunks."""

    def __init__(self, state_dim: int = 18, action_dim: int = 6, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        if state_dim != 18 or action_dim != 6:
            raise ValueError("The agreed PPO2 contract is fixed at state=18 and action=6")
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic = CriticNetwork(state_dim, hidden_dims)

    @property
    def mean_head(self):
        return self.actor.mean_head

    @property
    def log_std(self):
        return self.actor.log_std

    def forward(self, states):
        means, log_std = self.actor(states)
        return means, log_std, self.critic(states)

    def sample(self, states, deterministic=False):
        actions, log_density, entropy = self.actor.sample(
            states, deterministic=deterministic
        )
        return actions, log_density, self.critic(states), entropy

    def evaluate_actions(self, states, actions):
        log_density, entropy = self.actor.evaluate_actions(states, actions)
        return log_density, self.critic(states), entropy


def relative_pose_error6(cage_pose6: np.ndarray, tcp_pose6: np.ndarray):
    """Base-frame TCP-to-Cage translation and shortest rotation-vector error."""
    target = np.asarray(cage_pose6, dtype=float)
    tcp = np.asarray(tcp_pose6, dtype=float)
    if target.shape != (6,) or tcp.shape != (6,):
        raise ValueError("Cage and TCP poses must both use xyz_rotvec pose6")
    target_rotation = rotation_as_matrix(Rotation.from_rotvec(target[3:]))
    tcp_rotation = rotation_as_matrix(Rotation.from_rotvec(tcp[3:]))
    orientation_error = rotation_from_matrix(
        target_rotation @ tcp_rotation.T
    ).as_rotvec()
    return np.concatenate([target[:3] - tcp[:3], orientation_error])


def build_ppo_state(
    cage_pose6: np.ndarray,
    tcp_pose6: np.ndarray,
    joints: np.ndarray,
    joint_velocities: np.ndarray,
):
    state = np.concatenate(
        [
            relative_pose_error6(cage_pose6, tcp_pose6),
            np.asarray(joints),
            np.asarray(joint_velocities),
        ]
    ).astype(np.float32)
    if state.shape != (18,):
        raise ValueError("PPO2 state is relative pose error6 + q6 + qdot6 only")
    return state


def compute_gae(
    rewards,
    values,
    next_values,
    dones,
    gamma: float,
    gae_lambda: float,
    bootstrap_dones=None,
):
    """Compute GAE with separate episode termination and sequence boundaries.

    ``dones`` stops advantage recursion across tasks/slots/batches.  A time-limit
    collection truncation must also stop that recursion, but it may still
    bootstrap from V(next_state); ``bootstrap_dones`` controls only that TD
    bootstrap.  Existing callers that omit it retain the original behavior.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    bootstrap_dones = (
        dones
        if bootstrap_dones is None
        else np.asarray(bootstrap_dones, dtype=np.float64)
    )
    if not (
        rewards.shape
        == values.shape
        == next_values.shape
        == dones.shape
        == bootstrap_dones.shape
    ):
        raise ValueError("GAE arrays must have identical shapes")
    advantages = np.zeros_like(rewards)
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = 1.0 - bootstrap_dones[index]
        continuation = 1.0 - dones[index]
        td_error = (
            rewards[index]
            + gamma * bootstrap * next_values[index]
            - values[index]
        )
        accumulator = (
            td_error
            + gamma * gae_lambda * continuation * accumulator
        )
        advantages[index] = accumulator
    return advantages


@dataclass
class RolloutBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    old_log_densities: np.ndarray
    old_values: np.ndarray
    next_values: np.ndarray
    bootstrap_dones: Optional[np.ndarray] = None


class PPO2Trainer:
    def __init__(self, config: Dict, device: str = "auto"):
        self.config = config
        self.device = resolve_torch_device(device)
        self.model = SharedActorCritic(
            int(config["state_dim"]), int(config["action_dim"]), config["hidden_dims"]
        ).to(self.device)
        self.initialization_contract = "random_actor_critic_v1"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config["learning_rate"]))
        self.state_normalizer = RunningStateNormalizer(
            int(config["state_dim"]),
            float(config["state_normalization_clip"]),
            float(config["state_normalization_epsilon"]),
        )

    def normalize_state(self, state, update: bool):
        if update:
            self.state_normalizer.update(state)
        return self.state_normalizer.normalize(state)

    def rebuild_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.config["learning_rate"])
        )

    def reset_critic(self):
        self.model.critic = CriticNetwork(
            int(self.config["state_dim"]), self.config["hidden_dims"]
        ).to(self.device)

    def actor_auxiliary_loss(self, states):
        """Optional policy regularizer used by specialized PPO trainers."""
        zero = states.new_zeros(())
        return zero, {"actor_auxiliary_loss": zero}

    def after_optimizer_step(self):
        """Optional projection hook used by specialized PPO trainers."""
        return None

    def update(self, rollout: RolloutBatch):
        raw_advantages = compute_gae(
            rollout.rewards,
            rollout.old_values,
            rollout.next_values,
            rollout.dones,
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
            bootstrap_dones=rollout.bootstrap_dones,
        )
        # The value target is the same GAE lambda-return used by the actor,
        # rather than a separate high-variance Monte Carlo return.
        critic_targets = raw_advantages + np.asarray(
            rollout.old_values, dtype=np.float64
        )
        advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std() + 1e-8
        )
        tensors = {
            "states": torch.as_tensor(rollout.states, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(rollout.actions, dtype=torch.float32, device=self.device),
            "old_log": torch.as_tensor(rollout.old_log_densities, dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(critic_targets, dtype=torch.float32, device=self.device),
        }
        sample_count = len(rollout.states)
        minibatch = int(self.config["minibatch_size"])
        clip_epsilon = float(self.config["clip_epsilon"])
        target_kl = float(self.config["target_kl"])
        kl_stop = target_kl * float(self.config["kl_stop_multiplier"])
        metric_history = {
            "actor_loss": [],
            "actor_auxiliary_loss": [],
            "value_loss": [],
            "entropy": [],
            "density_ratio_mean": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        early_stopped = False
        epochs_completed = 0
        for epoch in range(int(self.config["update_epochs"])):
            permutation = np.random.permutation(sample_count)
            for start in range(0, sample_count, minibatch):
                indices = torch.as_tensor(permutation[start : start + minibatch], device=self.device)
                log_density, values, entropy = self.model.evaluate_actions(
                    tensors["states"][indices], tensors["actions"][indices]
                )
                log_ratio = log_density - tensors["old_log"][indices]
                density_ratio = torch.exp(log_ratio)
                approx_kl = torch.mean(
                    (density_ratio - 1.0) - log_ratio
                )
                clip_fraction = torch.mean(
                    (torch.abs(density_ratio - 1.0) > clip_epsilon).float()
                )
                advantages_batch = tensors["advantages"][indices]
                unclipped = density_ratio * advantages_batch
                clipped = torch.clamp(
                    density_ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon,
                ) * advantages_batch
                actor_loss = -torch.mean(torch.minimum(unclipped, clipped))
                auxiliary_loss, auxiliary_metrics = self.actor_auxiliary_loss(
                    tensors["states"][indices]
                )
                value_loss = 0.5 * torch.mean((values - tensors["returns"][indices]) ** 2)
                entropy_bonus = torch.mean(entropy)
                loss = (
                    actor_loss
                    + auxiliary_loss
                    + float(self.config["value_coefficient"]) * value_loss
                    - float(self.config["entropy_coefficient"]) * entropy_bonus
                )
                for key, value in (
                    ("actor_loss", actor_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy_bonus),
                    ("density_ratio_mean", density_ratio.mean()),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                ):
                    metric_history[key].append(float(value.item()))
                for key, value in auxiliary_metrics.items():
                    metric_history.setdefault(key, []).append(
                        float(value.detach().item())
                    )
                if float(approx_kl.item()) > kl_stop:
                    early_stopped = True
                    break
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.config["max_grad_norm"])
                )
                self.optimizer.step()
                self.after_optimizer_step()
            if early_stopped:
                break
            epochs_completed = epoch + 1

        with torch.no_grad():
            _, _, final_values = self.model(tensors["states"])
        final_values_numpy = final_values.detach().cpu().numpy()
        target_variance = float(np.var(critic_targets))
        explained_variance = (
            0.0
            if target_variance < 1e-12
            else 1.0
            - float(np.var(critic_targets - final_values_numpy))
            / target_variance
        )
        saturation_threshold = float(
            self.config["action_saturation_threshold"]
        )
        action_saturation_fraction = float(
            np.mean(np.abs(rollout.actions) >= saturation_threshold)
        )
        metrics = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in metric_history.items()
        }
        metrics.update(
            {
                "approx_kl_max": float(
                    max(metric_history["approx_kl"], default=0.0)
                ),
                "target_kl": target_kl,
                "kl_early_stopped": bool(early_stopped),
                "epochs_completed": int(epochs_completed),
                "explained_variance": float(explained_variance),
                "action_saturation_fraction": action_saturation_fraction,
                "state_normalizer_count": float(self.state_normalizer.count),
            }
        )
        return metrics

    def save(
        self,
        path: str,
        training_metadata=None,
        include_optimizer: bool = False,
    ):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
                "model_state": {
                    name: tensor.detach().cpu() for name, tensor in self.model.state_dict().items()
                },
                "format_version": PPO_CHECKPOINT_FORMAT_VERSION,
                "state_dim": 18,
                "action_dim": 6,
                "hidden_dims": list(self.config["hidden_dims"]),
                "critic_target": PPO_CRITIC_TARGET_CONTRACT,
                "actor_advantage": "gae",
                "action_distribution": PPO_ACTION_DISTRIBUTION_CONTRACT,
                "entropy_estimator": PPO_ENTROPY_CONTRACT,
                "reward_target_contract": PPO_REWARD_TARGET_CONTRACT,
                "state_normalization_contract": PPO_STATE_NORMALIZATION_CONTRACT,
                "state_contract": PPO_STATE_CONTRACT,
                "network_contract": PPO_NETWORK_CONTRACT,
                "initialization_contract": self.initialization_contract,
                "state_normalizer": self.state_normalizer.state_dict(),
                "decision_trigger": "one_action_per_completed_visual_update",
                "history_quality_contract": HISTORY_QUALITY_CONTRACT,
                "history_suppression_contract": PPO_HISTORY_SUPPRESSION_CONTRACT,
                "training_reset_contract": PPO_TRAINING_RESET_CONTRACT,
                "cage_geometry_contract": CAGE_GEOMETRY_CONTRACT,
                "training_metadata": dict(training_metadata or {}),
            }
        if include_optimizer:
            checkpoint["optimizer_state"] = self.optimizer.state_dict()
        # A fixed temporary sibling plus atomic replacement prevents a power or
        # CUDA failure from destroying the last complete checkpoint.
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(destination)


class FrozenPPOPolicy:
    """Inference wrapper that freezes training-time state normalization."""

    def __init__(self, model, normalizer_state, device):
        self.model = model
        self.device = device
        self.mean = torch.as_tensor(
            normalizer_state["mean"], dtype=torch.float32, device=device
        )
        self.variance = torch.as_tensor(
            normalizer_state["variance"], dtype=torch.float32, device=device
        )
        self.epsilon = float(normalizer_state["epsilon"])
        self.clip = float(normalizer_state["clip"])

    def normalize(self, states):
        states = states.to(device=self.device, dtype=self.mean.dtype)
        if states.shape[-1] != 18:
            raise ValueError("Frozen PPO policy expects raw states with 18 features")
        return torch.clamp(
            (states - self.mean) / torch.sqrt(self.variance + self.epsilon),
            -self.clip,
            self.clip,
        )

    def sample(self, raw_states, deterministic=True):
        return self.model.sample(
            self.normalize(raw_states), deterministic=deterministic
        )


def _validate_ppo_checkpoint(checkpoint):
    if (
        int(checkpoint.get("format_version", 0))
        != PPO_CHECKPOINT_FORMAT_VERSION
        or checkpoint.get("decision_trigger")
        != "one_action_per_completed_visual_update"
        or checkpoint.get("action_distribution")
        != PPO_ACTION_DISTRIBUTION_CONTRACT
        or checkpoint.get("entropy_estimator") != PPO_ENTROPY_CONTRACT
        or checkpoint.get("reward_target_contract")
        != PPO_REWARD_TARGET_CONTRACT
        or checkpoint.get("state_normalization_contract")
        != PPO_STATE_NORMALIZATION_CONTRACT
        or checkpoint.get("state_contract") != PPO_STATE_CONTRACT
        or checkpoint.get("network_contract") != PPO_NETWORK_CONTRACT
        or checkpoint.get("initialization_contract")
        not in PPO_ALLOWED_INITIALIZATION_CONTRACTS
        or checkpoint.get("critic_target") != PPO_CRITIC_TARGET_CONTRACT
        or checkpoint.get("history_quality_contract")
        != HISTORY_QUALITY_CONTRACT
        or checkpoint.get("history_suppression_contract")
        != PPO_HISTORY_SUPPRESSION_CONTRACT
        or checkpoint.get("training_reset_contract")
        != PPO_TRAINING_RESET_CONTRACT
        or "state_normalizer" not in checkpoint
        or checkpoint.get("cage_geometry_contract") != CAGE_GEOMETRY_CONTRACT
    ):
        raise ValueError(
            "Unsupported PPO2 checkpoint: expected relative TCP-to-Cage state, "
            "independent Actor/Critic, normalized state, tanh entropy, GAE "
            "lambda-return, same-Cage reward, and abnormal-history hold format"
        )


def load_frozen_ppo(path: str, device: str = "auto"):
    resolved_device = resolve_torch_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_ppo_checkpoint(checkpoint)
    model = SharedActorCritic(
        checkpoint["state_dim"], checkpoint["action_dim"], checkpoint["hidden_dims"]
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalizer_state = checkpoint["state_normalizer"]
    for key in ("mean", "variance", "epsilon", "clip"):
        if key not in normalizer_state:
            raise ValueError("PPO state normalizer checkpoint is incomplete")
    if (
        np.asarray(normalizer_state["mean"]).shape != (18,)
        or np.asarray(normalizer_state["variance"]).shape != (18,)
    ):
        raise ValueError("PPO state normalizer must contain 18-D statistics")
    return FrozenPPOPolicy(model, normalizer_state, resolved_device)


def load_trainable_ppo(path: str, config: Dict, device: str = "auto"):
    """Restore a PPO trainer and, when present, its Adam continuation state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_ppo_checkpoint(checkpoint)
    expected_hidden = [int(value) for value in config["hidden_dims"]]
    if (
        int(checkpoint["state_dim"]) != int(config["state_dim"])
        or int(checkpoint["action_dim"]) != int(config["action_dim"])
        or [int(value) for value in checkpoint["hidden_dims"]]
        != expected_hidden
    ):
        raise ValueError(
            "Resume checkpoint network dimensions do not match the current config"
        )
    trainer = PPO2Trainer(config, device=device)
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.initialization_contract = checkpoint["initialization_contract"]
    trainer.state_normalizer.load_state_dict(checkpoint["state_normalizer"])
    optimizer_restored = "optimizer_state" in checkpoint
    if optimizer_restored:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        for optimizer_state in trainer.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(trainer.device)
    else:
        # Expert-stage checkpoints intentionally do not contain stale PPO Adam
        # moments; PPO therefore begins with a clean optimizer from that stage.
        trainer.rebuild_optimizer()
    return (
        trainer,
        dict(checkpoint.get("training_metadata", {})),
        bool(optimizer_restored),
    )
