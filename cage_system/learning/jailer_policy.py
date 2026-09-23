"""JAiLeR position backbone and independently scaled pose residual."""

from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn

from cage_system.learning.ppo2 import (
    ActorNetwork,
    CriticNetwork,
    PPO2Trainer,
    SharedActorCritic,
    compute_gae,
)


LEGACY_COMBINED_ACTION = "combined_mean_then_global_stage_scale_v1"
SEPARATE_RESIDUAL_ACTION = (
    "frozen_position_plus_error_scaled_pose_residual_v2"
)
JOINT_REFERENCE_ACTION = (
    "joint_pose_actor_dynamic_speed_with_position_reference_v1"
)
JOINT_PPO_ACTION = "joint_pose_actor_dynamic_speed_ppo_only_v2"
JOINT_ACTION_MODES = {JOINT_REFERENCE_ACTION, JOINT_PPO_ACTION}
JOINT_PPO_OPTIMIZER_CONTRACT = (
    "independent_actor_critic_adam_and_gradient_clipping_v2"
)


def _explained_variance(targets, predictions):
    """Finite explained variance for a fixed rollout target."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    target_variance = float(np.var(targets))
    if target_variance < 1e-12:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / target_variance)


def pose_residual_gain(orientation_error_deg, config: Dict, stage_cap: float):
    """Map physical rotation error to a deterministic piecewise-linear gain."""
    errors = np.asarray(
        config["pose_residual_gain_orientation_error_deg"], dtype=float
    )
    gains = np.asarray(config["pose_residual_gain_values"], dtype=float)
    values = np.asarray(orientation_error_deg, dtype=float)
    if (
        errors.ndim != 1
        or gains.shape != errors.shape
        or len(errors) < 2
        or not np.all(np.diff(errors) > 0.0)
        or not np.all(np.diff(gains) >= 0.0)
        or errors[0] < 0.0
        or gains[0] < 0.0
        or gains[-1] > 1.0
    ):
        raise ValueError("Invalid monotone pose-residual gain schedule")
    cap = float(stage_cap)
    if cap <= 0.0 or cap > 1.0:
        raise ValueError("Pose-residual stage gain cap must be inside (0,1]")
    interpolated = np.interp(values, errors, gains)
    return np.minimum(interpolated, cap).astype(np.float32)


def joint_action_gain(raw_states, config: Dict, stage: Dict):
    """Scale one joint policy from the worst normalized TCP pose error."""
    raw = np.asarray(raw_states, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 18:
        raise ValueError("JAiLeR raw policy state must have shape [N,18]")
    position_ratio = np.linalg.norm(raw[:, :3], axis=1) / float(
        stage["success_position_m"]
    )
    if str(stage["task"]) == "pose":
        orientation_ratio = np.rad2deg(
            np.linalg.norm(raw[:, 3:6], axis=1)
        ) / float(stage["success_orientation_deg"])
    else:
        orientation_ratio = np.zeros_like(position_ratio)
    difficulty = np.maximum(position_ratio, orientation_ratio)
    anchors = np.asarray(config["joint_action_gain_error_ratio"], dtype=float)
    gains = np.asarray(config["joint_action_gain_values"], dtype=float)
    if (
        anchors.ndim != 1
        or anchors.shape != gains.shape
        or len(anchors) < 2
        or anchors[0] < 0.0
        or not np.all(np.diff(anchors) > 0.0)
        or gains[0] <= 0.0
        or gains[-1] > 1.0
        or not np.all(np.diff(gains) >= 0.0)
    ):
        raise ValueError("Invalid joint-action gain schedule")
    interpolated = np.interp(difficulty, anchors, gains)
    return np.minimum(
        interpolated, float(stage["action_velocity_scale"])
    ).astype(np.float32)


def _residual_mlp(input_dim: int, hidden_dims: Sequence[int]):
    layers = []
    current = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, int(hidden)), nn.Tanh()])
        current = int(hidden)
    return nn.Sequential(*layers), current


class JailerResidualActor(ActorNetwork):
    """Preserve a position policy while learning pose corrections separately."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        residual_hidden_dims: Sequence[int],
    ):
        super().__init__(state_dim, action_dim, hidden_dims)
        self.action_composition_mode = str(
            getattr(self, "action_composition_mode", LEGACY_COMBINED_ACTION)
        )
        self.pose_residual_features, residual_dim = _residual_mlp(
            state_dim, residual_hidden_dims
        )
        self.pose_residual_head = nn.Linear(residual_dim, action_dim)
        nn.init.zeros_(self.pose_residual_head.weight)
        nn.init.zeros_(self.pose_residual_head.bias)
        self.register_buffer("pose_residual_gate", torch.tensor(0.0))

    def configure_action_composition(self, config: Dict) -> None:
        mode = str(
            config.get("action_composition_mode", LEGACY_COMBINED_ACTION)
        )
        if mode not in {LEGACY_COMBINED_ACTION, SEPARATE_RESIDUAL_ACTION}:
            raise ValueError("Unsupported JAiLeR action-composition mode")
        self.action_composition_mode = mode
        if mode == SEPARATE_RESIDUAL_ACTION:
            initial_log_std = float(
                config.get("pose_residual_initial_log_std", -1.2)
            )
            self.pose_residual_log_std = nn.Parameter(
                torch.full_like(self.log_std.detach(), initial_log_std)
            )

    def position_means(self, states):
        return super().forward(states)[0]

    def deterministic_position_action(self, states):
        return torch.tanh(self.position_means(states))

    def forward(self, states):
        base_means, log_std = super().forward(states)
        residual = self.pose_residual_head(
            self.pose_residual_features(states)
        )
        if (
            self.action_composition_mode == SEPARATE_RESIDUAL_ACTION
            and bool(self.pose_residual_gate.item() > 0.5)
        ):
            residual_log_std = torch.clamp(
                self.pose_residual_log_std, -5.0, 1.0
            ).expand_as(residual)
            return residual, residual_log_std
        return base_means + self.pose_residual_gate * residual, log_std

    def set_curriculum_task(self, task: str) -> None:
        task = str(task)
        if task not in {"position", "pose"}:
            raise ValueError("JAiLeR Actor task must be position or pose")
        pose_enabled = task == "pose"
        self.pose_residual_gate.fill_(1.0 if pose_enabled else 0.0)
        # The learned position inverse mapping becomes a fixed base controller
        # in pose stages.  Only the residual and exploration parameter can
        # change, preventing pose learning from destroying position control.
        for parameter in self.features.parameters():
            parameter.requires_grad_(not pose_enabled)
        for parameter in self.mean_head.parameters():
            parameter.requires_grad_(not pose_enabled)
        self.log_std.requires_grad_(not pose_enabled)
        for parameter in self.pose_residual_features.parameters():
            parameter.requires_grad_(pose_enabled)
        for parameter in self.pose_residual_head.parameters():
            parameter.requires_grad_(pose_enabled)
        if hasattr(self, "pose_residual_log_std"):
            self.pose_residual_log_std.requires_grad_(pose_enabled)


class JailerResidualActorCritic(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.actor = JailerResidualActor(
            int(config["state_dim"]),
            int(config["action_dim"]),
            config["hidden_dims"],
            config["pose_residual_hidden_dims"],
        )
        self.actor.configure_action_composition(config)
        self.critic = CriticNetwork(
            int(config["state_dim"]), config["hidden_dims"]
        )

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


class JailerJointReferenceActorCritic(nn.Module):
    """One trainable pose Actor plus a permanently frozen P6 reference."""

    def __init__(self, config: Dict):
        super().__init__()
        state_dim = int(config["state_dim"])
        action_dim = int(config["action_dim"])
        hidden_dims = config["hidden_dims"]
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.reference_actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.reference_actor.load_state_dict(self.actor.state_dict())
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)
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


class JailerPPO2Trainer(PPO2Trainer):
    """PPO optimizer using the JAiLeR-specific residual policy contract."""

    def __init__(self, config: Dict, device: str = "auto"):
        self.action_composition_mode = str(
            config.get("action_composition_mode", LEGACY_COMBINED_ACTION)
        )
        super().__init__(config, device=device)
        if self.action_composition_mode == JOINT_REFERENCE_ACTION:
            self.model = JailerJointReferenceActorCritic(config).to(self.device)
        elif self.action_composition_mode == JOINT_PPO_ACTION:
            # The Observed-P6 Actor is used once as an initialization by the
            # warm-start loader.  No frozen copy or imitation loss exists in
            # this policy: subsequent changes are driven only by PPO returns.
            self.model = SharedActorCritic(
                int(config["state_dim"]),
                int(config["action_dim"]),
                config["hidden_dims"],
            ).to(self.device)
            # A from-scratch JAiLeR curriculum may request less exploration
            # than ActorNetwork's generic default. Resume and warm-start loads
            # subsequently replace this parameter from their checkpoint.
            if "initial_log_std" in config:
                with torch.no_grad():
                    self.model.actor.log_std.fill_(
                        float(config["initial_log_std"])
                    )
        else:
            self.model = JailerResidualActorCritic(config).to(self.device)
        self._training_stage = None
        self._training_updates_in_stage = 0
        self._adaptive_entropy_multiplier = 1.0
        self.adaptive_exploration_state = {
            "stage_name": None,
            "score_history": [],
            "boosts_used": 0,
            "boost_updates_remaining": 0,
        }
        self.rebuild_optimizer()

    def rebuild_optimizer(self):
        actor = getattr(getattr(self, "model", None), "actor", None)
        if self.action_composition_mode == JOINT_PPO_ACTION:
            # V12 deliberately has two independent optimization problems.
            # Keeping separate Adam instances prevents Critic gradients,
            # clipping and optimizer moments from affecting the Actor update.
            self.actor_optimizer = torch.optim.Adam(
                list(actor.parameters()),
                lr=float(self.config["joint_actor_learning_rate"]),
            )
            self.critic_optimizer = torch.optim.Adam(
                list(self.model.critic.parameters()),
                lr=float(self.config["joint_critic_learning_rate"]),
            )
            actor_ids = {
                id(parameter) for parameter in self.model.actor.parameters()
            }
            critic_ids = {
                id(parameter) for parameter in self.model.critic.parameters()
            }
            if actor_ids.intersection(critic_ids):
                raise RuntimeError(
                    "JAiLeR V12 Actor and Critic parameters must be disjoint"
                )
            # PPO2Trainer.__init__ created a temporary combined optimizer for
            # its temporary model.  Do not leave a misleading usable handle.
            if hasattr(self, "optimizer"):
                del self.optimizer
            return
        if self.action_composition_mode == JOINT_REFERENCE_ACTION:
            self.optimizer = torch.optim.Adam(
                [
                    {
                        "params": list(actor.parameters()),
                        "lr": float(self.config["joint_actor_learning_rate"]),
                    },
                    {
                        "params": list(self.model.critic.parameters()),
                        "lr": float(self.config["joint_critic_learning_rate"]),
                    },
                ]
            )
            return
        if actor is None or not hasattr(actor, "pose_residual_head"):
            super().rebuild_optimizer()
            return
        base_learning_rate = float(self.config["learning_rate"])
        residual_learning_rate = float(
            self.config["pose_residual_learning_rate"]
        )
        base_parameters = (
            list(actor.features.parameters())
            + list(actor.mean_head.parameters())
            + [actor.log_std]
        )
        residual_parameters = (
            list(actor.pose_residual_features.parameters())
            + list(actor.pose_residual_head.parameters())
        )
        if hasattr(actor, "pose_residual_log_std"):
            residual_parameters.append(actor.pose_residual_log_std)
        self.optimizer = torch.optim.Adam(
            [
                {"params": base_parameters, "lr": base_learning_rate},
                {"params": residual_parameters, "lr": residual_learning_rate},
                {
                    "params": list(self.model.critic.parameters()),
                    "lr": base_learning_rate,
                },
            ]
        )

    @property
    def uses_independent_actor_critic_optimizers(self):
        return self.action_composition_mode == JOINT_PPO_ACTION

    def set_curriculum_task(self, task: str) -> None:
        if self.action_composition_mode in JOINT_ACTION_MODES:
            if str(task) not in {"position", "pose"}:
                raise ValueError("JAiLeR Actor task must be position or pose")
            for parameter in self.model.actor.parameters():
                parameter.requires_grad_(True)
            if self.action_composition_mode == JOINT_REFERENCE_ACTION:
                for parameter in self.model.reference_actor.parameters():
                    parameter.requires_grad_(False)
            return
        self.model.actor.set_curriculum_task(task)

    def active_log_std(self):
        actor = self.model.actor
        if self.action_composition_mode in JOINT_ACTION_MODES:
            return actor.log_std
        if (
            self.action_composition_mode == SEPARATE_RESIDUAL_ACTION
            and bool(actor.pose_residual_gate.item() > 0.5)
        ):
            return actor.pose_residual_log_std
        return actor.log_std

    def prepare_training_stage(
        self,
        stage: Dict,
        entering: bool = False,
        updates_in_stage=None,
    ):
        self._training_stage = dict(stage)
        self._training_updates_in_stage = int(
            0 if entering or updates_in_stage is None else updates_in_stage
        )
        if self.action_composition_mode not in JOINT_ACTION_MODES:
            return {}
        log_std = self.model.actor.log_std
        before = log_std.detach().cpu().numpy().copy()
        with torch.no_grad():
            if entering:
                minimum = float(stage["joint_log_std_entry_min"])
                log_std.copy_(
                    torch.maximum(log_std, torch.full_like(log_std, minimum))
                )
            log_std.clamp_(min=float(stage["joint_log_std_floor"]), max=1.0)
        result = {
            "joint_log_std_before": before.tolist(),
            "joint_log_std_after": log_std.detach().cpu().numpy().tolist(),
        }
        if self.action_composition_mode == JOINT_REFERENCE_ACTION:
            result["position_reference_loss_weight"] = float(
                stage["position_reference_loss_weight"]
            )
        return result

    def effective_entropy_coefficient(self):
        """Return the active curriculum-stage entropy coefficient.

        V12 linearly interpolates from the stage's start coefficient to its
        end coefficient during ``minimum_updates`` PPO updates.  Critic-only
        alignment does not increment ``updates_in_stage``, so it cannot
        accidentally consume the Actor exploration schedule.  Older policy
        contracts retain their single global coefficient.
        """
        if self.action_composition_mode != JOINT_PPO_ACTION:
            return float(self.config["entropy_coefficient"])
        schedule = self.config.get("entropy_schedule")
        if schedule is None:
            return float(self.config["entropy_coefficient"]) * float(
                self._adaptive_entropy_multiplier
            )
        if self._training_stage is None:
            raise RuntimeError(
                "V12 entropy schedule requires an active curriculum stage"
            )
        stage_name = str(self._training_stage["name"])
        stage_schedule = schedule["stages"].get(stage_name)
        if stage_schedule is None:
            raise RuntimeError(
                "V12 entropy schedule has no entry for '{}'".format(
                    stage_name
                )
            )
        start = float(stage_schedule["start"])
        end = float(stage_schedule["end"])
        denominator = max(
            1, int(self._training_stage.get("minimum_updates", 1))
        )
        fraction = np.clip(
            float(self._training_updates_in_stage) / float(denominator),
            0.0,
            1.0,
        )
        coefficient = float(start + fraction * (end - start))
        return coefficient * float(self._adaptive_entropy_multiplier)

    def set_adaptive_entropy_multiplier(self, multiplier: float) -> None:
        multiplier = float(multiplier)
        if not np.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError(
                "Adaptive entropy multiplier must be finite and at least one"
            )
        self._adaptive_entropy_multiplier = multiplier

    def increase_exploration_log_std(self, increment: float):
        """Increase policy exploration once, subject to the stage floor."""
        increment = float(increment)
        if not np.isfinite(increment) or increment <= 0.0:
            raise ValueError("log_std exploration increment must be positive")
        parameter = self.active_log_std()
        with torch.no_grad():
            before = parameter.detach().cpu().numpy().copy()
            parameter.add_(increment)
            parameter.clamp_(max=1.0)
            if self._training_stage is not None:
                parameter.clamp_(
                    min=float(self._training_stage["joint_log_std_floor"])
                )
            after = parameter.detach().cpu().numpy().copy()
        return {
            "adaptive_log_std_before": before.tolist(),
            "adaptive_log_std_after": after.tolist(),
        }

    def prepare_adaptive_exploration(self, stage: Dict) -> None:
        """Reset the plateau controller at a real curriculum boundary."""
        stage_name = str(stage["name"])
        if self.adaptive_exploration_state.get("stage_name") == stage_name:
            return
        self._adaptive_entropy_multiplier = 1.0
        self.adaptive_exploration_state = {
            "stage_name": stage_name,
            "score_history": [],
            "boosts_used": 0,
            "boost_updates_remaining": 0,
        }

    def update_adaptive_exploration(
        self, stage: Dict, rollout_metrics: Dict, optimizer_metrics: Dict
    ) -> Dict:
        """Apply one bounded exploration pulse after a genuine plateau."""
        settings = self.config.get("adaptive_exploration")
        if not settings or not bool(settings.get("enabled", False)):
            return {"adaptive_exploration_event": "disabled"}
        self.prepare_adaptive_exploration(stage)
        state = self.adaptive_exploration_state
        remaining = int(state["boost_updates_remaining"])
        if remaining > 0:
            remaining -= 1
            state["boost_updates_remaining"] = remaining
            if remaining == 0:
                self.set_adaptive_entropy_multiplier(1.0)

        components = [
            float(rollout_metrics["final_position_error_mean_m"])
            / float(stage["promotion_position_mean_m"]),
        ]
        if str(stage["task"]) == "pose":
            components.append(
                float(rollout_metrics["final_orientation_error_mean_deg"])
                / float(stage["promotion_orientation_mean_deg"])
            )
        success_target = max(1e-6, float(stage["promotion_success_rate"]))
        components.append(
            max(
                0.0,
                success_target
                - float(rollout_metrics["terminal_success_rate"]),
            )
            / success_target
        )
        score = float(np.mean(components))
        history = list(state["score_history"])
        history.append(score)
        patience = int(settings["patience_updates"])
        history = history[-2 * patience :]
        state["score_history"] = history

        event = "observing"
        improvement = None
        critic_ev = float(
            optimizer_metrics.get(
                "critic_pre_update_explained_variance", float("-inf")
            )
        )
        if (
            remaining == 0
            and len(history) == 2 * patience
            and int(state["boosts_used"])
            < int(settings["maximum_boosts_per_stage"])
            and np.isfinite(critic_ev)
            and critic_ev >= float(settings["critic_ev_threshold"])
        ):
            older = float(np.mean(history[:patience]))
            newer = float(np.mean(history[patience:]))
            improvement = (older - newer) / max(abs(older), 1e-8)
            threshold = float(settings["minimum_progress_improvement"])
            regression_tolerance = float(
                settings.get("plateau_regression_tolerance", 0.0)
            )
            if -regression_tolerance <= improvement < threshold:
                # A small negative change can be sampling noise because every
                # rollout contains newly randomized tasks.  Treat only this
                # bounded band as a plateau; a larger regression must never
                # trigger more exploration.  The pulse changes the entropy
                # coefficient temporarily and does not directly edit log_std.
                details = {"adaptive_forced_log_std_increment": 0.0}
                self.set_adaptive_entropy_multiplier(
                    float(settings["entropy_multiplier"])
                )
                state["boosts_used"] = int(state["boosts_used"]) + 1
                state["boost_updates_remaining"] = int(
                    settings["boost_updates"]
                )
                state["score_history"] = []
                event = "plateau_boost_started"
            else:
                details = {}
                event = "improving" if improvement >= threshold else "regressing"
        else:
            details = {}
            if remaining > 0:
                event = "boost_active"
            elif len(history) == 2 * patience and critic_ev < float(
                settings["critic_ev_threshold"]
            ):
                event = "plateau_check_blocked_by_critic"
        return {
            "adaptive_exploration_event": event,
            "adaptive_exploration_score": score,
            "adaptive_exploration_relative_improvement": improvement,
            "adaptive_exploration_regression_tolerance": float(
                settings.get("plateau_regression_tolerance", 0.0)
            ),
            "adaptive_exploration_boosts_used": int(state["boosts_used"]),
            "adaptive_exploration_boost_updates_remaining": int(
                state["boost_updates_remaining"]
            ),
            "adaptive_entropy_multiplier": float(
                self._adaptive_entropy_multiplier
            ),
            **details,
        }

    def after_optimizer_step(self):
        if (
            self.action_composition_mode not in JOINT_ACTION_MODES
            or self._training_stage is None
        ):
            return
        with torch.no_grad():
            self.model.actor.log_std.clamp_(
                min=float(self._training_stage["joint_log_std_floor"]),
                max=1.0,
            )

    def actor_auxiliary_loss(self, states):
        if self.action_composition_mode != JOINT_REFERENCE_ACTION:
            return super().actor_auxiliary_loss(states)
        if self._training_stage is None:
            raise RuntimeError("Joint Actor PPO update has no active stage")

        mean = torch.as_tensor(
            self.state_normalizer.mean,
            dtype=states.dtype,
            device=states.device,
        )
        variance = torch.as_tensor(
            self.state_normalizer.variance,
            dtype=states.dtype,
            device=states.device,
        )
        raw_states = states * torch.sqrt(
            variance + float(self.state_normalizer.epsilon)
        ) + mean
        position_error = torch.linalg.vector_norm(raw_states[:, :3], dim=1)
        position_ratio = position_error / float(
            self._training_stage["success_position_m"]
        )
        fade_start, full_strength = [
            float(value)
            for value in self.config["position_reference_gate_error_ratio"]
        ]
        gate = torch.clamp(
            (position_ratio - fade_start) / (full_strength - fade_start),
            0.0,
            1.0,
        )
        joint_action = torch.tanh(self.model.actor(states)[0])
        with torch.no_grad():
            reference_action = torch.tanh(
                self.model.reference_actor(states)[0]
            )
        per_state = torch.mean((joint_action - reference_action) ** 2, dim=1)
        nominal = float(
            self._training_stage["position_reference_loss_weight"]
        )
        reference_loss = torch.mean(nominal * gate * per_state)

        fraction = float(self.config["position_rehearsal_fraction"])
        rehearsal_weight = float(
            self.config["position_rehearsal_loss_weight"]
        )
        if fraction > 0.0 and rehearsal_weight > 0.0:
            rehearsal_count = min(
                len(states), max(1, int(round(len(states) * fraction)))
            )
            indices = torch.randperm(len(states), device=states.device)[
                :rehearsal_count
            ]
            rehearsal_states = states[indices].clone()
            zero_orientation = (
                -mean[3:6]
                / torch.sqrt(
                    variance[3:6] + float(self.state_normalizer.epsilon)
                )
            )
            rehearsal_states[:, 3:6] = zero_orientation
            rehearsal_joint = torch.tanh(
                self.model.actor(rehearsal_states)[0]
            )
            with torch.no_grad():
                rehearsal_reference = torch.tanh(
                    self.model.reference_actor(rehearsal_states)[0]
                )
            rehearsal_loss = rehearsal_weight * torch.mean(
                (rehearsal_joint - rehearsal_reference) ** 2
            )
        else:
            rehearsal_loss = states.new_zeros(())
        total = reference_loss + rehearsal_loss
        return total, {
            "actor_auxiliary_loss": total,
            "position_reference_loss": reference_loss,
            "position_rehearsal_loss": rehearsal_loss,
            "position_reference_gate_mean": torch.mean(gate),
        }

    def update(self, rollout):
        """Update V12 Actor and Critic as two independent problems.

        GAE and the lambda-return targets are computed once from the rollout's
        frozen old-policy values.  Actor KL stopping never shortens the Critic
        regression loop, and no value coefficient is used because there is no
        combined scalar loss in this contract.
        """
        if self.action_composition_mode != JOINT_PPO_ACTION:
            return super().update(rollout)

        raw_advantages = compute_gae(
            rollout.rewards,
            rollout.old_values,
            rollout.next_values,
            rollout.dones,
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
            bootstrap_dones=rollout.bootstrap_dones,
        )
        critic_targets = raw_advantages + np.asarray(
            rollout.old_values, dtype=np.float64
        )
        pre_update_explained_variance = _explained_variance(
            critic_targets, rollout.old_values
        )
        normalized_advantages = (
            raw_advantages - raw_advantages.mean()
        ) / (raw_advantages.std() + 1e-8)
        tensors = {
            "states": torch.as_tensor(
                rollout.states, dtype=torch.float32, device=self.device
            ),
            "actions": torch.as_tensor(
                rollout.actions, dtype=torch.float32, device=self.device
            ),
            "old_log": torch.as_tensor(
                rollout.old_log_densities,
                dtype=torch.float32,
                device=self.device,
            ),
            "advantages": torch.as_tensor(
                normalized_advantages,
                dtype=torch.float32,
                device=self.device,
            ),
            "critic_targets": torch.as_tensor(
                critic_targets, dtype=torch.float32, device=self.device
            ),
        }
        sample_count = len(tensors["states"])
        if sample_count == 0:
            raise ValueError("Cannot update JAiLeR PPO from an empty rollout")
        minibatch_size = int(self.config["minibatch_size"])
        actor_update_epochs = int(self.config["actor_update_epochs"])
        critic_update_epochs = int(self.config["critic_update_epochs"])
        clip_epsilon = float(self.config["clip_epsilon"])
        target_kl = float(self.config["target_kl"])
        kl_stop = target_kl * float(self.config["kl_stop_multiplier"])
        entropy_coefficient = self.effective_entropy_coefficient()

        actor_history = {
            "clipped_policy_objective": [],
            "negative_clipped_policy_objective": [],
            "actor_maximization_objective": [],
            "actor_minimization_objective": [],
            "entropy": [],
            "density_ratio_mean": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        actor_early_stopped = False
        actor_epochs_completed = 0
        actor_minibatches_completed = 0
        self.model.actor.train()
        for epoch in range(actor_update_epochs):
            permutation = np.random.permutation(sample_count)
            for start in range(0, sample_count, minibatch_size):
                indices = torch.as_tensor(
                    permutation[start : start + minibatch_size],
                    device=self.device,
                )
                log_density, entropy = self.model.actor.evaluate_actions(
                    tensors["states"][indices], tensors["actions"][indices]
                )
                log_ratio = log_density - tensors["old_log"][indices]
                density_ratio = torch.exp(log_ratio)
                approx_kl = torch.mean(
                    (density_ratio - 1.0) - log_ratio
                )
                clip_fraction = torch.mean(
                    (
                        torch.abs(density_ratio - 1.0) > clip_epsilon
                    ).float()
                )
                advantages = tensors["advantages"][indices]
                unclipped = density_ratio * advantages
                clipped = torch.clamp(
                    density_ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon,
                ) * advantages
                clipped_policy_objective = torch.mean(
                    torch.minimum(unclipped, clipped)
                )
                entropy_mean = torch.mean(entropy)
                actor_maximization_objective = (
                    clipped_policy_objective
                    + entropy_coefficient * entropy_mean
                )
                actor_minimization_objective = (
                    -actor_maximization_objective
                )
                for key, value in (
                    ("clipped_policy_objective", clipped_policy_objective),
                    (
                        "negative_clipped_policy_objective",
                        -clipped_policy_objective,
                    ),
                    (
                        "actor_maximization_objective",
                        actor_maximization_objective,
                    ),
                    (
                        "actor_minimization_objective",
                        actor_minimization_objective,
                    ),
                    ("entropy", entropy_mean),
                    ("density_ratio_mean", density_ratio.mean()),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                ):
                    actor_history[key].append(float(value.detach().item()))
                if float(approx_kl.item()) > kl_stop:
                    actor_early_stopped = True
                    break
                self.actor_optimizer.zero_grad()
                actor_minimization_objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(),
                    float(self.config["max_grad_norm"]),
                )
                self.actor_optimizer.step()
                self.after_optimizer_step()
                actor_minibatches_completed += 1
            if actor_early_stopped:
                break
            actor_epochs_completed = epoch + 1

        critic_losses = []
        critic_minibatches_completed = 0
        self.model.critic.train()
        for _ in range(critic_update_epochs):
            permutation = np.random.permutation(sample_count)
            for start in range(0, sample_count, minibatch_size):
                indices = torch.as_tensor(
                    permutation[start : start + minibatch_size],
                    device=self.device,
                )
                predicted_values = self.model.critic(
                    tensors["states"][indices]
                )
                critic_loss = 0.5 * torch.mean(
                    (
                        predicted_values
                        - tensors["critic_targets"][indices]
                    )
                    ** 2
                )
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(),
                    float(self.config["max_grad_norm"]),
                )
                self.critic_optimizer.step()
                critic_losses.append(float(critic_loss.detach().item()))
                critic_minibatches_completed += 1

        with torch.no_grad():
            fitted_values = (
                self.model.critic(tensors["states"]).detach().cpu().numpy()
            )
        explained_variance = _explained_variance(
            critic_targets, fitted_values
        )
        metrics = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in actor_history.items()
        }
        critic_loss_mean = (
            float(np.mean(critic_losses)) if critic_losses else 0.0
        )
        metrics.update(
            {
                "critic_loss": critic_loss_mean,
                # Compatibility alias for existing training-history readers.
                "value_loss": critic_loss_mean,
                "entropy_coefficient": float(entropy_coefficient),
                "approx_kl_max": float(
                    max(actor_history["approx_kl"], default=0.0)
                ),
                "target_kl": target_kl,
                "actor_kl_early_stopped": bool(actor_early_stopped),
                "kl_early_stopped": bool(actor_early_stopped),
                "actor_epochs_completed": int(actor_epochs_completed),
                "critic_epochs_completed": int(critic_update_epochs),
                # Compatibility alias: this always means Actor epochs.
                "epochs_completed": int(actor_epochs_completed),
                "actor_minibatches_completed": int(
                    actor_minibatches_completed
                ),
                "critic_minibatches_completed": int(
                    critic_minibatches_completed
                ),
                "explained_variance": float(explained_variance),
                "critic_pre_update_explained_variance": float(
                    pre_update_explained_variance
                ),
                "raw_advantage_mean": float(np.mean(raw_advantages)),
                "raw_advantage_std": float(np.std(raw_advantages)),
                "raw_advantage_positive_fraction": float(
                    np.mean(raw_advantages > 0.0)
                ),
                "action_saturation_fraction": float(
                    np.mean(
                        np.abs(np.asarray(rollout.actions))
                        >= float(
                            self.config["action_saturation_threshold"]
                        )
                    )
                ),
                "state_normalizer_count": float(
                    self.state_normalizer.count
                ),
                "optimizer_contract": JOINT_PPO_OPTIMIZER_CONTRACT,
                "optimization_mode": "independent_actor_critic_ppo",
            }
        )
        return metrics

    def align_critic(self, rollout):
        """Fit only V(s) to a fresh stage's fixed GAE lambda-return targets."""
        raw_advantages = compute_gae(
            rollout.rewards,
            rollout.old_values,
            rollout.next_values,
            rollout.dones,
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
            bootstrap_dones=rollout.bootstrap_dones,
        )
        targets = raw_advantages + np.asarray(
            rollout.old_values, dtype=np.float64
        )
        pre_update_explained_variance = _explained_variance(
            targets, rollout.old_values
        )
        states = torch.as_tensor(
            rollout.states, dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            targets, dtype=torch.float32, device=self.device
        )
        sample_count = len(states)
        minibatch = int(self.config["minibatch_size"])
        critic_epochs = int(
            self.config.get(
                "critic_update_epochs", self.config["update_epochs"]
            )
        )
        losses = []
        optimizer = (
            self.critic_optimizer
            if self.action_composition_mode == JOINT_PPO_ACTION
            else self.optimizer
        )
        for _ in range(critic_epochs):
            permutation = np.random.permutation(sample_count)
            for start in range(0, sample_count, minibatch):
                indices = torch.as_tensor(
                    permutation[start : start + minibatch],
                    device=self.device,
                )
                values = self.model.critic(states[indices])
                value_loss = 0.5 * torch.mean(
                    (values - returns[indices]) ** 2
                )
                optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(),
                    float(self.config["max_grad_norm"]),
                )
                optimizer.step()
                losses.append(float(value_loss.item()))
        with torch.no_grad():
            fitted = self.model.critic(states).detach().cpu().numpy()
        explained_variance = _explained_variance(targets, fitted)
        metrics = {
            "actor_objective_applied": False,
            "value_loss": float(np.mean(losses)) if losses else 0.0,
            "critic_loss": float(np.mean(losses)) if losses else 0.0,
            "entropy": 0.0,
            "entropy_coefficient": (
                float(self.effective_entropy_coefficient())
                if self.action_composition_mode == JOINT_PPO_ACTION
                else float(self.config["entropy_coefficient"])
            ),
            "density_ratio_mean": 1.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "approx_kl_max": 0.0,
            "target_kl": float(self.config["target_kl"]),
            "kl_early_stopped": False,
            "actor_kl_early_stopped": False,
            "actor_epochs_completed": 0,
            "critic_epochs_completed": critic_epochs,
            # Compatibility alias consistently means Actor epochs in V12.
            "epochs_completed": 0,
            "explained_variance": float(explained_variance),
            "critic_pre_update_explained_variance": float(
                pre_update_explained_variance
            ),
            "raw_advantage_mean": float(np.mean(raw_advantages)),
            "raw_advantage_std": float(np.std(raw_advantages)),
            "raw_advantage_positive_fraction": float(
                np.mean(raw_advantages > 0.0)
            ),
            "action_saturation_fraction": float(
                np.mean(
                    np.abs(np.asarray(rollout.actions))
                    >= float(self.config["action_saturation_threshold"])
                )
            ),
            "state_normalizer_count": float(self.state_normalizer.count),
            "optimization_mode": "critic_only_alignment",
        }
        if self.action_composition_mode == JOINT_PPO_ACTION:
            metrics["optimizer_contract"] = JOINT_PPO_OPTIMIZER_CONTRACT
        return metrics

    def sample_for_environment(
        self,
        normalized_states,
        raw_states,
        stage: Dict,
        deterministic: bool = False,
    ):
        """Sample the optimized action and compose the physical command.

        In pose stages of the v2 contract, PPO owns only the stochastic
        residual.  The frozen position action and the physical-error gain are
        deterministic parts of the environment mapping.
        """
        policy_action, log_density, value, entropy = self.model.sample(
            normalized_states, deterministic=deterministic
        )
        task = str(stage["task"])
        if self.action_composition_mode in JOINT_ACTION_MODES:
            gains = joint_action_gain(raw_states, self.config, stage)
            gain_tensor = torch.as_tensor(
                gains,
                dtype=policy_action.dtype,
                device=policy_action.device,
            ).unsqueeze(-1)
            executed = gain_tensor * policy_action
            zeros = torch.zeros_like(policy_action)
            return {
                "policy_action": policy_action,
                "executed_action": executed,
                "base_action": policy_action,
                "residual_action": zeros,
                "scaled_residual_action": zeros,
                "residual_gain": gains,
                "composition_clipped": np.zeros_like(
                    policy_action.detach().cpu().numpy(), dtype=bool
                ),
                "log_density": log_density,
                "value": value,
                "entropy": entropy,
            }
        if (
            self.action_composition_mode != SEPARATE_RESIDUAL_ACTION
            or task == "position"
        ):
            zeros = torch.zeros_like(policy_action)
            zeros_gain = np.zeros(len(policy_action), dtype=np.float32)
            return {
                "policy_action": policy_action,
                "executed_action": policy_action,
                "base_action": policy_action,
                "residual_action": zeros,
                "scaled_residual_action": zeros,
                "residual_gain": zeros_gain,
                "composition_clipped": np.zeros_like(
                    policy_action.detach().cpu().numpy(), dtype=bool
                ),
                "log_density": log_density,
                "value": value,
                "entropy": entropy,
            }

        raw = np.asarray(raw_states, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 18:
            raise ValueError("JAiLeR raw policy state must have shape [N,18]")
        orientation_error_deg = np.rad2deg(
            np.linalg.norm(raw[:, 3:6], axis=1)
        )
        gains = pose_residual_gain(
            orientation_error_deg,
            self.config,
            stage_cap=float(stage["action_velocity_scale"]),
        )
        gain_tensor = torch.as_tensor(
            gains,
            dtype=policy_action.dtype,
            device=policy_action.device,
        ).unsqueeze(-1)
        with torch.no_grad():
            base_action = self.model.actor.deterministic_position_action(
                normalized_states
            )
        scaled_residual = gain_tensor * policy_action
        preclip = base_action + scaled_residual
        executed = torch.clamp(preclip, -1.0, 1.0)
        return {
            "policy_action": policy_action,
            "executed_action": executed,
            "base_action": base_action,
            "residual_action": policy_action,
            "scaled_residual_action": scaled_residual,
            "residual_gain": gains,
            "composition_clipped": (
                torch.abs(preclip) > 1.0
            ).detach().cpu().numpy(),
            "log_density": log_density,
            "value": value,
            "entropy": entropy,
        }
