import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    config = _load_with_parent(Path(path).resolve(), set())
    _expand_jailer_three_phase_curriculum(config)
    _normalize_jailer_position_reward_contract(config)
    _remove_obsolete_joint_reference_settings(config)
    validate_config(config)
    return config


def _normalize_jailer_position_reward_contract(config: Dict[str, Any]) -> None:
    """Make every position stage audibly and operationally position-only."""
    jailer = config.get("jailer")
    if jailer is None:
        return
    for stage in jailer.get("curriculum", {}).get("stages", []):
        if str(stage.get("task", "")) == "position":
            stage["reward_orientation_quality_weight"] = 0.0


def _remove_obsolete_joint_reference_settings(config: Dict[str, Any]) -> None:
    """Keep the PPO-only joint-policy resolved contract free of V11 keys."""
    jailer = config.get("jailer")
    if jailer is None:
        return
    ppo = jailer.get("ppo", {})
    if str(ppo.get("action_composition_mode", "")) != (
        "joint_pose_actor_dynamic_speed_ppo_only_v2"
    ):
        return
    for key in (
        "position_reference_gate_error_ratio",
        "position_rehearsal_fraction",
        "position_rehearsal_loss_weight",
        # V12 has no combined Actor+Critic scalar loss, so a value-loss
        # coefficient would be both unused and misleading in the resolved
        # experiment contract.
        "value_coefficient",
    ):
        ppo.pop(key, None)
    for stage in jailer.get("curriculum", {}).get("stages", []):
        stage.pop("position_reference_loss_weight", None)


def _expand_jailer_three_phase_curriculum(config: Dict[str, Any]) -> None:
    """Expand one position curriculum into masked and observed P1--P6."""
    jailer = config.get("jailer")
    if jailer is None:
        return
    curriculum = jailer.get("curriculum", {})
    if not bool(curriculum.get("three_phase_full_training", False)):
        return
    original = list(curriculum.get("stages", []))
    position = [stage for stage in original if stage.get("task") == "position"]
    pose = [stage for stage in original if stage.get("task") == "pose"]
    if len(position) < 2 or len(pose) < 1:
        raise ValueError(
            "Three-phase JAiLeR training needs at least two position stages "
            "followed by pose stages"
        )
    alignment_updates = int(
        curriculum.get("p6_to_b1_critic_alignment_updates", 0)
    )
    first_pose_alignment_mode = str(
        pose[0].get("entry_critic_alignment_mode", "fixed_updates_v1")
    )
    if alignment_updates <= 0 and first_pose_alignment_mode == "fixed_updates_v1":
        raise ValueError("P6 to B1 Critic alignment must contain updates")
    expanded = []
    for source in position:
        stage = copy.deepcopy(source)
        name = str(stage["name"])
        suffix = name[len("position_") :] if name.startswith("position_") else name
        stage["name"] = "position_masked_{}".format(suffix)
        stage["curriculum_phase"] = "position_masked"
        stage["orientation_observation_mode"] = "masked_zero"
        stage["reward_orientation_quality_weight"] = 0.0
        expanded.append(stage)
    for index, source in enumerate(position):
        stage = copy.deepcopy(source)
        name = str(stage["name"])
        suffix = name[len("position_") :] if name.startswith("position_") else name
        stage["name"] = "position_observed_{}".format(suffix)
        stage["curriculum_phase"] = "position_observed"
        stage["orientation_observation_mode"] = "observed"
        # This pass exposes orientation inputs only so their running
        # normalization is valid before pose learning.  It remains a strictly
        # position-only objective and promotion contract.
        stage["reward_orientation_quality_weight"] = 0.0
        if index == 0:
            stage["entry_transition"] = "masked_to_observed_new_critic"
        expanded.append(stage)
    for index, source in enumerate(pose):
        stage = copy.deepcopy(source)
        stage["curriculum_phase"] = (
            "pose_bridge"
            if str(stage["name"]).startswith("pose_bridge_")
            else "pose_orientation"
        )
        stage["orientation_observation_mode"] = "observed"
        if index == 0:
            stage.setdefault(
                "entry_transition", "preserve_critic_reset_adam_align"
            )
            if str(
                stage.get(
                    "entry_critic_alignment_mode", "fixed_updates_v1"
                )
            ) == "fixed_updates_v1":
                stage["entry_critic_alignment_updates"] = alignment_updates
        expanded.append(stage)
    curriculum["stages"] = expanded


def _load_with_parent(path: Path, active_paths) -> Dict[str, Any]:
    if path in active_paths:
        raise ValueError("Circular config inheritance involving {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        child = yaml.safe_load(handle) or {}
    parent_reference = child.pop("extends", None)
    if parent_reference is None:
        return child
    parent_path = (path.parent / str(parent_reference)).resolve()
    parent = _load_with_parent(parent_path, active_paths | {path})
    return _deep_merge(parent, child)


def _deep_merge(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    if "control_rate_hz" in config["timing"]:
        raise ValueError(
            "PPO is visual-event driven; timing.control_rate_hz must not define an independent decision loop"
        )
    if float(config["timing"]["detection_rate_hz"]) <= 0.0:
        raise ValueError("timing.detection_rate_hz must be positive")
    visual_period = 1.0 / float(config["timing"]["detection_rate_hz"])
    trajectory_period = float(config["trajectory"]["sample_period_s"])
    if abs(trajectory_period - visual_period) > 1e-9:
        raise ValueError(
            "Trajectory sample_period_s must equal the visual detection period"
        )
    actionable_latency = float(
        config["timing"]["maximum_visual_to_action_latency_s"]
    )
    if actionable_latency <= 0.0 or actionable_latency >= visual_period:
        raise ValueError("Visual-to-action latency budget must be inside one visual period")
    history = int(config["timing"]["history_length"])
    if history != 5:
        raise ValueError("The agreed Cage input contract requires exactly five history frames")
    warmup_measurements = config["kalman"].get(
        "innovation_gate_warmup_measurements", 1
    )
    if (
        isinstance(warmup_measurements, bool)
        or int(warmup_measurements) != float(warmup_measurements)
        or not 1 <= int(warmup_measurements) <= history
    ):
        raise ValueError(
            "Kalman innovation-gate warmup must be an integer inside [1, history_length]"
        )
    bootstrap_measurements = config["kalman"].get("visual_bootstrap_measurements", 1)
    if (
        isinstance(bootstrap_measurements, bool)
        or int(bootstrap_measurements) != float(bootstrap_measurements)
        or not 1 <= int(bootstrap_measurements) <= history
    ):
        raise ValueError(
            "Kalman visual bootstrap must be an integer inside [1, history_length]"
        )
    for key in (
        "bootstrap_position_std_floor_m",
        "bootstrap_velocity_std_floor_mps",
        "bootstrap_acceleration_std_floor_mps2",
        "bootstrap_max_residual_m",
    ):
        if key in config["kalman"] and float(config["kalman"][key]) <= 0.0:
            raise ValueError("Kalman {} must be positive".format(key))
    view_loss_misses = config["vision"].get("view_loss_consecutive_misses", 3)
    if (
        isinstance(view_loss_misses, bool)
        or int(view_loss_misses) != float(view_loss_misses)
        or int(view_loss_misses) < 1
    ):
        raise ValueError("Vision loss threshold must be a positive integer")
    if int(config["vision"].get("observation_pose_lookahead_frames", 0)) < 0:
        raise ValueError("Observation-pose lookahead cannot be negative")
    history_quality = config.get("history_quality", {})
    if float(history_quality.get("maximum_interval_factor", 0.0)) <= 1.0:
        raise ValueError("History maximum interval factor must be greater than one visual period")
    maximum_missed = int(history_quality.get("maximum_missed_detections", -1))
    if maximum_missed < 0 or maximum_missed > history:
        raise ValueError("History imputed-frame audit limit must be inside [0, history_length]")
    if float(history_quality.get("maximum_curvature_m_inv", 0.0)) <= 0.0:
        raise ValueError("History curvature limit must be positive")
    minimum_history_curvature = float(
        history_quality.get("minimum_curvature_m_inv", 0.0)
    )
    if not 0.0 <= minimum_history_curvature < float(
        history_quality["maximum_curvature_m_inv"]
    ):
        raise ValueError("History minimum curvature must be below its maximum")
    minimum_history_turn = float(
        history_quality.get("minimum_five_frame_turn_deg", 0.0)
    )
    maximum_history_turn = float(
        history_quality.get("maximum_five_frame_turn_deg", 0.0)
    )
    if maximum_history_turn <= 0.0:
        raise ValueError("History five-frame turn limit must be positive")
    if not 0.0 <= minimum_history_turn < maximum_history_turn:
        raise ValueError("History minimum five-frame turn must be below its maximum")
    minimum_history_plane_ratio = float(
        history_quality.get("minimum_five_frame_plane_ratio", 0.0)
    )
    if not 0.0 <= minimum_history_plane_ratio < 1.0:
        raise ValueError("History minimum five-frame plane ratio must be inside [0, 1)")
    if float(history_quality.get("minimum_speed_for_curvature_mps", -1.0)) < 0.0:
        raise ValueError("History minimum curvature speed cannot be negative")
    trajectory = config["trajectory"]
    if str(trajectory.get("timestamp_mode", "jittered_period_v1")) not in {
        "jittered_period_v1",
        "fixed_period_v1",
    }:
        raise ValueError("Unsupported trajectory timestamp mode")
    if int(trajectory["curvature_history_frames"]) != history:
        raise ValueError("Trajectory curvature validation must use the same five-frame history")
    low_curvature = [float(item) for item in trajectory["curvature_low_bounds_m_inv"]]
    high_curvature = [float(item) for item in trajectory["curvature_high_bounds_m_inv"]]
    if (
        len(low_curvature) != 2
        or len(high_curvature) != 2
        or low_curvature[0] <= 0.0
        or low_curvature[1] <= low_curvature[0]
        or high_curvature[1] <= high_curvature[0]
        or high_curvature[0] <= low_curvature[1]
    ):
        raise ValueError("Trajectory low/high curvature bands must be positive and separated")
    if float(trajectory["maximum_curvature_m_inv"]) < high_curvature[1]:
        raise ValueError("Maximum trajectory curvature must cover the sampled high band")
    minimum_five_frame_turn = float(
        trajectory.get("minimum_five_frame_turn_deg", 0.0)
    )
    maximum_five_frame_turn = float(
        trajectory.get("maximum_five_frame_turn_deg", 0.0)
    )
    if maximum_five_frame_turn <= 0.0:
        raise ValueError("Trajectory five-frame turn limit must be positive")
    if (
        minimum_five_frame_turn < 0.0
        or minimum_five_frame_turn >= maximum_five_frame_turn
    ):
        raise ValueError(
            "Trajectory minimum five-frame turn must be nonnegative and below the maximum"
        )
    minimum_five_frame_plane_ratio = float(
        trajectory.get("minimum_five_frame_plane_ratio", 0.0)
    )
    if not 0.0 <= minimum_five_frame_plane_ratio < 1.0:
        raise ValueError("Trajectory minimum five-frame plane ratio must be inside [0, 1)")
    five_frame_range = [
        float(item) for item in trajectory["five_frame_curvature_range_bounds_m_inv"]
    ]
    if (
        len(five_frame_range) != 2
        or five_frame_range[0] <= 0.0
        or five_frame_range[1] <= five_frame_range[0]
    ):
        raise ValueError("Five-frame curvature change bounds must be increasing and nonzero")
    plane_variation = [
        float(item) for item in trajectory["local_plane_total_variation_bounds_deg"]
    ]
    if (
        len(plane_variation) != 2
        or plane_variation[0] <= 0.0
        or plane_variation[1] <= plane_variation[0]
        or float(trajectory["local_plane_max_step_deg"]) <= 0.0
    ):
        raise ValueError("Local-plane variation must be nonzero, bounded and continuous")
    if int(trajectory["generation_max_attempts"]) <= 0:
        raise ValueError("Trajectory generation_max_attempts must be positive")
    if (
        float(config["plane"]["maximum_normal_step_deg"]) <= 0.0
        or float(config["plane"]["maximum_reference_deviation_deg"]) <= 0.0
    ):
        raise ValueError("Local-plane normal continuity limits must be positive")
    orientation_branches = [
        int(item) for item in config["candidate"]["orientation_branches"]
    ]
    if orientation_branches != [1]:
        raise ValueError(
            "Cage supervision requires exactly one orientation branch: "
            "z equals the stabilized positive motion-plane normal"
        )
    if int(config["ppo"]["state_dim"]) != 18 or int(config["ppo"]["action_dim"]) != 6:
        raise ValueError("PPO2 contract is fixed to an 18-D state and a 6-D action")
    jailer = config.get("jailer")
    if jailer is not None:
        if float(jailer.get("control_rate_hz", 0.0)) != 100.0:
            raise ValueError("JAiLeR curriculum control is fixed at exactly 100 Hz")
        if int(jailer.get("logical_slots", 0)) <= 0:
            raise ValueError("JAiLeR curriculum requires a positive logical slot count")
        jailer_ppo = jailer.get("ppo", {})
        if (
            int(jailer_ppo.get("state_dim", 0)) != 18
            or int(jailer_ppo.get("action_dim", 0)) != 6
        ):
            raise ValueError("JAiLeR state/action contract is fixed at 18/6")
        residual_hidden = [
            int(value)
            for value in jailer_ppo.get("pose_residual_hidden_dims", [])
        ]
        if not residual_hidden or any(value <= 0 for value in residual_hidden):
            raise ValueError(
                "JAiLeR pose residual requires positive hidden dimensions"
            )
        if float(jailer_ppo.get("pose_residual_learning_rate", 0.0)) <= 0.0:
            raise ValueError(
                "JAiLeR pose residual learning rate must be positive"
            )
        action_composition_mode = str(
            jailer_ppo.get(
                "action_composition_mode",
                "combined_mean_then_global_stage_scale_v1",
            )
        )
        if action_composition_mode not in {
            "combined_mean_then_global_stage_scale_v1",
            "frozen_position_plus_error_scaled_pose_residual_v2",
            "joint_pose_actor_dynamic_speed_with_position_reference_v1",
            "joint_pose_actor_dynamic_speed_ppo_only_v2",
        }:
            raise ValueError("Unsupported JAiLeR action-composition mode")
        boundary_mode = str(
            jailer.get(
                "rollout_boundary_mode", "truncate_and_bootstrap_v2"
            )
        )
        if boundary_mode not in {
            "truncate_and_bootstrap_v2",
            "stop_spawn_then_drain_active_tasks_v1",
            "single_task_per_slot_until_terminal_v1",
        }:
            raise ValueError("Unsupported JAiLeR rollout-boundary mode")
        termination_mode = str(
            jailer.get("termination_semantics_mode", "legacy_v1")
        )
        if termination_mode not in {
            "legacy_v1",
            "time_limit_bootstrap_failure_compensation_v1",
        }:
            raise ValueError("Unsupported JAiLeR termination-semantics mode")
        quality_cost_scale = float(
            jailer.get("reward_step_quality_cost_scale", 1.0)
        )
        if not 0.0 < quality_cost_scale <= 1.0:
            raise ValueError(
                "JAiLeR step-quality cost scale must be inside (0,1]"
            )
        if (
            termination_mode
            == "time_limit_bootstrap_failure_compensation_v1"
            and float(jailer.get("collision_penalty", 0.0)) <= 0.0
        ):
            raise ValueError(
                "Compensated terminal failures require a positive fixed penalty"
            )
        if action_composition_mode == (
            "frozen_position_plus_error_scaled_pose_residual_v2"
        ):
            residual_errors = [
                float(value)
                for value in jailer_ppo.get(
                    "pose_residual_gain_orientation_error_deg", []
                )
            ]
            residual_gains = [
                float(value)
                for value in jailer_ppo.get("pose_residual_gain_values", [])
            ]
            if (
                len(residual_errors) < 2
                or len(residual_errors) != len(residual_gains)
                or residual_errors[0] < 0.0
                or any(
                    second <= first
                    for first, second in zip(
                        residual_errors[:-1], residual_errors[1:]
                    )
                )
                or residual_gains[0] < 0.0
                or residual_gains[-1] > 1.0
                or any(
                    second < first
                    for first, second in zip(
                        residual_gains[:-1], residual_gains[1:]
                    )
                )
                or not -5.0
                <= float(
                    jailer_ppo.get("pose_residual_initial_log_std", -6.0)
                )
                <= 1.0
            ):
                raise ValueError(
                    "JAiLeR V9 needs a monotone physical-error residual-gain schedule"
                )
        joint_action_modes = {
            "joint_pose_actor_dynamic_speed_with_position_reference_v1",
            "joint_pose_actor_dynamic_speed_ppo_only_v2",
        }
        if action_composition_mode in joint_action_modes:
            gain_errors = [
                float(value)
                for value in jailer_ppo.get(
                    "joint_action_gain_error_ratio", []
                )
            ]
            gain_values = [
                float(value)
                for value in jailer_ppo.get("joint_action_gain_values", [])
            ]
            if (
                len(gain_errors) < 2
                or len(gain_errors) != len(gain_values)
                or gain_errors[0] < 0.0
                or any(
                    second <= first
                    for first, second in zip(
                        gain_errors[:-1], gain_errors[1:]
                    )
                )
                or gain_values[0] <= 0.0
                or gain_values[-1] > 1.0
                or any(
                    second < first
                    for first, second in zip(
                        gain_values[:-1], gain_values[1:]
                    )
                )
                or float(jailer_ppo.get("joint_actor_learning_rate", 0.0))
                <= 0.0
                or float(jailer_ppo.get("joint_critic_learning_rate", 0.0))
                <= 0.0
            ):
                raise ValueError("Invalid joint JAiLeR policy settings")
            if (
                action_composition_mode
                == "joint_pose_actor_dynamic_speed_ppo_only_v2"
                and (
                    int(jailer_ppo.get("actor_update_epochs", 0)) <= 0
                    or int(jailer_ppo.get("critic_update_epochs", 0)) <= 0
                )
            ):
                raise ValueError(
                    "Joint PPO-only JAiLeR requires positive independent "
                    "Actor and Critic update epochs"
                )
            if (
                str(jailer.get("reward_weight_mode", ""))
                not in {
                    "dynamic_normalized_pose_error_v1",
                    "dynamic_normalized_pose_error_achieved_floor_v2",
                }
                or not 0.0
                < float(jailer.get("dynamic_reward_min_weight", 0.0))
                < 0.5
                or float(jailer.get("dynamic_reward_urgency_power", 0.0))
                <= 0.0
            ):
                raise ValueError("Invalid dynamic joint-pose reward settings")
        if action_composition_mode == (
            "joint_pose_actor_dynamic_speed_with_position_reference_v1"
        ):
            reference_gate = [
                float(value)
                for value in jailer_ppo.get(
                    "position_reference_gate_error_ratio", []
                )
            ]
            if (
                len(reference_gate) != 2
                or reference_gate[0] < 0.0
                or reference_gate[1] <= reference_gate[0]
                or not 0.0
                <= float(jailer_ppo.get("position_rehearsal_fraction", -1.0))
                <= 1.0
                or float(
                    jailer_ppo.get("position_rehearsal_loss_weight", -1.0)
                )
                < 0.0
            ):
                raise ValueError("Invalid legacy reference-Actor settings")
        velocity_fraction = float(
            jailer.get("action_joint_velocity_limit_fraction", 0.0)
        )
        if velocity_fraction <= 0.0 or velocity_fraction > 1.0:
            raise ValueError("JAiLeR joint-velocity fraction must be inside (0,1]")
        acceleration = [
            float(value)
            for value in jailer.get("joint_acceleration_limit_radps2", [])
        ]
        if len(acceleration) != 6 or any(value <= 0.0 for value in acceleration):
            raise ValueError("JAiLeR requires six positive joint acceleration limits")
        if float(jailer.get("joint_position_margin_rad", 0.0)) <= 0.0:
            raise ValueError("JAiLeR joint-position margin must be positive")
        if (
            int(jailer.get("target_sampling_max_attempts", 0)) <= 0
            or int(jailer.get("target_sampling_pool_retries", 0)) <= 0
            or int(jailer.get("success_curve_interval_updates", 0)) <= 0
            or float(
                jailer.get("success_curve_position_bin_width_m", 0.0)
            ) <= 0.0
            or float(
                jailer.get("success_curve_orientation_bin_width_deg", 0.0)
            ) <= 0.0
            or int(
                jailer.get(
                    "movement_trend_interval_updates",
                    jailer.get("success_curve_interval_updates", 0),
                )
            ) <= 0
            or float(
                jailer.get("movement_trend_position_bin_width_m", 0.005)
            ) <= 0.0
            or float(
                jailer.get("movement_trend_orientation_bin_width_deg", 2.0)
            ) <= 0.0
            or float(jailer.get("legacy_orientation_input_std_deg", 0.0)) <= 0.0
            or not -5.0
            <= float(jailer.get("pose_transition_min_log_std", -6.0))
            <= 1.0
            or float(jailer.get("reward_position_error_gain", 0.0)) <= 0.0
            or float(jailer.get("reward_orientation_marker_gain", 0.0)) <= 0.0
            or float(jailer.get("reward_orientation_progress_weight", -1.0)) < 0.0
            or float(
                jailer.get("reward_orientation_progress_reference_deg", 0.0)
            ) <= 0.0
            or float(jailer.get("reward_position_progress_weight", -1.0)) < 0.0
            or float(
                jailer.get("reward_position_progress_reference_m", 0.0)
            ) <= 0.0
            or float(
                jailer.get("reward_near_target_position_m", 0.0)
            ) <= 0.0
            or float(
                jailer.get("reward_near_target_orientation_deg", 0.0)
            ) <= 0.0
            or float(jailer.get("reward_step_quality_baseline", -1.0)) < 0.0
            or float(jailer.get("reward_joint_acceleration_weight", -1.0)) < 0.0
            or float(jailer.get("success_bonus", 0.0)) <= 0.0
            or float(jailer.get("collision_penalty", 0.0)) <= 0.0
        ):
            raise ValueError("JAiLeR sampling and reward settings are invalid")
        position_quality_mode = str(
            jailer.get(
                "reward_position_quality_mode", "legacy_single_exponential_v1"
            )
        )
        position_progress_mode = str(
            jailer.get(
                "reward_position_progress_mode", "normalized_error_reduction_v1"
            )
        )
        component_progress_mode = str(
            jailer.get(
                "reward_component_progress_after_success_mode",
                "reward_both_directions_v1",
            )
        )
        if (
            position_quality_mode
            not in {
                "legacy_single_exponential_v1",
                "stage_normalized_fine_exp_coarse_inverse_quadratic_v2",
            }
            or position_progress_mode
            not in {
                "normalized_error_reduction_v1",
                "position_quality_delta_v2",
            }
            or component_progress_mode
            not in {
                "reward_both_directions_v1",
                "no_positive_after_component_success_v2",
            }
            or (
                position_quality_mode
                == "stage_normalized_fine_exp_coarse_inverse_quadratic_v2"
                and not 0.0
                < float(jailer.get("reward_position_fine_quality_weight", -1.0))
                < 1.0
            )
            or (
                position_progress_mode == "position_quality_delta_v2"
                and float(
                    jailer.get(
                        "reward_position_progress_quality_reference", 0.0
                    )
                )
                <= 0.0
            )
        ):
            raise ValueError("Invalid JAiLeR multi-scale reward settings")
        pool = jailer.get("initial_pose_pool", {})
        offsets = [float(value) for value in pool.get("joint_offset_max_rad", [])]
        if (
            len(offsets) != 6
            or any(value <= 0.0 for value in offsets)
            or float(pool.get("joint_limit_margin_rad", 0.0)) <= 0.0
            or float(pool.get("minimum_pairwise_joint_distance_rad", -1.0)) < 0.0
            or int(pool.get("maximum_attempts", 0)) <= 0
        ):
            raise ValueError("JAiLeR initial q6 pool settings are invalid")
        curriculum = jailer.get("curriculum", {})
        stages = curriculum.get("stages", [])
        tasks = [str(stage.get("task", "")) for stage in stages]
        pose_only_warm_start = bool(
            curriculum.get("pose_only_actor_warm_start", False)
        )
        if (
            not stages
            or any(task not in {"position", "pose"} for task in tasks)
            or tasks != sorted(tasks, key=lambda item: {"position": 0, "pose": 1}[item])
            or int(curriculum.get("promotion_window_updates", 0)) <= 0
            or (
                pose_only_warm_start
                and any(task != "pose" for task in tasks)
            )
            or (
                not pose_only_warm_start
                and ("position" not in tasks or "pose" not in tasks)
            )
        ):
            raise ValueError(
                "JAiLeR curriculum must contain a valid ordered task sequence"
            )
        entropy_schedule = jailer_ppo.get("entropy_schedule")
        if entropy_schedule is not None:
            schedule_error = None
            stage_names = [str(stage.get("name", "")) for stage in stages]
            if not isinstance(entropy_schedule, dict):
                schedule_error = "must be a mapping"
                schedule_stages = {}
            else:
                schedule_stages = entropy_schedule.get("stages", {})
            if isinstance(entropy_schedule, dict) and set(
                entropy_schedule
            ) != {"mode", "stages"}:
                schedule_error = "accepts only mode/stages values"
            if (
                schedule_error is None
                and action_composition_mode
                != "joint_pose_actor_dynamic_speed_ppo_only_v2"
            ):
                schedule_error = (
                    "is supported only by the independent joint PPO policy"
                )
            elif schedule_error is None and str(
                entropy_schedule.get("mode", "")
            ) != (
                "linear_within_stage_minimum_updates_v1"
            ):
                schedule_error = "has an unsupported mode"
            elif schedule_error is None and not isinstance(
                schedule_stages, dict
            ):
                schedule_error = "must provide a stage mapping"
            elif schedule_error is None and (
                any(not name for name in stage_names)
                or len(stage_names) != len(set(stage_names))
                or set(schedule_stages) != set(stage_names)
            ):
                schedule_error = (
                    "must cover every curriculum stage exactly once"
                )
            elif schedule_error is None:
                for stage_name in stage_names:
                    coefficients = schedule_stages[stage_name]
                    try:
                        coefficient_keys = set(coefficients)
                        start = float(coefficients["start"])
                        end = float(coefficients["end"])
                    except (KeyError, TypeError, ValueError):
                        schedule_error = (
                            "stage '{}' must define numeric start/end values"
                        ).format(stage_name)
                        break
                    if coefficient_keys != {"start", "end"}:
                        schedule_error = (
                            "stage '{}' accepts only start/end values"
                        ).format(stage_name)
                        break
                    if not 0.0 <= end <= start <= 0.1:
                        schedule_error = (
                            "stage '{}' must satisfy "
                            "0 <= end <= start <= 0.1"
                        ).format(stage_name)
                        break
            if schedule_error is not None:
                raise ValueError(
                    "JAiLeR entropy-coefficient schedule {}".format(
                        schedule_error
                    )
                )
        for stage in stages:
            position = [float(value) for value in stage.get("position_error_range_m", [])]
            orientation = [
                float(value) for value in stage.get("orientation_error_range_deg", [])
            ]
            target_offset = [
                float(value) for value in stage.get("target_joint_offset_max_rad", [])
            ]
            orientation_observation_mode = str(
                stage.get("orientation_observation_mode", "observed")
            )
            entry_transition = str(
                stage.get("entry_transition", "continuous")
            )
            if (
                len(position) != 2
                or position[0] < 0.0
                or position[1] <= position[0]
                or len(orientation) != 2
                or orientation[0] < 0.0
                or orientation[1] < orientation[0]
                or len(target_offset) != 6
                or any(value <= 0.0 for value in target_offset)
                or int(stage.get("rollout_steps_per_slot", 0)) <= 0
                or int(stage.get("minimum_updates", 0)) <= 0
                or not 0.0
                <= float(stage.get("reward_orientation_quality_weight", -1.0))
                < 1.0
                or (
                    str(stage.get("task")) == "position"
                    and float(stage.get("reward_orientation_quality_weight", 0.0))
                    != 0.0
                )
                or not 0.0 < float(stage.get("action_velocity_scale", 0.0)) <= 1.0
                or float(stage.get("success_position_m", 0.0)) <= 0.0
                or float(stage.get("promotion_position_mean_m", 0.0)) <= 0.0
                or float(stage.get("promotion_position_mean_m", 0.0))
                < float(stage.get("success_position_m", 0.0))
                or float(stage.get("success_orientation_deg", 0.0)) <= 0.0
                or float(stage.get("promotion_orientation_mean_deg", 0.0)) <= 0.0
                or float(stage.get("success_tcp_speed_mps", 0.0)) <= 0.0
                or float(stage.get("success_tcp_angular_speed_radps", 0.0)) <= 0.0
                or not 0.0 <= float(stage.get("promotion_success_rate", -1.0)) <= 1.0
                or orientation_observation_mode
                not in {"masked_zero", "observed"}
                or entry_transition
                not in {
                    "continuous",
                    "masked_to_observed_new_critic",
                    "preserve_critic_reset_adam_align",
                    "new_critic_reset_adam_align",
                }
                or (
                    entry_transition == "preserve_critic_reset_adam_align"
                    and str(
                        stage.get(
                            "entry_critic_alignment_mode", "fixed_updates_v1"
                        )
                    ) == "fixed_updates_v1"
                    and int(stage.get("entry_critic_alignment_updates", 0)) <= 0
                )
                or (
                    action_composition_mode in joint_action_modes
                    and (
                        not -5.0
                        <= float(stage.get("joint_log_std_floor", -6.0))
                        <= 1.0
                        or not -5.0
                        <= float(stage.get("joint_log_std_entry_min", -6.0))
                        <= 1.0
                        or float(stage.get("joint_log_std_entry_min", -6.0))
                        < float(stage.get("joint_log_std_floor", -6.0))
                    )
                )
                or (
                    action_composition_mode
                    == "joint_pose_actor_dynamic_speed_with_position_reference_v1"
                    and float(stage.get("position_reference_loss_weight", -1.0))
                    < 0.0
                )
                ):
                raise ValueError(
                    "Invalid JAiLeR curriculum stage '{}'".format(
                        stage.get("name", "unnamed")
                    )
                )
            alignment_mode = str(
                stage.get("entry_critic_alignment_mode", "fixed_updates_v1")
            )
            if alignment_mode not in {
                "fixed_updates_v1",
                "pre_update_ev_gate_v1",
            }:
                raise ValueError("Invalid Critic alignment mode")
            if alignment_mode == "pre_update_ev_gate_v1" and (
                not 0.0
                <= float(stage.get("entry_critic_alignment_ev_threshold", -1.0))
                <= 1.0
                or int(
                    stage.get(
                        "entry_critic_alignment_consecutive_rollouts", 0
                    )
                )
                <= 0
                or int(
                    stage.get("entry_critic_alignment_max_rollouts", 0)
                )
                <= 0
            ):
                raise ValueError("Invalid adaptive Critic alignment gate")
        adaptive_exploration = jailer_ppo.get("adaptive_exploration")
        if adaptive_exploration is not None:
            if (
                not isinstance(adaptive_exploration, dict)
                or not isinstance(
                    adaptive_exploration.get("enabled", False), bool
                )
                or int(adaptive_exploration.get("patience_updates", 0)) <= 0
                or float(
                    adaptive_exploration.get(
                        "minimum_progress_improvement", -1.0
                    )
                )
                < 0.0
                or not 0.0
                <= float(
                    adaptive_exploration.get(
                        "plateau_regression_tolerance", 0.0
                    )
                )
                <= 0.10
                or float(
                    adaptive_exploration.get("entropy_multiplier", 0.0)
                )
                < 1.0
                or int(adaptive_exploration.get("boost_updates", 0)) <= 0
                or int(
                    adaptive_exploration.get(
                        "maximum_boosts_per_stage", 0
                    )
                )
                <= 0
                or not 0.0
                <= float(
                    adaptive_exploration.get("critic_ev_threshold", -1.0)
                )
                <= 1.0
            ):
                raise ValueError("Invalid adaptive exploration settings")
    constrained_ik = config.get("constrained_differential_ik")
    if constrained_ik is not None:
        if float(constrained_ik.get("control_rate_hz", 0.0)) != 100.0:
            raise ValueError("Constrained differential IK is fixed at 100 Hz")
        constrained_acceleration = [
            float(value)
            for value in constrained_ik.get(
                "joint_acceleration_limit_radps2", []
            )
        ]
        positive_keys = (
            "position_gain_s_inv",
            "orientation_gain_s_inv",
            "maximum_tcp_linear_speed_mps",
            "maximum_tcp_angular_speed_radps",
            "position_task_weight",
            "orientation_task_weight",
            "regularization",
            "joint_position_margin_rad",
            "success_position_m",
            "success_orientation_deg",
            "success_tcp_speed_mps",
            "success_tcp_angular_speed_radps",
        )
        if (
            len(constrained_acceleration) != 6
            or any(value <= 0.0 for value in constrained_acceleration)
            or any(float(constrained_ik.get(key, 0.0)) <= 0.0 for key in positive_keys)
            or not 0.0
            < float(constrained_ik.get("joint_velocity_limit_fraction", 0.0))
            <= 1.0
            or int(constrained_ik.get("qp_max_iterations", 0)) <= 0
            or float(constrained_ik.get("settling_time_s", -1.0)) < 0.0
        ):
            raise ValueError("Constrained differential IK settings are invalid")
    if not isinstance(
        config["robot"].get("compensate_bias_forces_in_position_control"), bool
    ):
        raise ValueError("Robot position-control bias compensation must be boolean")
    ppo = config["ppo"]
    if (
        float(ppo["state_normalization_clip"]) <= 0.0
        or float(ppo["state_normalization_epsilon"]) <= 0.0
    ):
        raise ValueError("PPO state normalization clip and epsilon must be positive")
    if (
        float(ppo["target_kl"]) <= 0.0
        or float(ppo["kl_stop_multiplier"]) <= 1.0
    ):
        raise ValueError("PPO KL target must be positive and stop multiplier above one")
    saturation_threshold = float(ppo["action_saturation_threshold"])
    if saturation_threshold <= 0.0 or saturation_threshold >= 1.0:
        raise ValueError("PPO action saturation threshold must be inside (0,1)")
    reward_mode = str(ppo.get("reward_mode", "legacy_squared"))
    normalized_reward_modes = {
        "normalized_cage_tracking_v1",
        "normalized_cage_tracking_v2",
    }
    if reward_mode not in {"legacy_squared", *normalized_reward_modes}:
        raise ValueError("Unsupported PPO reward_mode '{}'".format(reward_mode))
    if reward_mode in normalized_reward_modes:
        quality_weights = [
            float(ppo.get("reward_position_quality_weight", 0.0)),
            float(ppo.get("reward_orientation_quality_weight", 0.0)),
        ]
        if (
            any(item <= 0.0 for item in quality_weights)
            or abs(sum(quality_weights) - 1.0) > 1e-6
            or float(ppo.get("reward_position_reference_m", 0.0)) <= 0.0
            or float(ppo.get("reward_orientation_reference_deg", 0.0)) <= 0.0
            or float(ppo.get("reward_progress_weight", -1.0)) < 0.0
            or float(ppo.get("reward_success_bonus", -1.0)) < 0.0
            or float(ppo.get("reward_joint_limit_weight", -1.0)) < 0.0
            or float(ppo.get("reward_velocity_weight", -1.0)) < 0.0
            or float(ppo.get("reward_action_weight", -1.0)) < 0.0
            or float(ppo.get("visual_escape_penalty", -1.0)) < 0.0
        ):
            raise ValueError(
                "Normalized Cage reward requires positive unit-sum pose weights, "
                "positive reference scales and nonnegative shaping coefficients"
            )
    expert = config.get("expert_training", {})
    if bool(expert.get("enabled", False)):
        positive_integer_keys = (
            "demonstration_steps",
            "behavior_cloning_epochs",
            "dagger_steps_per_round",
            "dagger_supervised_epochs",
            "supervised_batch_size",
            "supervised_early_stopping_patience",
            "critic_warmup_steps",
            "critic_warmup_epochs",
            "critic_warmup_batch_size",
        )
        if any(int(expert.get(key, 0)) <= 0 for key in positive_integer_keys):
            raise ValueError("Expert/DAgger sample, epoch and batch counts must be positive")
        weights = [float(item) for item in expert.get("dagger_expert_weights", [])]
        if (
            not weights
            or any(item < 0.0 or item > 1.0 for item in weights)
            or any(weights[index + 1] > weights[index] for index in range(len(weights) - 1))
            or weights[-1] != 0.0
        ):
            raise ValueError(
                "DAgger expert weights must decrease inside [0,1] and end at zero"
            )
        if len(expert.get("joint_loss_weights", [])) != 6 or any(
            float(item) <= 0.0 for item in expert.get("joint_loss_weights", [])
        ):
            raise ValueError("Expert supervised loss requires six positive joint weights")
        if not 0.0 < float(expert.get("validation_fraction", 0.0)) < 0.5:
            raise ValueError("Expert validation fraction must be inside (0,0.5)")
        jacobian = expert.get("jacobian", {})
        positive_jacobian_keys = (
            "position_gain_s_inv",
            "orientation_gain_s_inv",
            "maximum_tcp_linear_speed_mps",
            "maximum_tcp_angular_speed_radps",
            "base_damping",
            "singularity_threshold",
            "singularity_damping_gain",
            "joint_velocity_feedback_gain",
            "joint_velocity_limit_fraction",
        )
        if any(float(jacobian.get(key, 0.0)) <= 0.0 for key in positive_jacobian_keys):
            raise ValueError("Damped-Jacobian expert parameters must be positive")
        velocity_fraction = float(jacobian["joint_velocity_limit_fraction"])
        if velocity_fraction > 1.0:
            raise ValueError("Jacobian expert velocity fraction cannot exceed one")
        random_demo = expert.get("random_demonstration", {})
        if (
            int(random_demo.get("episode_horizon_steps", 0)) <= 0
            or int(random_demo.get("maximum_reset_attempts", 0)) <= 0
        ):
            raise ValueError("Random expert episode horizon and reset attempts must be positive")
        offsets = random_demo.get("joint_offset_max_rad", [])
        if len(offsets) != 6 or any(float(item) <= 0.0 for item in offsets):
            raise ValueError("Random expert reset needs six positive joint offsets")
        if (
            float(random_demo.get("joint_velocity_std_radps", -1.0)) < 0.0
            or float(random_demo.get("joint_limit_margin_rad", 0.0)) <= 0.0
            or float(random_demo.get("success_position_m", 0.0)) <= 0.0
            or float(random_demo.get("success_orientation_deg", 0.0)) <= 0.0
        ):
            raise ValueError("Random expert velocity, margin and success thresholds are invalid")
        for key in ("position_error_range_m", "orientation_error_range_deg"):
            bounds = [float(item) for item in random_demo.get(key, [])]
            if len(bounds) != 2 or bounds[0] < 0.0 or bounds[1] <= bounds[0]:
                raise ValueError("Random expert {} must be an increasing range".format(key))
    residual = config.get("residual_training", {})
    if bool(residual.get("enabled", False)):
        if reward_mode != "normalized_cage_tracking_v2":
            raise ValueError(
                "Residual PPO production training requires normalized_cage_tracking_v2"
            )
        positive_integer_keys = (
            "actor_update_epochs",
            "critic_update_epochs",
            "critic_alignment_steps",
            "critic_alignment_epochs",
            "critic_alignment_batch_size",
            "checkpoint_interval_updates",
            "validation_interval_updates",
            "validation_episodes",
        )
        if any(int(residual.get(key, 0)) <= 0 for key in positive_integer_keys):
            raise ValueError("Residual PPO counts and intervals must be positive")
        if (
            float(residual.get("actor_learning_rate", 0.0)) <= 0.0
            or float(residual.get("critic_learning_rate", 0.0)) <= 0.0
            or not -5.0 <= float(residual.get("initial_log_std", -99.0)) <= 1.0
        ):
            raise ValueError("Residual PPO learning rates or initial log_std are invalid")
        hidden_dims = [int(item) for item in residual.get("hidden_dims", [])]
        if not hidden_dims or any(item <= 0 for item in hidden_dims):
            raise ValueError("Residual Actor hidden dimensions must be positive")
        scale = residual.get("action_scale", 0.0)
        scales = scale if isinstance(scale, list) else [scale]
        if len(scales) not in (1, 6) or any(
            float(item) <= 0.0 or float(item) > 1.0 for item in scales
        ):
            raise ValueError("Residual action scale must contain one or six values in (0,1]")
        if (
            int(residual["validation_interval_updates"])
            % int(residual["checkpoint_interval_updates"])
            != 0
        ):
            raise ValueError(
                "Residual validation interval must be a multiple of checkpoint interval"
            )
    cage_network = config["cage_network"]
    position_dim = int(cage_network["per_frame_position_dim"])
    speed_dim = int(cage_network["per_frame_speed_dim"])
    if position_dim != 3 or speed_dim != 1:
        raise ValueError("Each Cage history frame must contain xyz plus one speed scalar")
    if int(cage_network["epochs"]) <= 0 or int(cage_network["batch_size"]) <= 0:
        raise ValueError("Cage epochs and batch size must be positive")
    if int(cage_network["early_stopping_patience"]) <= 0:
        raise ValueError("Cage early-stopping patience must be positive")
    if float(cage_network["early_stopping_min_delta"]) < 0.0:
        raise ValueError("Cage early-stopping minimum delta cannot be negative")
    if int(cage_network["lr_plateau_patience"]) <= 0:
        raise ValueError("Cage learning-rate plateau patience must be positive")
    decay = float(cage_network["lr_decay_factor"])
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError("Cage learning-rate decay factor must be inside (0,1)")
    minimum_lr = float(cage_network["minimum_learning_rate"])
    initial_lr = float(cage_network["learning_rate"])
    if minimum_lr <= 0.0 or minimum_lr >= initial_lr:
        raise ValueError("Cage minimum learning rate must be positive and below its initial rate")
    if (
        float(cage_network["position_loss_weight"]) <= 0.0
        or float(cage_network["orientation_loss_weight"]) <= 0.0
        or float(cage_network["validation_position_threshold_m"]) <= 0.0
        or float(cage_network["validation_orientation_threshold_deg"]) <= 0.0
    ):
        raise ValueError("Cage loss weights and validation thresholds must be positive")
    cage_training_contract = cage_network.get("training_contract")
    if cage_training_contract == "state_arc_continuous_cage_mlp_v1":
        hidden_dims = [int(item) for item in cage_network.get("hidden_dims", [])]
        if hidden_dims != [128, 128, 64]:
            raise ValueError(
                "The state-arc Cage MLP contract requires hidden_dims [128,128,64]"
            )
        positive_state_arc_fields = (
            "arc_huber_beta",
            "position_error_scale_m",
            "orientation_error_scale_deg",
            "arc_loss_weight",
            "position_loss_weight",
            "orientation_loss_weight",
        )
        if any(
            float(cage_network.get(field, 0.0)) <= 0.0
            for field in positive_state_arc_fields
        ):
            raise ValueError(
                "State-arc Cage loss scales and weights must all be positive"
            )
        if float(cage_network["arc_huber_beta"]) > 1.0:
            raise ValueError("State-arc normalized Huber beta cannot exceed one")
        if float(cage_network.get("weight_decay", 0.0)) < 0.0:
            raise ValueError("State-arc Cage weight decay cannot be negative")
    if config["system"].get("object_model") != "point_mass":
        raise ValueError("The current system contract requires the point_mass object model")
    if config["system"].get("coordinate_handedness") != "right":
        raise ValueError("Base, link6, Cage, TCP and camera 3-D frames must all be right handed")
    if (
        config["system"].get("camera_3d_convention")
        != "mujoco_right_handed_camera_looks_minus_z"
    ):
        raise ValueError("Unsupported camera 3-D axis convention")
    event_delta = config["robot"].get("normalized_delta_max_rad_per_vision_update")
    if event_delta is None or len(event_delta) != 6 or any(float(item) <= 0.0 for item in event_delta):
        raise ValueError("Visual-event PPO requires six positive per-update joint delta limits")
    region = config["gripper"]["internal_region_endpoint_frame"]
    dataset_contract = str(
        config["labeling"].get("dataset_contract", "legacy_history_pose_v1")
    )
    state_arc_contract = (
        "current_base_position_velocity_acceleration_plane_normal_signed_curvature_"
        "to_best_normalized_minor_arc_v1"
    )
    if dataset_contract not in {"legacy_history_pose_v1", state_arc_contract}:
        raise ValueError("Unsupported Cage label dataset contract")
    if dataset_contract == state_arc_contract:
        plane_history = int(
            config["labeling"].get("minimum_local_plane_history_frames", 0)
        )
        if plane_history < 3:
            raise ValueError(
                "State-arc labels require at least three local-plane history frames"
            )
        if config.get("vision", {}).get("require_rendered_cage_training_inputs", False):
            raise ValueError(
                "State-arc geometry labels cannot require rendered visual inputs"
            )
    speed_bounds = config["candidate"]["speed_radius_speed_bounds_mps"]
    if len(speed_bounds) != 2 or float(speed_bounds[0]) < 0.0 or float(speed_bounds[1]) <= float(speed_bounds[0]):
        raise ValueError("Speed-to-radius mapping requires increasing nonnegative speed bounds")
    if float(config["candidate"]["radius_max_m"]) <= float(config["candidate"]["radius_min_m"]):
        raise ValueError("Candidate radius bounds must be increasing")
    minimum_visible_standoff = (
        float(region["mouth_x"])
        + float(config["vision"]["target_radius_m"])
        + float(config["candidate"]["minimum_mouth_to_object_surface_clearance_m"])
    )
    if float(config["candidate"]["radius_min_m"]) < minimum_visible_standoff:
        raise ValueError("Minimum follow standoff does not keep the full target clear of the mouth")
    if config["est"].get("dynamics_backend") != "mujoco":
        raise ValueError("EST expansion must use the MuJoCo dynamics backend")
    if int(config["est"].get("repeat_count", 0)) != 3:
        raise ValueError("The label contract requires exactly three EST repeats")
    disturbance_mode = str(
        config["est"].get("disturbance_direction_sampling_mode", "uniform_box_v1")
    )
    if disturbance_mode not in {"uniform_box_v1", "motion_arc_mixture_v1"}:
        raise ValueError("Unsupported EST disturbance direction sampling mode")
    if disturbance_mode == "motion_arc_mixture_v1":
        probability = float(config["est"].get("arc_direction_probability", -1.0))
        if not 0.0 < probability < 1.0:
            raise ValueError("EST arc-direction probability must be inside (0,1)")
        if float(config["est"].get("arc_direction_out_of_plane_std", -1.0)) < 0.0:
            raise ValueError("EST out-of-plane direction spread cannot be negative")
        magnitude = config["est"].get("arc_direction_force_magnitude_fraction", [])
        if (
            len(magnitude) != 2
            or float(magnitude[0]) < 0.0
            or float(magnitude[1]) <= float(magnitude[0])
            or float(magnitude[1]) > 1.0
        ):
            raise ValueError(
                "EST arc-direction force fractions must be increasing inside [0,1]"
            )
    local_expansion = config["est"].get("local_expansion_bounds_endpoint_frame")
    if local_expansion is not None:
        local_min = np.asarray(local_expansion.get("min_m", []), dtype=float)
        local_max = np.asarray(local_expansion.get("max_m", []), dtype=float)
        if (
            local_min.shape != (3,)
            or local_max.shape != (3,)
            or not np.all(np.isfinite(local_min))
            or not np.all(np.isfinite(local_max))
            or np.any(local_max <= local_min)
        ):
            raise ValueError("EST local endpoint-frame expansion bounds are invalid")
    vision = config.get("vision", {})
    if vision.get("enabled", False):
        if int(vision["image_width"]) <= 0 or int(vision["image_height"]) <= 0:
            raise ValueError("Eye-on-hand image dimensions must be positive")
        if float(vision["depth_foreground_band_m"]) <= 0.0:
            raise ValueError("vision.depth_foreground_band_m must be positive")
        if vision["detector_backend"] not in {
            "red_color", "mujoco_segmentation", "ultralytics"
        }:
            raise ValueError("Unsupported eye-on-hand detector backend")
        if vision["detector_backend"] == "ultralytics" and not vision["coco_weights_path"]:
            raise ValueError("Ultralytics vision requires a local coco_weights_path")
        for probability_name in (
            "depth_pixel_dropout_probability", "frame_dropout_probability"
        ):
            probability = float(vision[probability_name])
            if probability < 0.0 or probability >= 1.0:
                raise ValueError("{} must be in [0,1)".format(probability_name))
    backend = config["labeling"]["feasibility_backend"]
    if config["system"].get("production_mode", False) and backend == "workspace":
        raise ValueError("Production label generation requires a MoveIt or MuJoCo feasibility backend")
