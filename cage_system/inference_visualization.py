"""Deterministic Cage/PPO inference with a MuJoCo visual audit figure."""

import json
from pathlib import Path

import numpy as np
import torch

from cage_system.geometry import pose6_to_transform
from cage_system.io import load_generated_trajectories
from cage_system.learning.cage_network import FrozenCagePredictor
from cage_system.learning.residual_ppo import load_frozen_control_policy
from cage_system.sim.mujoco_env import MujocoVisionUR5eCageEnv
from cage_system.system_visualization import _add_capsule, _add_trajectory_to_scene


def _add_pose_axes(scene, mujoco, pose6, length=0.045, radius=0.0027):
    """Draw the Cage target frame as non-physical red/green/blue capsules."""
    transform = pose6_to_transform(np.asarray(pose6, dtype=float))
    origin = transform[:3, 3]
    colors = (
        (1.0, 0.12, 0.10, 1.0),
        (0.10, 1.0, 0.18, 1.0),
        (0.15, 0.32, 1.0, 1.0),
    )
    for axis, color in zip(transform[:3, :3].T, colors):
        _add_capsule(
            scene,
            mujoco,
            origin,
            origin + float(length) * axis,
            float(radius),
            color,
        )


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "p95": None, "final": None}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "final": float(values[-1]),
    }


def run_inference_visualization(
    config,
    model_path,
    trajectory_path,
    cage_checkpoint,
    ppo_checkpoint,
    output_path,
    tcp_site="attachment_site",
    device="auto",
    track_id=-1,
    snapshot_count=6,
    max_steps=0,
):
    """Run one deterministic visual episode and save a PNG plus JSON metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    trajectories = load_generated_trajectories(trajectory_path)
    if int(track_id) >= 0:
        trajectories = [
            item for item in trajectories if int(item.track_id) == int(track_id)
        ]
        if not trajectories:
            raise ValueError("Requested track_id is absent from the trajectory archive")

    cage = FrozenCagePredictor(cage_checkpoint, device=device)
    ppo = load_frozen_control_policy(ppo_checkpoint, device=str(cage.device))
    environment = MujocoVisionUR5eCageEnv(
        model_path,
        trajectories,
        cage,
        config,
        tcp_site_name=tcp_site,
        detector_device=str(cage.device),
    )
    renderer = mujoco.Renderer(
        environment.model, height=480, width=640, max_geom=10000
    )
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False

    records = []
    reset_info = {}
    try:
        state, reset_info = environment.reset(seed=int(config["system"]["seed"]))
        while True:
            stale_before_action = (
                environment.last_valid_detection_time is None
                or environment.object_time
                - environment.last_valid_detection_time
                > float(environment.timing["max_observation_age_s"])
            )
            policy_inferred = bool(
                environment.action_ready and not stale_before_action
            )
            if policy_inferred:
                state_tensor = torch.from_numpy(
                    np.asarray(state, dtype=np.float32)
                ).to(ppo.device).unsqueeze(0)
                with torch.inference_mode():
                    action_tensor, _, value_tensor, _ = ppo.sample(
                        state_tensor, deterministic=True
                    )
                action = action_tensor.cpu().numpy()[0]
                critic_value = float(value_tensor.item())
            else:
                # Match the deployed contract: an abnormal/stale visual
                # history does not call PPO and publishes a zero increment.
                action = np.zeros(6, dtype=np.float32)
                critic_value = None
            state, reward, terminated, truncated, info = environment.step(action)

            renderer.update_scene(
                environment.data, camera="overview", scene_option=option
            )
            _add_trajectory_to_scene(
                renderer.scene,
                mujoco,
                environment.object_trajectory,
                environment.object_time,
                radius=0.0042,
                vivid=True,
            )
            _add_pose_axes(
                renderer.scene, mujoco, np.asarray(info["reward_cage_pose6"])
            )
            overview = renderer.render().copy()
            eye_rgb = environment.last_vision_detection.rgb.copy()
            records.append(
                {
                    "time": float(environment.object_time),
                    "overview": overview,
                    "eye_rgb": eye_rgb,
                    "reward": float(reward),
                    "position_error_m": float(info["position_error_m"]),
                    "orientation_error_rad": float(info["orientation_error_rad"]),
                    "action": action.copy(),
                    "value": critic_value,
                    "policy_inferred": policy_inferred,
                    "detected": bool(info["vision_detected"]),
                    "history_valid": bool(info["history_quality_valid"]),
                    "history_reason": str(info["history_quality_reason"]),
                    "collision": bool(info["contact_collision"]),
                    "cage_pose6": np.asarray(info["reward_cage_pose6"]).copy(),
                }
            )
            if terminated or truncated:
                break
            if int(max_steps) > 0 and len(records) >= int(max_steps):
                break
    finally:
        renderer.close()
        environment.close()

    if not records:
        raise RuntimeError("Inference episode produced no transitions")

    snapshot_count = min(max(1, int(snapshot_count)), len(records))
    selected = np.unique(
        np.linspace(0, len(records) - 1, snapshot_count).astype(int)
    )
    columns = min(3, len(selected))
    blocks = int(np.ceil(len(selected) / columns))
    figure = plt.figure(figsize=(6.2 * columns, 6.0 * blocks + 4.6))
    grid = figure.add_gridspec(2 * blocks + 1, columns, height_ratios=[
        value for _ in range(blocks) for value in (1.0, 0.48)
    ] + [0.9])
    for order, record_index in enumerate(selected):
        row_block, column = divmod(order, columns)
        record = records[int(record_index)]
        overview_axis = figure.add_subplot(grid[2 * row_block, column])
        eye_axis = figure.add_subplot(grid[2 * row_block + 1, column])
        overview_axis.imshow(record["overview"])
        overview_axis.set_title(
            "step {} | t={:.2f}s | TCP error={:.1f} mm / {:.1f} deg".format(
                int(record_index),
                record["time"],
                1000.0 * record["position_error_m"],
                np.degrees(record["orientation_error_rad"]),
            )
        )
        eye_axis.imshow(record["eye_rgb"])
        eye_axis.set_title(
            "eye-on-hand RGB | detected={} | history={} | PPO={}".format(
                record["detected"],
                record["history_valid"],
                record["policy_inferred"],
            ),
            fontsize=9,
        )
        overview_axis.axis("off")
        eye_axis.axis("off")
    for order in range(len(selected), blocks * columns):
        row_block, column = divmod(order, columns)
        for row in (2 * row_block, 2 * row_block + 1):
            axis = figure.add_subplot(grid[row, column])
            axis.axis("off")

    times = np.asarray([item["time"] for item in records])
    times = times - times[0]
    position_mm = 1000.0 * np.asarray(
        [item["position_error_m"] for item in records]
    )
    orientation_deg = np.degrees(
        [item["orientation_error_rad"] for item in records]
    )
    rewards = np.asarray([item["reward"] for item in records])
    max_actions = np.asarray(
        [np.max(np.abs(item["action"])) for item in records]
    )
    inference_mask = np.asarray(
        [item["policy_inferred"] for item in records], dtype=bool
    )
    metrics_grid = grid[-1, :].subgridspec(1, 3)
    error_axis = figure.add_subplot(metrics_grid[0, 0])
    reward_axis = figure.add_subplot(metrics_grid[0, 1])
    action_axis = figure.add_subplot(metrics_grid[0, 2])
    error_axis.plot(times, position_mm, label="position [mm]")
    error_axis.plot(times, orientation_deg, label="orientation [deg]")
    error_axis.axhline(
        float(config["ppo"]["success_position_m"]) * 1000.0,
        color="tab:blue",
        linestyle="--",
        alpha=0.45,
    )
    error_axis.axhline(
        float(config["ppo"]["success_orientation_deg"]),
        color="tab:orange",
        linestyle="--",
        alpha=0.45,
    )
    error_axis.legend(fontsize=8)
    error_axis.set_title("Actual TCP vs predicted Cage target")
    reward_axis.plot(times, rewards, color="tab:purple")
    reward_axis.set_title("same-Cage transition reward")
    action_axis.plot(times, max_actions, color="tab:red")
    action_axis.axhline(0.98, color="black", linestyle="--", alpha=0.5)
    action_axis.set_ylim(0.0, 1.05)
    action_axis.set_title("max absolute normalized action")
    for axis in (error_axis, reward_axis, action_axis):
        axis.set_xlabel("episode time [s]")
        axis.grid(alpha=0.24)

    track = environment.object_trajectory
    figure.suptitle(
        "Deterministic eye-on-hand → Cage → PPO inference | track {}\n"
        "physical robot/Cage shown in MuJoCo; RGB axes = predicted Cage target; "
        "full trajectory is audit overlay only".format(int(track.track_id)),
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(destination), dpi=150)
    plt.close(figure)

    position_summary = _summary([item["position_error_m"] for item in records])
    orientation_summary = _summary(
        [np.degrees(item["orientation_error_rad"]) for item in records]
    )
    metrics = {
        "output": str(destination),
        "metrics": str(destination.with_suffix(".json")),
        "track_id": int(track.track_id),
        "steps": int(len(records)),
        "deterministic_policy": True,
        "warmup_frames": int(reset_info.get("warmup_frames", 0)),
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "position_error_m": position_summary,
        "orientation_error_deg": orientation_summary,
        "vision_detection_rate": float(
            np.mean([item["detected"] for item in records])
        ),
        "valid_history_rate": float(
            np.mean([item["history_valid"] for item in records])
        ),
        "policy_inference_rate": float(np.mean(inference_mask)),
        "collision_steps": int(sum(item["collision"] for item in records)),
        "action_saturation_rate_when_inferred": (
            float(np.mean(max_actions[inference_mask] >= 0.98))
            if np.any(inference_mask)
            else 0.0
        ),
        "trajectory_role": "inference_visual_audit_only",
        "future_trajectory_used_by_policy": False,
    }
    with destination.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
