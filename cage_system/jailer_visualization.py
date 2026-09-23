"""Deterministic MuJoCo video audit for a trained JAiLeR checkpoint."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from cage_system.learning.jailer_policy import JailerPPO2Trainer
from cage_system.learning.train_jailer import (
    JAILER_CHECKPOINT_FORMAT,
    JAILER_POLICY_CONTRACT,
    JAILER_STATE_CONTRACT,
)
from cage_system.sim.jailer_env import JailerLogicalVectorEnv, load_initial_joint_pool
from cage_system.sim.ur5e_control import UR5eControlModel
from cage_system.system_visualization import _add_capsule


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _add_pose_axes(scene, mujoco, pose6, length=0.075):
    origin = np.asarray(pose6[:3], dtype=float)
    rotation = Rotation.from_rotvec(np.asarray(pose6[3:], dtype=float)).as_matrix()
    for axis, color in zip(
        rotation.T,
        ((1.0, 0.1, 0.1, 1.0), (0.1, 1.0, 0.1, 1.0), (0.1, 0.3, 1.0, 1.0)),
    ):
        _add_capsule(scene, mujoco, origin, origin + length * axis, 0.004, color)


def render_jailer_video(
    config,
    checkpoint_path,
    model_path,
    tcp_site_name,
    initial_pool_path,
    stage_name,
    slot_id,
    seed,
    control_steps,
    video_fps,
    output_path,
    device="auto",
):
    """Run one fixed target at 100 Hz and save overview plus wrist-camera video."""
    try:
        import cv2
        import mujoco
    except ImportError as error:
        raise RuntimeError("OpenCV and MuJoCo are required for JAiLeR video") from error

    checkpoint = _torch_load(checkpoint_path)
    if (
        checkpoint.get("format") != JAILER_CHECKPOINT_FORMAT
        or checkpoint.get("state_contract") != JAILER_STATE_CONTRACT
        or checkpoint.get("policy_contract") != JAILER_POLICY_CONTRACT
    ):
        raise ValueError(
            "The video command currently expects a legacy V8 full-observation JAiLeR checkpoint"
        )
    settings = copy.deepcopy(config["jailer"])
    # Reproduce the checkpoint's own curriculum instead of trusting a later
    # YAML edit to retain the old O5 physical action scale.
    saved_curriculum = checkpoint.get("curriculum_definition")
    if not saved_curriculum:
        raise ValueError("JAiLeR checkpoint does not contain its curriculum definition")
    settings["curriculum"] = copy.deepcopy(saved_curriculum)
    stages = {
        str(stage["name"]): copy.deepcopy(stage)
        for stage in settings["curriculum"]["stages"]
    }
    if stage_name not in stages:
        raise ValueError("Checkpoint has no curriculum stage '{}'".format(stage_name))
    stage = stages[stage_name]
    if str(stage["task"]) != "pose":
        raise ValueError("The requested inference-video stage must be a pose stage")
    settings["ppo"].pop("action_composition_mode", None)
    settings["rollout_boundary_mode"] = "truncate_and_bootstrap_v2"

    ur5e = UR5eControlModel(
        model_path,
        config["robot"],
        tcp_site_name=tcp_site_name,
        control_rate_hz=float(settings["control_rate_hz"]),
    )
    if checkpoint.get("ur5e_contract") != ur5e.contract_dict():
        raise ValueError("Checkpoint and requested UR5e/TCP model do not match")
    pool = load_initial_joint_pool(
        initial_pool_path, ur5e, int(settings["logical_slots"])
    )
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, int(seed))
    trainer = JailerPPO2Trainer(settings["ppo"], device=device)
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.state_normalizer.load_state_dict(checkpoint["state_normalizer"])
    trainer.set_curriculum_task("pose")
    trainer.model.eval()

    slot_id = int(slot_id)
    if slot_id < 0 or slot_id >= environment.slot_count:
        raise ValueError("slot-id is outside the initial-pose pool")
    environment.begin_rollout(stage)
    slot = environment.slots[slot_id]
    initial_pose = ur5e.tcp_pose6(slot.data)
    target_pose = slot.target_pose6.copy()

    control_rate = float(settings["control_rate_hz"])
    video_fps = float(video_fps)
    stride = int(round(control_rate / video_fps))
    if stride <= 0 or not np.isclose(control_rate / stride, video_fps):
        raise ValueError("video-fps must divide the 100 Hz control rate exactly")
    control_steps = int(control_steps)
    if control_steps <= 0:
        raise ValueError("control-steps must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (960, 720),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the requested MP4 output")
    renderer = mujoco.Renderer(ur5e.model, height=720, width=960, max_geom=10000)
    traces = []
    first_success_step = None
    success_after_first = []

    def render_frame(step, latest):
        renderer.update_scene(slot.data, camera="overview")
        _add_pose_axes(renderer.scene, mujoco, target_pose)
        current = ur5e.tcp_pose6(slot.data)
        _add_capsule(
            renderer.scene,
            mujoco,
            current[:3],
            target_pose[:3],
            0.0025,
            (1.0, 0.75, 0.1, 0.85),
        )
        rgb = renderer.render().copy()
        renderer.update_scene(slot.data, camera="eye_on_hand")
        eye = renderer.render().copy()
        eye = cv2.resize(eye, (288, 216), interpolation=cv2.INTER_AREA)
        rgb[480:696, 648:936] = eye
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (644, 452), (940, 700), (245, 245, 245), 2)
        cv2.putText(
            frame,
            "eye_on_hand audit",
            (656, 474),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if latest is None:
            status = "initial"
            position_mm = 1000.0 * np.linalg.norm(target_pose[:3] - current[:3])
            angle_deg = np.rad2deg(
                np.linalg.norm(
                    Rotation.from_matrix(
                        Rotation.from_rotvec(target_pose[3:]).as_matrix()
                        @ Rotation.from_rotvec(current[3:]).as_matrix().T
                    ).as_rotvec()
                )
            )
        else:
            status = "success" if latest["task_success"] else latest["terminal_reason"]
            position_mm = 1000.0 * float(latest["position_error_m"])
            angle_deg = float(latest["orientation_error_deg"])
        lines = (
            "V8 best checkpoint | {} | 100 Hz".format(stage_name),
            "t={:.2f}s  step={}  status={}".format(step / control_rate, step, status),
            "TCP error: {:.2f} mm / {:.2f} deg".format(position_mm, angle_deg),
            "target axes: X red, Y green, Z blue",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (24, 36 + 28 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)

    try:
        render_frame(0, None)
        for step in range(1, control_steps + 1):
            raw = environment.observations(stage)[slot_id : slot_id + 1]
            normalized = trainer.normalize_state(raw, update=False)
            state_tensor = torch.as_tensor(
                normalized, dtype=torch.float32, device=trainer.device
            )
            with torch.no_grad():
                decision = trainer.sample_for_environment(
                    state_tensor, raw, stage, deterministic=True
                )
            actions = np.zeros((environment.slot_count, 6), dtype=np.float32)
            actions[slot_id] = decision["executed_action"].cpu().numpy()[0]
            _, reward, _, infos = environment.step(
                actions,
                stage,
                active_mask=np.arange(environment.slot_count) == slot_id,
                recover_terminated=False,
            )
            latest = infos[0]
            if latest["task_success"] and first_success_step is None:
                first_success_step = step
            if first_success_step is not None:
                success_after_first.append(bool(latest["task_success"]))
            traces.append(
                {
                    "step": step,
                    "time_s": step / control_rate,
                    "position_error_m": float(latest["position_error_m"]),
                    "orientation_error_deg": float(latest["orientation_error_deg"]),
                    "tcp_linear_speed_mps": float(latest["tcp_linear_speed_mps"]),
                    "tcp_angular_speed_radps": float(latest["tcp_angular_speed_radps"]),
                    "reward": float(reward[slot_id]),
                    "task_success": bool(latest["task_success"]),
                    "contact": bool(latest["contact"]),
                    "workspace_violation": bool(latest["workspace_violation"]),
                }
            )
            if step % stride == 0:
                render_frame(step, latest)
    finally:
        writer.release()
        renderer.close()

    final = traces[-1]
    report = {
        "video": str(output),
        "checkpoint": str(checkpoint_path),
        "checkpoint_update": int(checkpoint["completed_updates"]),
        "stage": stage_name,
        "seed": int(seed),
        "slot_id": slot_id,
        "control_rate_hz": control_rate,
        "control_steps": control_steps,
        "video_fps": video_fps,
        "duration_s": control_steps / control_rate,
        "initial_tcp_pose6": initial_pose.tolist(),
        "target_tcp_pose6": target_pose.tolist(),
        "first_success_step": first_success_step,
        "first_success_time_s": (
            None if first_success_step is None else first_success_step / control_rate
        ),
        "post_first_success_hold_rate": (
            None
            if not success_after_first
            else float(np.mean(success_after_first))
        ),
        "final_position_error_m": final["position_error_m"],
        "final_orientation_error_deg": final["orientation_error_deg"],
        "trace": traces,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report"] = str(report_path)
    return report
