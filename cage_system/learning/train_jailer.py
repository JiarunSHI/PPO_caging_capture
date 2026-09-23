"""Curriculum PPO for a 100 Hz UR5e TCP-reaching feedback policy."""

from collections import deque
import copy
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from cage_system.learning.ppo2 import PPO2Trainer, RolloutBatch
from cage_system.learning.jailer_policy import (
    JOINT_ACTION_MODES,
    JOINT_PPO_ACTION,
    JOINT_REFERENCE_ACTION,
    JOINT_PPO_OPTIMIZER_CONTRACT,
    LEGACY_COMBINED_ACTION,
    SEPARATE_RESIDUAL_ACTION,
    JailerPPO2Trainer,
)
from cage_system.sim.jailer_env import (
    JailerLogicalVectorEnv,
    generate_initial_joint_pool,
    load_initial_joint_pool,
    save_initial_joint_pool,
)
from cage_system.sim.ur5e_control import UR5eControlModel


JAILER_CHECKPOINT_FORMAT = "ur5e_jailer_curriculum_ppo_v1"
JAILER_STATE_CONTRACT = (
    "tcp_target_error_xyz_rotvec_always_observed_q_qdot_18d_v2"
)
JAILER_PHASED_STATE_CONTRACT = (
    "tcp_target_error_xyz_rotvec_masked_position_then_observed_18d_v3"
)
JAILER_LEGACY_POSITION_MASKED_STATE_CONTRACT = (
    "tcp_target_error_xyz_rotvec_q_qdot_18d_v1"
)
JAILER_ACTION_CONTRACT = (
    "normalized_stage_scaled_desired_joint_velocity_at_100hz_v2"
)
JAILER_SEPARATED_ACTION_CONTRACT = (
    "frozen_position_plus_error_scaled_pose_residual_velocity_at_100hz_v3"
)
JAILER_JOINT_ACTION_CONTRACT = (
    "joint_pose_actor_state_error_scaled_velocity_at_100hz_v1"
)
JAILER_LEGACY_ACTION_CONTRACT = (
    "normalized_desired_joint_velocity_at_100hz_v1"
)
JAILER_POLICY_CONTRACT = (
    "frozen_position_backbone_zero_initialized_pose_residual_v1"
)
JAILER_SEPARATED_POLICY_CONTRACT = (
    "frozen_position_action_plus_stochastic_scaled_pose_residual_v2"
)
JAILER_JOINT_POLICY_CONTRACT = (
    "joint_pose_actor_frozen_p6_reference_regularized_v1"
)
JAILER_JOINT_PPO_ONLY_POLICY_CONTRACT = (
    "observed_p6_initialized_joint_pose_actor_ppo_only_v2"
)
JAILER_ROLLOUT_BOUNDARY_CONTRACT = (
    "stage_dynamic_reset_truncation_bootstrap_v2"
)
JAILER_DRAINED_ROLLOUT_BOUNDARY_CONTRACT = (
    "nominal_horizon_stop_spawn_then_drain_every_started_task_v3"
)
JAILER_SINGLE_TASK_ROLLOUT_BOUNDARY_CONTRACT = (
    "one_random_task_per_slot_until_true_terminal_equal_task_weight_v1"
)


def _configured_policy_contract(trainer):
    if trainer.action_composition_mode == JOINT_PPO_ACTION:
        return JAILER_JOINT_PPO_ONLY_POLICY_CONTRACT
    if trainer.action_composition_mode == JOINT_REFERENCE_ACTION:
        return JAILER_JOINT_POLICY_CONTRACT
    if trainer.action_composition_mode == SEPARATE_RESIDUAL_ACTION:
        return JAILER_SEPARATED_POLICY_CONTRACT
    return JAILER_POLICY_CONTRACT


def _configured_state_contract(environment):
    stages = environment.config.get("curriculum", {}).get("stages", [])
    if any(
        str(stage.get("orientation_observation_mode", "observed"))
        == "masked_zero"
        for stage in stages
    ):
        return JAILER_PHASED_STATE_CONTRACT
    return JAILER_STATE_CONTRACT


def _configured_action_contract(trainer):
    if trainer.action_composition_mode in JOINT_ACTION_MODES:
        return JAILER_JOINT_ACTION_CONTRACT
    if trainer.action_composition_mode == SEPARATE_RESIDUAL_ACTION:
        return JAILER_SEPARATED_ACTION_CONTRACT
    return JAILER_ACTION_CONTRACT


def _configured_boundary_contract(environment):
    mode = str(
        environment.config.get(
            "rollout_boundary_mode", "truncate_and_bootstrap_v2"
        )
    )
    if mode == "stop_spawn_then_drain_active_tasks_v1":
        return JAILER_DRAINED_ROLLOUT_BOUNDARY_CONTRACT
    if mode == "single_task_per_slot_until_terminal_v1":
        return JAILER_SINGLE_TASK_ROLLOUT_BOUNDARY_CONTRACT
    if mode == "truncate_and_bootstrap_v2":
        return JAILER_ROLLOUT_BOUNDARY_CONTRACT
    raise ValueError("Unsupported JAiLeR rollout-boundary mode")


def _prepare_success_database(path: Path, reset: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        for candidate in (
            path,
            path.with_name(path.name + "-wal"),
            path.with_name(path.name + "-shm"),
        ):
            if candidate.exists():
                candidate.unlink()
    connection = sqlite3.connect(str(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS successful_reaches (
            update_index INTEGER NOT NULL,
            stage TEXT NOT NULL,
            task TEXT NOT NULL,
            slot_id INTEGER NOT NULL,
            task_sequence INTEGER NOT NULL,
            initial_position_error_m REAL NOT NULL,
            initial_orientation_error_deg REAL NOT NULL,
            steps INTEGER NOT NULL,
            control_time_s REAL NOT NULL,
            PRIMARY KEY (update_index, slot_id, task_sequence)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS movement_trend_bins (
            phase TEXT NOT NULL,
            stage TEXT NOT NULL,
            task TEXT NOT NULL,
            error_axis TEXT NOT NULL,
            bin_index INTEGER NOT NULL,
            bin_width REAL NOT NULL,
            move50_count INTEGER NOT NULL,
            move50_sum_m REAL NOT NULL,
            move50_sumsq_m2 REAL NOT NULL,
            move100_count INTEGER NOT NULL,
            move100_sum_m REAL NOT NULL,
            move100_sumsq_m2 REAL NOT NULL,
            PRIMARY KEY (stage, error_axis, bin_index)
        )
        """
    )
    connection.commit()
    return connection


def _movement_trend_bins(all_infos, stage, config):
    """Aggregate MOVE50/100 against current errors without storing samples."""
    widths = {
        "position_m": float(
            config.get("movement_trend_position_bin_width_m", 0.005)
        ),
        "orientation_deg": float(
            config.get("movement_trend_orientation_bin_width_deg", 2.0)
        ),
    }
    keys = {
        "position_m": "position_error_m",
        "orientation_deg": "orientation_error_deg",
    }
    output = []
    for axis, width in widths.items():
        bins = {}
        for info in all_infos:
            error = float(info[keys[axis]])
            if not np.isfinite(error) or error < 0.0:
                continue
            index = int(np.floor(error / width))
            row = bins.setdefault(
                index,
                {
                    "move50_count": 0,
                    "move50_sum_m": 0.0,
                    "move50_sumsq_m2": 0.0,
                    "move100_count": 0,
                    "move100_sum_m": 0.0,
                    "move100_sumsq_m2": 0.0,
                },
            )
            for duration, info_key in (
                ("move50", "tcp_50ms_net_displacement_m"),
                ("move100", "tcp_100ms_net_displacement_m"),
            ):
                movement = float(info[info_key])
                if not np.isfinite(movement):
                    continue
                row[duration + "_count"] += 1
                row[duration + "_sum_m"] += movement
                row[duration + "_sumsq_m2"] += movement * movement
        for index, row in bins.items():
            output.append(
                {
                    "phase": str(
                        stage.get("curriculum_phase", stage["task"])
                    ),
                    "stage": str(stage["name"]),
                    "task": str(stage["task"]),
                    "error_axis": axis,
                    "bin_index": int(index),
                    "bin_width": float(width),
                    **row,
                }
            )
    return output


def _append_movement_trend_bins(connection, rows):
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO movement_trend_bins VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stage, error_axis, bin_index) DO UPDATE SET
          move50_count = move50_count + excluded.move50_count,
          move50_sum_m = move50_sum_m + excluded.move50_sum_m,
          move50_sumsq_m2 = move50_sumsq_m2 + excluded.move50_sumsq_m2,
          move100_count = move100_count + excluded.move100_count,
          move100_sum_m = move100_sum_m + excluded.move100_sum_m,
          move100_sumsq_m2 = move100_sumsq_m2 + excluded.move100_sumsq_m2
        """,
        [
            (
                row["phase"],
                row["stage"],
                row["task"],
                row["error_axis"],
                row["bin_index"],
                row["bin_width"],
                row["move50_count"],
                row["move50_sum_m"],
                row["move50_sumsq_m2"],
                row["move100_count"],
                row["move100_sum_m"],
                row["move100_sumsq_m2"],
            )
            for row in rows
        ],
    )
    connection.commit()


def _append_success_cases(
    connection,
    update_index: int,
    stage: Dict,
    cases,
    control_rate_hz: float,
):
    rows = [
        (
            int(update_index),
            str(stage["name"]),
            str(stage["task"]),
            int(case["slot_id"]),
            int(case["task_sequence"]),
            float(case["initial_position_error_m"]),
            float(case["initial_orientation_error_deg"]),
            int(case["steps"]),
            float(case["steps"]) / float(control_rate_hz),
        )
        for case in cases
    ]
    if rows:
        connection.executemany(
            """
            INSERT OR REPLACE INTO successful_reaches VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def _export_success_csv(connection, output_path: Path):
    temporary = output_path.with_name(output_path.name + ".tmp")
    columns = (
        "update_index",
        "stage",
        "task",
        "slot_id",
        "task_sequence",
        "initial_position_error_m",
        "initial_orientation_error_deg",
        "steps",
        "control_time_s",
    )
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(
            connection.execute(
                "SELECT {} FROM successful_reaches ORDER BY update_index, "
                "slot_id, task_sequence".format(", ".join(columns))
            )
        )
    temporary.replace(output_path)


def _binned_reach_statistics(errors_m, steps, bin_width_m, maximum_error_m):
    edges = np.arange(
        0.0,
        float(maximum_error_m) + float(bin_width_m) * 1.0001,
        float(bin_width_m),
    )
    if len(edges) < 2:
        edges = np.asarray([0.0, float(maximum_error_m)], dtype=float)
    indices = np.clip(np.digitize(errors_m, edges) - 1, 0, len(edges) - 2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(len(centers), dtype=int)
    means = np.full(len(centers), np.nan, dtype=float)
    medians = np.full(len(centers), np.nan, dtype=float)
    p95 = np.full(len(centers), np.nan, dtype=float)
    for index in range(len(centers)):
        values = steps[indices == index]
        counts[index] = len(values)
        if len(values):
            means[index] = float(np.mean(values))
            medians[index] = float(np.median(values))
            p95[index] = float(np.percentile(values, 95.0))
    return centers, counts, means, medians, p95


def _plot_success_curves(connection, output_path: Path, settings: Dict):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = list(
        connection.execute(
            "SELECT stage, task, initial_position_error_m, "
            "initial_orientation_error_deg, steps FROM successful_reaches "
            "ORDER BY update_index, slot_id, task_sequence"
        )
    )
    temporary = output_path.with_name(output_path.stem + ".tmp.pdf")
    maximum_error = max(
        float(stage["position_error_range_m"][1])
        for stage in settings["curriculum"]["stages"]
    )
    bin_width = float(settings["success_curve_position_bin_width_m"])
    with PdfPages(temporary) as pdf:
        if not rows:
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No successful reaching tasks have been recorded yet.",
                ha="center",
                va="center",
                fontsize=16,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
        else:
            stages = np.asarray([str(row[0]) for row in rows])
            tasks = np.asarray([str(row[1]) for row in rows])
            errors = np.asarray([float(row[2]) for row in rows], dtype=float)
            orientation = np.asarray([float(row[3]) for row in rows], dtype=float)
            reach_steps = np.asarray([int(row[4]) for row in rows], dtype=float)

            figure, (curve_axis, count_axis) = plt.subplots(
                2,
                1,
                figsize=(11.69, 8.27),
                gridspec_kw={"height_ratios": [3.0, 1.0]},
                sharex=True,
            )
            colors = {"position": "#1769aa", "pose": "#c43c39"}
            for task_name in ("position", "pose"):
                mask = tasks == task_name
                if not np.any(mask):
                    continue
                centers, counts, means, medians, p95 = (
                    _binned_reach_statistics(
                        errors[mask],
                        reach_steps[mask],
                        bin_width,
                        maximum_error,
                    )
                )
                valid = counts > 0
                x_mm = centers[valid] * 1000.0
                curve_axis.plot(
                    x_mm,
                    means[valid],
                    color=colors[task_name],
                    linewidth=2.0,
                    label="{} mean".format(task_name),
                )
                curve_axis.plot(
                    x_mm,
                    medians[valid],
                    color=colors[task_name],
                    linestyle="--",
                    linewidth=1.4,
                    label="{} median".format(task_name),
                )
                curve_axis.fill_between(
                    x_mm,
                    medians[valid],
                    p95[valid],
                    color=colors[task_name],
                    alpha=0.15,
                    label="{} median--P95".format(task_name),
                )
                count_axis.step(
                    x_mm,
                    counts[valid],
                    where="mid",
                    color=colors[task_name],
                    linewidth=1.5,
                    label=task_name,
                )
            curve_axis.axhline(
                5.0,
                color="#555555",
                linestyle=":",
                linewidth=1.4,
                label="50 ms reference (5 steps)",
            )
            curve_axis.set_title(
                "Successful UR5e reaches: initial TCP error vs. control steps"
            )
            curve_axis.set_ylabel("Control steps to success (100 Hz)")
            curve_axis.grid(True, alpha=0.25)
            curve_axis.legend(ncol=2)
            count_axis.set_xlabel("Initial TCP position error (mm)")
            count_axis.set_ylabel("Cases")
            count_axis.grid(True, alpha=0.25)
            count_axis.legend()
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            for stage_name in dict.fromkeys(stages.tolist()):
                mask = stages == stage_name
                centers, counts, means, _, _ = _binned_reach_statistics(
                    errors[mask],
                    reach_steps[mask],
                    bin_width,
                    maximum_error,
                )
                valid = counts > 0
                axis.plot(
                    centers[valid] * 1000.0,
                    means[valid],
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=stage_name,
                )
            axis.set_title("Mean successful reach steps by curriculum stage")
            axis.set_xlabel("Initial TCP position error (mm)")
            axis.set_ylabel("Mean control steps to success")
            axis.grid(True, alpha=0.25)
            axis.legend(ncol=3, fontsize=8)
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

            pose_mask = tasks == "pose"
            if np.any(pose_mask):
                orientation_width = float(
                    settings["success_curve_orientation_bin_width_deg"]
                )
                maximum_orientation = max(
                    float(stage["orientation_error_range_deg"][1])
                    for stage in settings["curriculum"]["stages"]
                    if str(stage["task"]) == "pose"
                )
                centers, counts, means, medians, p95 = (
                    _binned_reach_statistics(
                        orientation[pose_mask],
                        reach_steps[pose_mask],
                        orientation_width,
                        maximum_orientation,
                    )
                )
                valid = counts > 0
                figure, axis = plt.subplots(figsize=(11.69, 8.27))
                axis.plot(
                    centers[valid], means[valid], linewidth=2, label="mean"
                )
                axis.plot(
                    centers[valid],
                    medians[valid],
                    linestyle="--",
                    label="median",
                )
                axis.fill_between(
                    centers[valid],
                    medians[valid],
                    p95[valid],
                    alpha=0.2,
                    label="median--P95",
                )
                axis.set_title(
                    "Successful pose reaches: initial orientation error vs. steps"
                )
                axis.set_xlabel("Initial TCP orientation error (deg)")
                axis.set_ylabel("Control steps to success (100 Hz)")
                axis.grid(True, alpha=0.25)
                axis.legend()
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)
        metadata = pdf.infodict()
        metadata["Title"] = "JAiLeR successful reach step statistics"
        metadata["Subject"] = (
            "Initial TCP pose error versus 100 Hz steps for successful tasks"
        )
    temporary.replace(output_path)


def _movement_trend_rows(connection):
    return list(
        connection.execute(
            """
            SELECT phase, error_axis, bin_index, bin_width,
                   SUM(move50_count), SUM(move50_sum_m),
                   SUM(move50_sumsq_m2), SUM(move100_count),
                   SUM(move100_sum_m), SUM(move100_sumsq_m2)
            FROM movement_trend_bins
            GROUP BY phase, error_axis, bin_index, bin_width
            ORDER BY phase, error_axis, bin_index
            """
        )
    )


def _movement_mean_std(count, total, total_square):
    count = int(count)
    if count <= 0:
        return float("nan"), float("nan")
    mean = float(total) / count
    variance = max(0.0, float(total_square) / count - mean * mean)
    return mean, float(np.sqrt(variance))


def _export_movement_trends(connection, csv_path: Path, pdf_path: Path):
    """Save MOVE50/MOVE100 trends against current pose error."""
    rows = _movement_trend_rows(connection)
    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    with gzip.open(temporary_csv, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "phase",
                "error_axis",
                "error_bin_center",
                "move50_count",
                "move50_mean_m",
                "move50_std_m",
                "move100_count",
                "move100_mean_m",
                "move100_std_m",
            )
        )
        for row in rows:
            phase, axis, index, width = row[:4]
            mean50, std50 = _movement_mean_std(*row[4:7])
            mean100, std100 = _movement_mean_std(*row[7:10])
            writer.writerow(
                (
                    phase,
                    axis,
                    (int(index) + 0.5) * float(width),
                    int(row[4]),
                    mean50,
                    std50,
                    int(row[7]),
                    mean100,
                    std100,
                )
            )
    temporary_csv.replace(csv_path)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    temporary_pdf = pdf_path.with_name(pdf_path.stem + ".tmp.pdf")
    with PdfPages(temporary_pdf) as pdf:
        if not rows:
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.axis("off")
            axis.text(0.5, 0.5, "No MOVE50/MOVE100 samples yet.", ha="center")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
        else:
            phases = list(dict.fromkeys(str(row[0]) for row in rows))
            for error_axis in ("position_m", "orientation_deg"):
                figure, axes = plt.subplots(
                    len(phases),
                    1,
                    figsize=(11.69, max(4.0, 3.2 * len(phases))),
                    squeeze=False,
                )
                for plot_axis, phase in zip(axes[:, 0], phases):
                    selected = [
                        row
                        for row in rows
                        if str(row[0]) == phase and str(row[1]) == error_axis
                    ]
                    for duration, offset, color in (
                        ("MOVE50", 4, "#1769aa"),
                        ("MOVE100", 7, "#c43c39"),
                    ):
                        valid = [row for row in selected if int(row[offset]) > 0]
                        if not valid:
                            continue
                        x = np.asarray(
                            [
                                (int(row[2]) + 0.5) * float(row[3])
                                for row in valid
                            ]
                        )
                        statistics = [
                            _movement_mean_std(*row[offset : offset + 3])
                            for row in valid
                        ]
                        mean = 1000.0 * np.asarray([item[0] for item in statistics])
                        std = 1000.0 * np.asarray([item[1] for item in statistics])
                        if error_axis == "position_m":
                            x = 1000.0 * x
                        plot_axis.plot(x, mean, color=color, label=duration)
                        plot_axis.fill_between(
                            x,
                            np.maximum(0.0, mean - std),
                            mean + std,
                            color=color,
                            alpha=0.12,
                        )
                    plot_axis.set_title(phase)
                    plot_axis.set_ylabel("TCP displacement (mm)")
                    plot_axis.grid(True, alpha=0.25)
                    plot_axis.legend()
                axes[-1, 0].set_xlabel(
                    "Current position error (mm)"
                    if error_axis == "position_m"
                    else "Current orientation error (deg)"
                )
                figure.suptitle(
                    "MOVE50/MOVE100 versus current {} error".format(
                        "position" if error_axis == "position_m" else "orientation"
                    )
                )
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)
        metadata = pdf.infodict()
        metadata["Title"] = "JAiLeR MOVE50/MOVE100 versus current error"
    temporary_pdf.replace(pdf_path)


def _slot_major(values):
    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("Logical rollout tensor needs time and slot dimensions")
    axes = [1, 0] + list(range(2, array.ndim))
    transposed = np.transpose(array, axes)
    return transposed.reshape((-1,) + transposed.shape[2:])


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_snapshot(
    checkpoint_directory: Path,
    config,
    start_update: int,
    model_path: str,
    tcp_site_name: str,
    pool_path: str,
    resume_checkpoint,
    actor_warm_start_checkpoint,
):
    """Archive exact V9-era sources/configuration for reproducible resumes."""
    project_root = Path(__file__).resolve().parents[2]
    snapshot = checkpoint_directory / "run_snapshots" / "start_{:06d}".format(
        int(start_update)
    )
    source_root = snapshot / "source"
    source_files = [
        "cage_system/config.py",
        "cage_system/learning/jailer_policy.py",
        "cage_system/learning/ppo2.py",
        "cage_system/learning/train_jailer.py",
        "cage_system/sim/jailer_env.py",
        "cage_system/sim/ur5e_control.py",
    ]
    hashes = {}
    for relative in source_files:
        source = project_root / relative
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))
        hashes[relative] = _sha256_file(source)

    resolved_config_path = snapshot / "resolved_config.json"
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    external_files = {}
    for name, raw_path in (
        ("mujoco_model", model_path),
        ("initial_joint_pool", pool_path),
        ("resume_checkpoint", resume_checkpoint),
        ("actor_warm_start_checkpoint", actor_warm_start_checkpoint),
    ):
        if raw_path is None:
            external_files[name] = None
            continue
        path = Path(raw_path).resolve()
        external_files[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    manifest = {
        "snapshot_contract": "resolved_config_exact_sources_and_inputs_v1",
        "start_update": int(start_update),
        "tcp_site_name": str(tcp_site_name),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "source_sha256": hashes,
        "external_inputs": external_files,
        "resolved_config": str(resolved_config_path),
    }
    manifest_path = snapshot / "manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest_path


def _failure_semantics_metrics(infos):
    """Aggregate explicit terminal-failure and time-limit bootstrap audits."""
    failure_infos = [
        item for item in infos if bool(item.get("failure_terminal", False))
    ]

    def failure_sum(key):
        return float(np.sum([float(item.get(key, 0.0)) for item in failure_infos]))

    def failure_mean(key):
        if not failure_infos:
            return 0.0
        return float(
            np.mean([float(item.get(key, 0.0)) for item in failure_infos])
        )

    return {
        "failure_terminal_count": int(len(failure_infos)),
        "failure_fixed_penalty_sum": failure_sum("failure_fixed_penalty"),
        "failure_fixed_penalty_mean": failure_mean("failure_fixed_penalty"),
        "failure_discounted_remaining_compensation_sum": failure_sum(
            "failure_discounted_remaining_compensation"
        ),
        "failure_discounted_remaining_compensation_mean": failure_mean(
            "failure_discounted_remaining_compensation"
        ),
        "failure_total_penalty_sum": failure_sum("failure_total_penalty"),
        "failure_total_penalty_mean": failure_mean("failure_total_penalty"),
        "failure_remaining_steps_mean": failure_mean(
            "failure_remaining_steps"
        ),
        "time_limit_bootstrap_count": int(
            np.sum(
                [
                    bool(item.get("task_timeout", False))
                    and not bool(
                        item.get(
                            "bootstrap_terminal",
                            item.get("episode_done", False),
                        )
                    )
                    for item in infos
                ]
            )
        ),
    }


def _collect_truncated_logical_rollout(
    trainer, environment, stage, progress_bar=None
):
    """Legacy fixed-horizon collection retained for V8 checkpoint replay."""
    steps = int(stage["rollout_steps_per_slot"])
    if steps <= 0:
        raise ValueError("JAiLeR rollout must contain at least one control step")
    slot_count = environment.slot_count
    environment.begin_rollout(stage)
    raw_state = environment.observations(stage)
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    old_logs = []
    old_values = []
    next_values = []
    bootstrap_dones = []
    all_infos = []

    for step_index in range(steps):
        policy_state = trainer.normalize_state(
            raw_state, update=str(stage["task"]) == "position"
        )
        state_tensor = torch.as_tensor(
            policy_state, dtype=torch.float32, device=trainer.device
        )
        with torch.no_grad():
            action_tensor, log_density, value, _ = trainer.model.sample(
                state_tensor, deterministic=False
            )
        executed_action = action_tensor.detach().cpu().numpy()
        raw_next_state, reward, done, infos = environment.step(
            executed_action,
            stage,
            horizon_boundary=(step_index == steps - 1),
        )
        policy_next_state = trainer.normalize_state(raw_next_state, update=False)
        next_tensor = torch.as_tensor(
            policy_next_state, dtype=torch.float32, device=trainer.device
        )
        with torch.no_grad():
            _, _, next_value = trainer.model(next_tensor)
        next_value_array = next_value.detach().cpu().numpy()
        bootstrap_done = np.zeros(slot_count, dtype=np.float32)
        for info in infos:
            if bool(
                info.get("bootstrap_terminal", info["episode_done"])
            ):
                bootstrap_done[int(info["slot_id"])] = 1.0
        next_value_array = np.where(
            bootstrap_done > 0.5, 0.0, next_value_array
        )

        states.append(policy_state)
        actions.append(executed_action)
        rewards.append(reward)
        next_states.append(policy_next_state)
        dones.append(done)
        old_logs.append(log_density.detach().cpu().numpy())
        old_values.append(value.detach().cpu().numpy())
        next_values.append(next_value_array)
        bootstrap_dones.append(bootstrap_done)
        all_infos.extend(infos)
        raw_state = raw_next_state
        if progress_bar is not None:
            progress_bar.update(slot_count)
            if step_index % 10 == 0 or step_index == steps - 1:
                progress_bar.set_postfix(
                    reward="{:.3f}".format(float(np.mean(reward))),
                    refresh=False,
                )

    rollout = RolloutBatch(
        states=_slot_major(states).astype(np.float32),
        actions=_slot_major(actions).astype(np.float32),
        rewards=_slot_major(rewards).astype(np.float32),
        next_states=_slot_major(next_states).astype(np.float32),
        dones=_slot_major(dones).astype(np.float32),
        old_log_densities=_slot_major(old_logs).astype(np.float32),
        old_values=_slot_major(old_values).astype(np.float32),
        next_values=_slot_major(next_values).astype(np.float32),
        bootstrap_dones=_slot_major(bootstrap_dones).astype(np.float32),
    )
    # Slot-major flattening plus a terminal flag at the last step of every
    # slot prevents GAE recursion from leaking between logical environments.
    if not np.all(rollout.dones.reshape(slot_count, steps)[:, -1] == 1.0):
        raise RuntimeError("Every logical rollout must end at a GAE boundary")

    rollout_endpoint_infos = sorted(
        all_infos[-slot_count:], key=lambda item: int(item["slot_id"])
    )
    terminal_infos = [
        item for item in all_infos if bool(item["episode_done"])
    ]
    truncated_infos = [
        item for item in all_infos if bool(item["rollout_truncated"])
    ]
    if len(terminal_infos) < slot_count:
        raise RuntimeError(
            "Every logical slot must contribute at least one task outcome"
        )
    terminal_position = np.asarray(
        [item["position_error_m"] for item in terminal_infos], dtype=float
    )
    terminal_orientation = np.asarray(
        [item["orientation_error_deg"] for item in terminal_infos], dtype=float
    )
    terminal_linear_speed = np.asarray(
        [item["tcp_linear_speed_mps"] for item in terminal_infos], dtype=float
    )
    terminal_angular_speed = np.asarray(
        [item["tcp_angular_speed_radps"] for item in terminal_infos], dtype=float
    )
    terminal_success = np.asarray(
        [bool(item["task_success"]) for item in terminal_infos], dtype=bool
    )
    endpoint_position = np.asarray(
        [item["position_error_m"] for item in rollout_endpoint_infos], dtype=float
    )
    endpoint_orientation = np.asarray(
        [item["orientation_error_deg"] for item in rollout_endpoint_infos],
        dtype=float,
    )
    endpoint_success = np.asarray(
        [bool(item["task_success"]) for item in rollout_endpoint_infos],
        dtype=bool,
    )
    successful_steps = np.asarray(
        [
            int(item["episode_step_count"])
            for item in terminal_infos
            if bool(item["task_success"])
        ],
        dtype=float,
    )
    movement_50ms_net = np.asarray(
        [
            item["tcp_50ms_net_displacement_m"]
            for item in all_infos
            if np.isfinite(item["tcp_50ms_net_displacement_m"])
        ],
        dtype=float,
    )
    movement_50ms_path = np.asarray(
        [
            item["tcp_50ms_path_length_m"]
            for item in all_infos
            if np.isfinite(item["tcp_50ms_path_length_m"])
        ],
        dtype=float,
    )
    movement_100ms_net = np.asarray(
        [
            item["tcp_100ms_net_displacement_m"]
            for item in all_infos
            if np.isfinite(item["tcp_100ms_net_displacement_m"])
        ],
        dtype=float,
    )
    movement_100ms_path = np.asarray(
        [
            item["tcp_100ms_path_length_m"]
            for item in all_infos
            if np.isfinite(item["tcp_100ms_path_length_m"])
        ],
        dtype=float,
    )

    def movement_mean(values):
        return float(np.mean(values)) if len(values) else float("nan")

    def movement_p95(values):
        return (
            float(np.percentile(values, 95.0))
            if len(values)
            else float("nan")
        )
    normalized_orientation = np.abs(rollout.states[:, 3:6])
    normalization_clip = float(trainer.state_normalizer.clip)
    metrics = {
        "rollout_steps_per_slot": int(steps),
        "stage_action_velocity_scale": float(stage["action_velocity_scale"]),
        "logical_slots": int(slot_count),
        "rollout_transitions": int(len(rollout.states)),
        "rollout_reward_mean": float(np.mean(rollout.rewards)),
        "rollout_reward_sum": float(np.sum(rollout.rewards)),
        "normalized_orientation_input_abs_p95": float(
            np.percentile(normalized_orientation, 95.0)
        ),
        "normalized_orientation_input_clip_fraction": float(
            np.mean(normalized_orientation >= normalization_clip - 1e-6)
        ),
        "orientation_progress_mean": float(
            np.mean([item["orientation_progress"] for item in all_infos])
        ),
        "position_progress_mean": float(
            np.mean([item["position_progress"] for item in all_infos])
        ),
        "position_progress_raw_mean": float(
            np.mean([item["position_progress_raw"] for item in all_infos])
        ),
        "position_progress_reward_mean": float(
            np.mean(
                [item["position_progress_reward"] for item in all_infos]
            )
        ),
        "quality_reward_mean": float(
            np.mean([item["quality_reward"] for item in all_infos])
        ),
        "position_quality_mean": float(
            np.mean([item["position_quality"] for item in all_infos])
        ),
        "position_fine_quality_mean": float(
            np.mean([item["position_fine_quality"] for item in all_infos])
        ),
        "position_coarse_quality_mean": float(
            np.mean([item["position_coarse_quality"] for item in all_infos])
        ),
        "orientation_progress_reward_mean": float(
            np.mean(
                [item["orientation_progress_reward"] for item in all_infos]
            )
        ),
        "orientation_progress_raw_mean": float(
            np.mean([item["orientation_progress_raw"] for item in all_infos])
        ),
        "step_quality_cost_mean": float(
            np.mean([item["step_quality_cost"] for item in all_infos])
        ),
        "joint_acceleration_penalty_mean": float(
            np.mean(
                [item["joint_acceleration_penalty"] for item in all_infos]
            )
        ),
        "success_reward_mean": float(
            np.mean([item["success_reward"] for item in all_infos])
        ),
        "near_target_progress_weight_mean": float(
            np.mean(
                [item["near_target_progress_weight"] for item in all_infos]
            )
        ),
        # Curriculum promotion uses only true task outcomes.  Tasks that have
        # not consumed their full stage-local budget at the collection boundary
        # are truncated, bootstrapped, and excluded from these endpoint metrics.
        "final_position_error_mean_m": float(np.mean(terminal_position)),
        "final_position_error_p95_m": float(
            np.percentile(terminal_position, 95.0)
        ),
        "final_orientation_error_mean_deg": float(
            np.mean(terminal_orientation)
        ),
        "final_orientation_error_p95_deg": float(
            np.percentile(terminal_orientation, 95.0)
        ),
        "final_tcp_linear_speed_mean_mps": float(
            np.mean(terminal_linear_speed)
        ),
        "final_tcp_angular_speed_mean_radps": float(
            np.mean(terminal_angular_speed)
        ),
        "terminal_success_rate": float(np.mean(terminal_success)),
        "completed_task_count": int(len(terminal_infos)),
        "successful_task_count": int(np.sum(terminal_success)),
        "task_timeout_count": int(
            np.sum([bool(item["task_timeout"]) for item in terminal_infos])
        ),
        "rollout_truncated_task_count": int(len(truncated_infos)),
        "rollout_truncated_position_error_mean_m": (
            None
            if not truncated_infos
            else float(
                np.mean(
                    [item["position_error_m"] for item in truncated_infos]
                )
            )
        ),
        "successful_task_steps_mean": (
            None if len(successful_steps) == 0 else float(np.mean(successful_steps))
        ),
        "successful_task_steps_p95": (
            None
            if len(successful_steps) == 0
            else float(np.percentile(successful_steps, 95.0))
        ),
        "rollout_endpoint_position_error_mean_m": float(
            np.mean(endpoint_position)
        ),
        "rollout_endpoint_orientation_error_mean_deg": float(
            np.mean(endpoint_orientation)
        ),
        "rollout_endpoint_success_rate": float(np.mean(endpoint_success)),
        "contact_transition_rate": float(
            np.mean([bool(item["contact"]) for item in all_infos])
        ),
        "workspace_violation_transition_rate": float(
            np.mean(
                [bool(item["workspace_violation"]) for item in all_infos]
            )
        ),
        # These are actual post-PD FK displacements across five and ten
        # 100 Hz control intervals, not estimates from commanded joints.
        "tcp_50ms_net_displacement_mean_m": movement_mean(
            movement_50ms_net
        ),
        "tcp_50ms_net_displacement_p95_m": movement_p95(
            movement_50ms_net
        ),
        "tcp_50ms_path_length_mean_m": movement_mean(
            movement_50ms_path
        ),
        "tcp_100ms_net_displacement_mean_m": movement_mean(
            movement_100ms_net
        ),
        "tcp_100ms_net_displacement_p95_m": movement_p95(
            movement_100ms_net
        ),
        "tcp_100ms_path_length_mean_m": movement_mean(
            movement_100ms_path
        ),
    }
    metrics.update(_failure_semantics_metrics(all_infos))
    successful_cases = [
        {
            "slot_id": int(item["slot_id"]),
            "task_sequence": int(item["task_sequence"]),
            "initial_position_error_m": float(
                item["initial_position_error_m"]
            ),
            "initial_orientation_error_deg": float(
                item["initial_orientation_error_deg"]
            ),
            "steps": int(item["episode_step_count"]),
        }
        for item in all_infos
        if bool(item["task_success"])
    ]
    metrics["_movement_trend_bins"] = _movement_trend_bins(
        all_infos, stage, environment.config
    )
    return rollout, metrics, successful_cases


def _drained_rollout_metrics(
    trainer,
    rollout,
    all_infos,
    endpoint_infos,
    stage,
    nominal_transition_count,
    total_control_iterations,
    action_audit,
    movement_config,
    boundary_mode="stop_spawn_then_drain_active_tasks_v1",
):
    """Summarize a rollout for which every started task has a true outcome."""
    terminal_infos = [
        item for item in all_infos if bool(item["episode_done"])
    ]
    truncated_infos = [
        item for item in all_infos if bool(item["rollout_truncated"])
    ]
    if truncated_infos:
        raise RuntimeError("A drained rollout must not contain truncations")
    if len(endpoint_infos) != int(action_audit["slot_count"]):
        raise RuntimeError("Every drained slot needs one final task outcome")
    if not all(bool(item["episode_done"]) for item in endpoint_infos):
        raise RuntimeError("A drained rollout endpoint must be a true terminal")

    terminal_position = np.asarray(
        [item["position_error_m"] for item in terminal_infos], dtype=float
    )
    terminal_orientation = np.asarray(
        [item["orientation_error_deg"] for item in terminal_infos], dtype=float
    )
    terminal_linear_speed = np.asarray(
        [item["tcp_linear_speed_mps"] for item in terminal_infos], dtype=float
    )
    terminal_angular_speed = np.asarray(
        [item["tcp_angular_speed_radps"] for item in terminal_infos], dtype=float
    )
    terminal_success = np.asarray(
        [bool(item["task_success"]) for item in terminal_infos], dtype=bool
    )
    endpoint_position = np.asarray(
        [item["position_error_m"] for item in endpoint_infos], dtype=float
    )
    endpoint_orientation = np.asarray(
        [item["orientation_error_deg"] for item in endpoint_infos], dtype=float
    )
    endpoint_success = np.asarray(
        [bool(item["task_success"]) for item in endpoint_infos], dtype=bool
    )
    successful_steps = np.asarray(
        [
            int(item["episode_step_count"])
            for item in terminal_infos
            if bool(item["task_success"])
        ],
        dtype=float,
    )

    def finite_info_values(key):
        return np.asarray(
            [
                float(item[key])
                for item in all_infos
                if np.isfinite(item[key])
            ],
            dtype=float,
        )

    def mean_or_nan(values):
        return float(np.mean(values)) if len(values) else float("nan")

    def p95_or_nan(values):
        return (
            float(np.percentile(values, 95.0))
            if len(values)
            else float("nan")
        )

    movement_50ms_net = finite_info_values("tcp_50ms_net_displacement_m")
    movement_50ms_path = finite_info_values("tcp_50ms_path_length_m")
    movement_100ms_net = finite_info_values("tcp_100ms_net_displacement_m")
    movement_100ms_path = finite_info_values("tcp_100ms_path_length_m")
    normalized_orientation = np.abs(rollout.states[:, 3:6])
    normalization_clip = float(trainer.state_normalizer.clip)
    single_task_per_slot = bool(
        boundary_mode == "single_task_per_slot_until_terminal_v1"
    )
    if single_task_per_slot:
        unused_transition_capacity = int(
            nominal_transition_count - len(rollout.states)
        )
        if unused_transition_capacity < 0:
            raise RuntimeError(
                "A single-task rollout exceeded its per-slot task budgets"
            )
        drain_transition_count = 0
        drain_control_iterations = 0
    else:
        drain_transition_count = int(
            len(rollout.states) - nominal_transition_count
        )
        if drain_transition_count < 0:
            raise RuntimeError(
                "Drained rollout is shorter than its nominal horizon"
            )
        unused_transition_capacity = 0
        drain_control_iterations = int(
            total_control_iterations - int(stage["rollout_steps_per_slot"])
        )

    task_respawn_count = int(len(terminal_infos) - action_audit["slot_count"])
    if single_task_per_slot and task_respawn_count != 0:
        raise RuntimeError("A single-task rollout must terminate exactly one task per slot")

    metrics = {
        "rollout_boundary_mode": str(boundary_mode),
        "rollout_task_contract": (
            "one_random_task_per_slot_equal_task_weight_v1"
            if single_task_per_slot
            else "respawn_until_nominal_boundary_then_drain_v1"
        ),
        "one_task_per_slot_enforced": bool(single_task_per_slot),
        "rollout_steps_per_slot": int(stage["rollout_steps_per_slot"]),
        "stage_action_velocity_scale": float(stage["action_velocity_scale"]),
        "stage_pose_residual_gain_cap": float(stage["action_velocity_scale"]),
        "physical_action_velocity_scale_mean": float(
            np.mean(action_audit["residual_gain"])
            if trainer.action_composition_mode in JOINT_ACTION_MODES
            else np.mean(
                [item["action_velocity_scale"] for item in all_infos]
            )
        ),
        "logical_slots": int(action_audit["slot_count"]),
        "nominal_rollout_transitions": int(nominal_transition_count),
        "nominal_rollout_transition_capacity": int(nominal_transition_count),
        "unused_rollout_transition_capacity": int(unused_transition_capacity),
        "rollout_transition_utilization": float(
            len(rollout.states) / nominal_transition_count
        ),
        "drain_transition_count": drain_transition_count,
        "drain_control_iterations": drain_control_iterations,
        "rollout_total_control_iterations": int(total_control_iterations),
        "rollout_transitions": int(len(rollout.states)),
        "rollout_reward_mean": float(np.mean(rollout.rewards)),
        "rollout_reward_sum": float(np.sum(rollout.rewards)),
        "normalized_orientation_input_abs_p95": float(
            np.percentile(normalized_orientation, 95.0)
        ),
        "normalized_orientation_input_clip_fraction": float(
            np.mean(normalized_orientation >= normalization_clip - 1e-6)
        ),
        "orientation_progress_mean": float(
            np.mean([item["orientation_progress"] for item in all_infos])
        ),
        "position_progress_mean": float(
            np.mean([item["position_progress"] for item in all_infos])
        ),
        "position_progress_raw_mean": float(
            np.mean([item["position_progress_raw"] for item in all_infos])
        ),
        "position_progress_reward_mean": float(
            np.mean([item["position_progress_reward"] for item in all_infos])
        ),
        "quality_reward_mean": float(
            np.mean([item["quality_reward"] for item in all_infos])
        ),
        "position_quality_mean": float(
            np.mean([item["position_quality"] for item in all_infos])
        ),
        "position_fine_quality_mean": float(
            np.mean([item["position_fine_quality"] for item in all_infos])
        ),
        "position_coarse_quality_mean": float(
            np.mean([item["position_coarse_quality"] for item in all_infos])
        ),
        "orientation_progress_reward_mean": float(
            np.mean(
                [item["orientation_progress_reward"] for item in all_infos]
            )
        ),
        "orientation_progress_raw_mean": float(
            np.mean([item["orientation_progress_raw"] for item in all_infos])
        ),
        "step_quality_cost_mean": float(
            np.mean([item["step_quality_cost"] for item in all_infos])
        ),
        "joint_acceleration_penalty_mean": float(
            np.mean(
                [item["joint_acceleration_penalty"] for item in all_infos]
            )
        ),
        "success_reward_mean": float(
            np.mean([item["success_reward"] for item in all_infos])
        ),
        "near_target_progress_weight_mean": float(
            np.mean([item["near_target_progress_weight"] for item in all_infos])
        ),
        "position_reward_weight_mean": float(
            np.mean([item["position_reward_weight"] for item in all_infos])
        ),
        "orientation_reward_weight_mean": float(
            np.mean(
                [item["orientation_reward_weight"] for item in all_infos]
            )
        ),
        "joint_action_gain_mean": (
            mean_or_nan(action_audit["residual_gain"])
            if trainer.action_composition_mode in JOINT_ACTION_MODES
            else float("nan")
        ),
        "pose_residual_gain_mean": mean_or_nan(action_audit["residual_gain"]),
        "pose_residual_gain_p95": p95_or_nan(action_audit["residual_gain"]),
        "position_action_mean_abs": mean_or_nan(action_audit["base_mean_abs"]),
        "pose_residual_action_mean_abs": mean_or_nan(
            action_audit["residual_mean_abs"]
        ),
        "scaled_pose_residual_mean_abs": mean_or_nan(
            action_audit["scaled_residual_mean_abs"]
        ),
        "composed_action_clip_fraction": mean_or_nan(
            action_audit["composition_clip_fraction"]
        ),
        "final_position_error_mean_m": float(np.mean(terminal_position)),
        "final_position_error_p95_m": float(
            np.percentile(terminal_position, 95.0)
        ),
        "final_orientation_error_mean_deg": float(
            np.mean(terminal_orientation)
        ),
        "final_orientation_error_p95_deg": float(
            np.percentile(terminal_orientation, 95.0)
        ),
        "final_tcp_linear_speed_mean_mps": float(
            np.mean(terminal_linear_speed)
        ),
        "final_tcp_angular_speed_mean_radps": float(
            np.mean(terminal_angular_speed)
        ),
        "terminal_success_rate": float(np.mean(terminal_success)),
        "equal_task_success_rate": (
            float(np.mean(terminal_success))
            if single_task_per_slot
            else None
        ),
        "started_task_count": int(len(terminal_infos)),
        "task_respawn_count": task_respawn_count,
        "completed_task_count": int(len(terminal_infos)),
        "successful_task_count": int(np.sum(terminal_success)),
        "task_timeout_count": int(
            np.sum([bool(item["task_timeout"]) for item in terminal_infos])
        ),
        "rollout_truncated_task_count": 0,
        "rollout_truncated_position_error_mean_m": None,
        "successful_task_steps_mean": (
            None if len(successful_steps) == 0 else float(np.mean(successful_steps))
        ),
        "successful_task_steps_p95": (
            None
            if len(successful_steps) == 0
            else float(np.percentile(successful_steps, 95.0))
        ),
        "rollout_endpoint_position_error_mean_m": float(
            np.mean(endpoint_position)
        ),
        "rollout_endpoint_orientation_error_mean_deg": float(
            np.mean(endpoint_orientation)
        ),
        "rollout_endpoint_success_rate": float(np.mean(endpoint_success)),
        "contact_transition_rate": float(
            np.mean([bool(item["contact"]) for item in all_infos])
        ),
        "workspace_violation_transition_rate": float(
            np.mean([bool(item["workspace_violation"]) for item in all_infos])
        ),
        "tcp_50ms_net_displacement_mean_m": mean_or_nan(movement_50ms_net),
        "tcp_50ms_net_displacement_p95_m": p95_or_nan(movement_50ms_net),
        "tcp_50ms_path_length_mean_m": mean_or_nan(movement_50ms_path),
        "tcp_100ms_net_displacement_mean_m": mean_or_nan(movement_100ms_net),
        "tcp_100ms_net_displacement_p95_m": p95_or_nan(movement_100ms_net),
        "tcp_100ms_path_length_mean_m": mean_or_nan(movement_100ms_path),
    }
    metrics.update(_failure_semantics_metrics(all_infos))
    successful_cases = [
        {
            "slot_id": int(item["slot_id"]),
            "task_sequence": int(item["task_sequence"]),
            "initial_position_error_m": float(item["initial_position_error_m"]),
            "initial_orientation_error_deg": float(
                item["initial_orientation_error_deg"]
            ),
            "steps": int(item["episode_step_count"]),
        }
        for item in terminal_infos
        if bool(item["task_success"])
    ]
    metrics["_movement_trend_bins"] = _movement_trend_bins(
        all_infos, stage, movement_config
    )
    return metrics, successful_cases


def _collect_drained_logical_rollout(
    trainer,
    environment,
    stage,
    progress_bar=None,
    single_task_per_slot=False,
):
    """Collect complete task chains, optionally exactly one task per slot."""
    nominal_steps = int(stage["rollout_steps_per_slot"])
    if nominal_steps <= 0:
        raise ValueError("JAiLeR rollout must contain at least one control step")
    slot_count = environment.slot_count
    environment.begin_rollout(stage)
    raw_state = environment.observations(stage)
    active = np.ones(slot_count, dtype=bool)
    fields = (
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "old_logs",
        "old_values",
        "next_values",
        "bootstrap_dones",
    )
    slot_buffers = [
        {name: [] for name in fields} for _ in range(slot_count)
    ]
    all_infos = []
    endpoint_by_slot = {}
    action_audit = {
        "slot_count": int(slot_count),
        "residual_gain": [],
        "base_mean_abs": [],
        "residual_mean_abs": [],
        "scaled_residual_mean_abs": [],
        "composition_clip_fraction": [],
    }
    control_iteration = 0
    maximum_iterations = (
        nominal_steps if bool(single_task_per_slot) else 2 * nominal_steps
    )

    while control_iteration < nominal_steps or np.any(active):
        if control_iteration >= maximum_iterations:
            raise RuntimeError("Drained rollout exceeded its provable time bound")
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            break
        active_raw = raw_state[active_indices]
        policy_state = trainer.normalize_state(
            active_raw, update=str(stage["task"]) == "position"
        )
        state_tensor = torch.as_tensor(
            policy_state, dtype=torch.float32, device=trainer.device
        )
        with torch.no_grad():
            decision = trainer.sample_for_environment(
                state_tensor,
                active_raw,
                stage,
                deterministic=False,
            )
        executed_full = np.zeros((slot_count, 6), dtype=np.float32)
        executed_active = (
            decision["executed_action"].detach().cpu().numpy()
        )
        executed_full[active_indices] = executed_active
        # Legacy drained rollouts replace completed tasks until the nominal
        # boundary.  The equal-task mode never replaces a slot: its first
        # randomly sampled task is its only contribution to this PPO update.
        allow_recovery = bool(
            not single_task_per_slot
            and control_iteration < nominal_steps - 1
        )
        raw_next_state, reward, _, infos = environment.step(
            executed_full,
            stage,
            active_mask=active,
            recover_terminated=allow_recovery,
        )
        info_by_slot = {int(item["slot_id"]): item for item in infos}
        if set(info_by_slot) != set(active_indices.tolist()):
            raise RuntimeError("Active logical slots and environment infos diverged")
        next_policy_state = trainer.normalize_state(
            raw_next_state[active_indices], update=False
        )
        next_tensor = torch.as_tensor(
            next_policy_state, dtype=torch.float32, device=trainer.device
        )
        with torch.no_grad():
            next_value_array = (
                trainer.model.critic(next_tensor).detach().cpu().numpy()
            )

        policy_actions = decision["policy_action"].detach().cpu().numpy()
        log_densities = decision["log_density"].detach().cpu().numpy()
        values = decision["value"].detach().cpu().numpy()
        base_actions = decision["base_action"].detach().cpu().numpy()
        residual_actions = decision["residual_action"].detach().cpu().numpy()
        scaled_residual = (
            decision["scaled_residual_action"].detach().cpu().numpy()
        )
        gains = np.asarray(decision["residual_gain"], dtype=float)
        composition_clipped = np.asarray(
            decision["composition_clipped"], dtype=bool
        )
        for local_index, slot_index in enumerate(active_indices):
            info = info_by_slot[int(slot_index)]
            true_terminal = bool(info["episode_done"])
            bootstrap_terminal = bool(
                info.get("bootstrap_terminal", true_terminal)
            )
            next_value = 0.0 if bootstrap_terminal else float(
                next_value_array[local_index]
            )
            buffer = slot_buffers[int(slot_index)]
            buffer["states"].append(policy_state[local_index])
            buffer["actions"].append(policy_actions[local_index])
            buffer["rewards"].append(float(reward[int(slot_index)]))
            buffer["next_states"].append(next_policy_state[local_index])
            buffer["dones"].append(float(info["sequence_boundary"]))
            buffer["old_logs"].append(float(log_densities[local_index]))
            buffer["old_values"].append(float(values[local_index]))
            buffer["next_values"].append(next_value)
            buffer["bootstrap_dones"].append(float(bootstrap_terminal))
            action_audit["residual_gain"].append(float(gains[local_index]))
            action_audit["base_mean_abs"].append(
                float(np.mean(np.abs(base_actions[local_index])))
            )
            action_audit["residual_mean_abs"].append(
                float(np.mean(np.abs(residual_actions[local_index])))
            )
            action_audit["scaled_residual_mean_abs"].append(
                float(np.mean(np.abs(scaled_residual[local_index])))
            )
            action_audit["composition_clip_fraction"].append(
                float(np.mean(composition_clipped[local_index]))
            )
            all_infos.append(info)
            if true_terminal and (
                bool(single_task_per_slot)
                or control_iteration >= nominal_steps - 1
            ):
                active[int(slot_index)] = False
                endpoint_by_slot[int(slot_index)] = info

        raw_state = raw_next_state
        control_iteration += 1
        if progress_bar is not None:
            progress_bar.update(len(active_indices))
            if control_iteration == nominal_steps and np.any(active):
                maximum_remaining = sum(
                    max(
                        0,
                        nominal_steps
                        - int(environment.slots[index].episode_step_count),
                    )
                    for index in np.flatnonzero(active)
                )
                progress_bar.total = int(progress_bar.n + maximum_remaining)
                progress_bar.refresh()
            if control_iteration % 10 == 0 or not np.any(active):
                progress_bar.set_postfix(
                    reward="{:.3f}".format(
                        float(np.mean(reward[active_indices]))
                    ),
                    drain=max(0, control_iteration - nominal_steps),
                    active=int(np.sum(active)),
                    refresh=False,
                )

    if set(endpoint_by_slot) != set(range(slot_count)):
        raise RuntimeError("Drain ended without a true outcome for every slot")

    def concatenate(name, dtype):
        return np.concatenate(
            [np.asarray(buffer[name], dtype=dtype) for buffer in slot_buffers],
            axis=0,
        )

    rollout = RolloutBatch(
        states=concatenate("states", np.float32),
        actions=concatenate("actions", np.float32),
        rewards=concatenate("rewards", np.float32),
        next_states=concatenate("next_states", np.float32),
        dones=concatenate("dones", np.float32),
        old_log_densities=concatenate("old_logs", np.float32),
        old_values=concatenate("old_values", np.float32),
        next_values=concatenate("next_values", np.float32),
        bootstrap_dones=concatenate("bootstrap_dones", np.float32),
    )
    cursor = 0
    for buffer in slot_buffers:
        cursor += len(buffer["dones"])
        if rollout.dones[cursor - 1] != 1.0:
            raise RuntimeError("Every drained logical chain must end at a terminal")
    endpoint_infos = [endpoint_by_slot[index] for index in range(slot_count)]
    metrics, successful_cases = _drained_rollout_metrics(
        trainer,
        rollout,
        all_infos,
        endpoint_infos,
        stage,
        nominal_transition_count=nominal_steps * slot_count,
        total_control_iterations=control_iteration,
        action_audit=action_audit,
        movement_config=environment.config,
        boundary_mode=(
            "single_task_per_slot_until_terminal_v1"
            if bool(single_task_per_slot)
            else "stop_spawn_then_drain_active_tasks_v1"
        ),
    )
    if progress_bar is not None:
        progress_bar.total = int(len(rollout.states))
        progress_bar.refresh()
    return rollout, metrics, successful_cases


def collect_logical_rollout(trainer, environment, stage, progress_bar=None):
    """Collect according to the immutable boundary contract of this run."""
    boundary_mode = str(
        environment.config.get(
            "rollout_boundary_mode", "truncate_and_bootstrap_v2"
        )
    )
    if boundary_mode == "stop_spawn_then_drain_active_tasks_v1":
        return _collect_drained_logical_rollout(
            trainer, environment, stage, progress_bar=progress_bar
        )
    if boundary_mode == "single_task_per_slot_until_terminal_v1":
        return _collect_drained_logical_rollout(
            trainer,
            environment,
            stage,
            progress_bar=progress_bar,
            single_task_per_slot=True,
        )
    if boundary_mode != "truncate_and_bootstrap_v2":
        raise ValueError("Unsupported JAiLeR rollout-boundary mode")
    return _collect_truncated_logical_rollout(
        trainer, environment, stage, progress_bar=progress_bar
    )


@dataclass
class CurriculumState:
    stage_index: int
    updates_in_stage: int
    promotion_window: deque
    final_mastered: bool = False
    critic_alignment_updates_remaining: int = 0
    critic_alignment_mode: str = "none"
    critic_alignment_updates_completed: int = 0
    critic_alignment_consecutive_passes: int = 0
    critic_alignment_forced_exit: bool = False


class JailerCurriculum:
    def __init__(self, config: Dict):
        self.config = config
        self.stages = list(config["stages"])
        if not self.stages:
            raise ValueError("JAiLeR curriculum requires at least one stage")
        self.state = CurriculumState(
            stage_index=0,
            updates_in_stage=0,
            promotion_window=deque(
                maxlen=int(config["promotion_window_updates"])
            ),
        )

    @property
    def stage(self):
        return self.stages[self.state.stage_index]

    def start_at_stage(self, stage_name: str) -> None:
        """Select an explicit stage and clear only its promotion history."""
        matches = [
            index
            for index, stage in enumerate(self.stages)
            if str(stage["name"]) == str(stage_name)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Requested stage '{}' is not a unique curriculum "
                "stage".format(stage_name)
            )
        self.state.stage_index = int(matches[0])
        self.state.updates_in_stage = 0
        self.state.promotion_window.clear()
        self.state.final_mastered = False
        self.state.critic_alignment_updates_remaining = 0
        self.state.critic_alignment_mode = "none"
        self.state.critic_alignment_updates_completed = 0
        self.state.critic_alignment_consecutive_passes = 0
        self.state.critic_alignment_forced_exit = False

    @property
    def critic_alignment_active(self):
        return self.state.critic_alignment_updates_remaining > 0

    def begin_critic_alignment(self, updates: int) -> None:
        updates = int(updates)
        if updates <= 0:
            raise ValueError("Critic alignment requires a positive update count")
        self.state.critic_alignment_updates_remaining = updates
        self.state.critic_alignment_mode = "fixed_updates_v1"
        self.state.critic_alignment_updates_completed = 0
        self.state.critic_alignment_consecutive_passes = 0
        self.state.critic_alignment_forced_exit = False

    def begin_stage_critic_alignment(self, stage) -> bool:
        mode = str(stage.get("entry_critic_alignment_mode", "fixed_updates_v1"))
        if mode == "pre_update_ev_gate_v1":
            maximum = int(stage["entry_critic_alignment_max_rollouts"])
            self.state.critic_alignment_updates_remaining = maximum
            self.state.critic_alignment_mode = mode
            self.state.critic_alignment_updates_completed = 0
            self.state.critic_alignment_consecutive_passes = 0
            self.state.critic_alignment_forced_exit = False
            return True
        updates = int(stage.get("entry_critic_alignment_updates", 0))
        if updates > 0:
            self.begin_critic_alignment(updates)
            return True
        return False

    def consume_critic_alignment_update(self, metrics=None) -> Dict:
        if not self.critic_alignment_active:
            raise RuntimeError("No Critic alignment update is pending")
        self.state.critic_alignment_updates_completed += 1
        if self.state.critic_alignment_mode == "pre_update_ev_gate_v1":
            threshold = float(
                self.stage["entry_critic_alignment_ev_threshold"]
            )
            required = int(
                self.stage["entry_critic_alignment_consecutive_rollouts"]
            )
            value = float((metrics or {}).get(
                "critic_pre_update_explained_variance", float("-inf")
            ))
            if np.isfinite(value) and value >= threshold:
                self.state.critic_alignment_consecutive_passes += 1
            else:
                self.state.critic_alignment_consecutive_passes = 0
            self.state.critic_alignment_updates_remaining -= 1
            ready = self.state.critic_alignment_consecutive_passes >= required
            exhausted = self.state.critic_alignment_updates_remaining <= 0
            if ready or exhausted:
                self.state.critic_alignment_forced_exit = bool(exhausted and not ready)
                self.state.critic_alignment_updates_remaining = 0
            return {
                "critic_alignment_gate_value": value,
                "critic_alignment_gate_threshold": threshold,
                "critic_alignment_consecutive_passes": int(
                    self.state.critic_alignment_consecutive_passes
                ),
                "critic_alignment_gate_ready": bool(ready),
                "critic_alignment_forced_exit": bool(
                    self.state.critic_alignment_forced_exit
                ),
            }
        self.state.critic_alignment_updates_remaining -= 1
        return {
            "critic_alignment_gate_ready": bool(
                self.state.critic_alignment_updates_remaining == 0
            ),
            "critic_alignment_forced_exit": False,
        }

    def observe(self, metrics):
        if self.state.final_mastered:
            # Continue optimizing the final stage until the configured total
            # update count without repeatedly emitting a promotion event.
            self.state.updates_in_stage += 1
            return None
        self.state.updates_in_stage += 1
        self.state.promotion_window.append(
            {
                "position": float(metrics["final_position_error_mean_m"]),
                "orientation": float(
                    metrics["final_orientation_error_mean_deg"]
                ),
                "success": float(metrics["terminal_success_rate"]),
            }
        )
        window_size = int(self.config["promotion_window_updates"])
        if (
            self.state.updates_in_stage
            < int(self.stage["minimum_updates"])
            or len(self.state.promotion_window) < window_size
        ):
            return None
        position = float(
            np.mean([item["position"] for item in self.state.promotion_window])
        )
        orientation = float(
            np.mean([item["orientation"] for item in self.state.promotion_window])
        )
        success = float(
            np.mean([item["success"] for item in self.state.promotion_window])
        )
        success_threshold = float(self.stage["promotion_success_rate"])
        success_strict = bool(
            self.stage.get("promotion_success_rate_strict", False)
        )
        success_mastered = bool(
            success > success_threshold
            if success_strict
            else success >= success_threshold
        )
        mastered = bool(
            position <= float(self.stage["promotion_position_mean_m"])
            and success_mastered
            and (
                str(self.stage["task"]) == "position"
                or orientation
                <= float(self.stage["promotion_orientation_mean_deg"])
            )
        )
        if not mastered:
            return None
        completed_name = str(self.stage["name"])
        if self.state.stage_index == len(self.stages) - 1:
            self.state.final_mastered = True
            return {
                "completed": completed_name,
                "next": None,
                "final_mastered": True,
            }
        self.state.stage_index += 1
        self.state.updates_in_stage = 0
        self.state.promotion_window.clear()
        return {
            "completed": completed_name,
            "next": str(self.stage["name"]),
            "final_mastered": False,
        }

    def state_dict(self):
        return {
            "stage_index": int(self.state.stage_index),
            "updates_in_stage": int(self.state.updates_in_stage),
            "promotion_window": list(self.state.promotion_window),
            "final_mastered": bool(self.state.final_mastered),
            "critic_alignment_updates_remaining": int(
                self.state.critic_alignment_updates_remaining
            ),
            "critic_alignment_mode": str(self.state.critic_alignment_mode),
            "critic_alignment_updates_completed": int(
                self.state.critic_alignment_updates_completed
            ),
            "critic_alignment_consecutive_passes": int(
                self.state.critic_alignment_consecutive_passes
            ),
            "critic_alignment_forced_exit": bool(
                self.state.critic_alignment_forced_exit
            ),
        }

    def load_state_dict(self, state):
        stage_index = int(state["stage_index"])
        if stage_index < 0 or stage_index >= len(self.stages):
            raise ValueError("Resume checkpoint curriculum stage is invalid")
        self.state.stage_index = stage_index
        self.state.updates_in_stage = int(state["updates_in_stage"])
        self.state.promotion_window.clear()
        self.state.promotion_window.extend(state["promotion_window"])
        self.state.final_mastered = bool(state.get("final_mastered", False))
        self.state.critic_alignment_updates_remaining = int(
            state.get("critic_alignment_updates_remaining", 0)
        )
        self.state.critic_alignment_mode = str(
            state.get("critic_alignment_mode", "fixed_updates_v1")
        )
        self.state.critic_alignment_updates_completed = int(
            state.get("critic_alignment_updates_completed", 0)
        )
        self.state.critic_alignment_consecutive_passes = int(
            state.get("critic_alignment_consecutive_passes", 0)
        )
        self.state.critic_alignment_forced_exit = bool(
            state.get("critic_alignment_forced_exit", False)
        )


def _checkpoint_payload(
    trainer,
    environment,
    curriculum,
    completed_updates,
    last_metrics,
):
    payload = {
        "format": JAILER_CHECKPOINT_FORMAT,
        "state_contract": _configured_state_contract(environment),
        "action_contract": _configured_action_contract(trainer),
        "policy_contract": _configured_policy_contract(trainer),
        "rollout_boundary_contract": _configured_boundary_contract(environment),
        "state_dim": 18,
        "action_dim": 6,
        "model_state": {
            name: value.detach().cpu()
            for name, value in trainer.model.state_dict().items()
        },
        "state_normalizer": trainer.state_normalizer.state_dict(),
        "completed_updates": int(completed_updates),
        "curriculum": curriculum.state_dict(),
        # Store the complete stage definition separately from the mutable
        # curriculum state.  Resume can then preserve learned weights and the
        # current stage without mixing promotion statistics collected under
        # different success/termination thresholds.
        "curriculum_definition": dict(curriculum.config),
        "logical_environment": environment.state_dict(),
        "ur5e_contract": environment.ur5e.contract_dict(),
        "last_metrics": dict(last_metrics),
        "training_initialization": dict(
            getattr(trainer, "jailer_training_initialization", {})
        ),
        "best_key": (
            None
            if getattr(trainer, "jailer_best_key", None) is None
            else list(trainer.jailer_best_key)
        ),
    }
    if trainer.action_composition_mode == JOINT_PPO_ACTION:
        payload.update(
            {
                "optimizer_contract": JOINT_PPO_OPTIMIZER_CONTRACT,
                "optimization_definition": (
                    _joint_ppo_optimization_definition(trainer)
                ),
                "actor_optimizer_state": (
                    trainer.actor_optimizer.state_dict()
                ),
                "critic_optimizer_state": (
                    trainer.critic_optimizer.state_dict()
                ),
                "adaptive_exploration_state": copy.deepcopy(
                    trainer.adaptive_exploration_state
                ),
                "adaptive_entropy_multiplier": float(
                    trainer._adaptive_entropy_multiplier
                ),
            }
        )
    else:
        payload["optimizer_state"] = trainer.optimizer.state_dict()
    return payload


def _joint_ppo_optimization_definition(trainer):
    """Freeze every setting that changes the V12 optimization problem."""
    keys = (
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "target_kl",
        "kl_stop_multiplier",
        "max_grad_norm",
        "joint_actor_learning_rate",
        "joint_critic_learning_rate",
        "actor_update_epochs",
        "critic_update_epochs",
        "minibatch_size",
    )
    definition = {
        **{key: copy.deepcopy(trainer.config[key]) for key in keys},
        "entropy_schedule": copy.deepcopy(
            trainer.config["entropy_schedule"]
        ),
    }
    if "adaptive_exploration" in trainer.config:
        definition["adaptive_exploration"] = copy.deepcopy(
            trainer.config["adaptive_exploration"]
        )
    return definition


def _save_checkpoint(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(destination))


def _move_optimizer_state_to_device(optimizer, device):
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if torch.is_tensor(value):
                parameter_state[key] = value.to(device)


def _restore_checkpoint(
    path,
    trainer,
    environment,
    curriculum,
):
    checkpoint = _torch_load(path, map_location="cpu")
    expected_state_contract = _configured_state_contract(environment)
    expected_action_contract = _configured_action_contract(trainer)
    expected_policy_contract = _configured_policy_contract(trainer)
    expected_boundary_contract = _configured_boundary_contract(environment)
    if (
        checkpoint.get("format") != JAILER_CHECKPOINT_FORMAT
        or checkpoint.get("state_contract") != expected_state_contract
        or checkpoint.get("action_contract") != expected_action_contract
        or checkpoint.get("policy_contract") != expected_policy_contract
        or checkpoint.get("rollout_boundary_contract")
        != expected_boundary_contract
        or int(checkpoint.get("state_dim", -1)) != 18
        or int(checkpoint.get("action_dim", -1)) != 6
    ):
        raise ValueError(
            "Resume file is not compatible with the configured JAiLeR "
            "contract; use it only as --actor-warm-start-checkpoint"
        )
    if checkpoint.get("ur5e_contract") != environment.ur5e.contract_dict():
        raise ValueError("Resume checkpoint was built for a different UR5e/TCP contract")
    trainer.model.load_state_dict(checkpoint["model_state"])
    if trainer.action_composition_mode == JOINT_PPO_ACTION:
        if checkpoint.get("optimizer_contract") != JOINT_PPO_OPTIMIZER_CONTRACT:
            raise ValueError(
                "V12 full resume requires an independent Actor/Critic Adam "
                "checkpoint; use an older file only as Actor warm start"
            )
        if checkpoint.get("optimization_definition") != (
            _joint_ppo_optimization_definition(trainer)
        ):
            raise ValueError(
                "V12 resume optimization settings differ from the saved "
                "Actor/Critic/entropy contract"
            )
        required_optimizer_states = {
            "actor_optimizer_state",
            "critic_optimizer_state",
        }
        if not required_optimizer_states.issubset(checkpoint):
            raise ValueError(
                "V12 resume checkpoint is missing one or both independent "
                "optimizer states"
            )
        trainer.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state"]
        )
        trainer.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state"]
        )
        _move_optimizer_state_to_device(
            trainer.actor_optimizer, trainer.device
        )
        _move_optimizer_state_to_device(
            trainer.critic_optimizer, trainer.device
        )
        trainer.adaptive_exploration_state = copy.deepcopy(
            checkpoint.get(
                "adaptive_exploration_state",
                {
                    "stage_name": None,
                    "score_history": [],
                    "boosts_used": 0,
                    "boost_updates_remaining": 0,
                },
            )
        )
        trainer.set_adaptive_entropy_multiplier(
            float(checkpoint.get("adaptive_entropy_multiplier", 1.0))
        )
    else:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        _move_optimizer_state_to_device(trainer.optimizer, trainer.device)
    trainer.state_normalizer.load_state_dict(checkpoint["state_normalizer"])
    trainer.jailer_training_initialization = dict(
        checkpoint.get(
            "training_initialization",
            {
                "mode": "legacy_full_resume",
                "source": str(path),
            },
        )
    )
    saved_curriculum_definition = checkpoint.get("curriculum_definition")
    curriculum_definition_changed = bool(
        saved_curriculum_definition is None
        or saved_curriculum_definition != curriculum.config
    )
    saved_best_key = checkpoint.get("best_key")
    trainer.jailer_best_key = (
        None
        if saved_best_key is None or curriculum_definition_changed
        else (int(saved_best_key[0]), float(saved_best_key[1]))
    )
    environment.load_state_dict(checkpoint["logical_environment"])
    curriculum.load_state_dict(checkpoint["curriculum"])
    last_metrics = dict(checkpoint.get("last_metrics", {}))
    if curriculum_definition_changed:
        curriculum.state.promotion_window.clear()
        curriculum.state.final_mastered = False
        last_metrics["resume_curriculum_definition_changed"] = True
        last_metrics["resume_promotion_window_cleared"] = True
    return int(checkpoint["completed_updates"]), last_metrics


def _warm_start_actor(
    path,
    trainer,
    environment,
    legacy_orientation_input_std_deg,
):
    """Keep a learned feedback Actor while resetting the obsolete Critic."""
    checkpoint = _torch_load(path, map_location="cpu")
    source_state_contract = checkpoint.get("state_contract")
    source_policy_contract = checkpoint.get("policy_contract")
    if (
        checkpoint.get("format") != JAILER_CHECKPOINT_FORMAT
        or source_state_contract
        not in {
            JAILER_STATE_CONTRACT,
            JAILER_PHASED_STATE_CONTRACT,
            JAILER_LEGACY_POSITION_MASKED_STATE_CONTRACT,
        }
        or checkpoint.get("action_contract")
        not in {
            JAILER_ACTION_CONTRACT,
            JAILER_LEGACY_ACTION_CONTRACT,
            JAILER_SEPARATED_ACTION_CONTRACT,
            JAILER_JOINT_ACTION_CONTRACT,
        }
        or int(checkpoint.get("state_dim", -1)) != 18
        or int(checkpoint.get("action_dim", -1)) != 6
    ):
        raise ValueError(
            "Actor warm start is not a compatible UR5e JAiLeR checkpoint"
        )
    if checkpoint.get("ur5e_contract") != environment.ur5e.contract_dict():
        raise ValueError(
            "Actor warm-start checkpoint uses a different UR5e/TCP contract"
        )
    actor_state = {
        name[len("actor.") :]: value
        for name, value in checkpoint["model_state"].items()
        if name.startswith("actor.")
    }
    if not actor_state:
        raise ValueError("Actor warm-start checkpoint contains no Actor weights")
    target_separated = bool(
        trainer.action_composition_mode == SEPARATE_RESIDUAL_ACTION
    )
    target_joint = bool(trainer.action_composition_mode in JOINT_ACTION_MODES)
    if target_joint:
        position_actor_state = {
            name: value
            for name, value in actor_state.items()
            if name == "log_std"
            or name.startswith("features.")
            or name.startswith("mean_head.")
        }
        trainer.model.actor.load_state_dict(position_actor_state, strict=True)
        if trainer.action_composition_mode == JOINT_REFERENCE_ACTION:
            trainer.model.reference_actor.load_state_dict(
                position_actor_state, strict=True
            )
            for parameter in trainer.model.reference_actor.parameters():
                parameter.requires_grad_(False)
    elif target_separated:
        # V9 deliberately imports only the proven position controller.  Old
        # pose-residual weights were optimized under a different physical
        # composition and must never leak into the new residual policy.
        position_actor_state = {
            name: value
            for name, value in actor_state.items()
            if name == "log_std"
            or name.startswith("features.")
            or name.startswith("mean_head.")
        }
        incompatible = trainer.model.actor.load_state_dict(
            position_actor_state, strict=False
        )
        disallowed_missing = [
            name
            for name in incompatible.missing_keys
            if not (
                name == "pose_residual_gate"
                or name == "pose_residual_log_std"
                or name.startswith("pose_residual_features.")
                or name.startswith("pose_residual_head.")
            )
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Position Actor cannot be isolated for the V9 residual policy: "
                "missing={}, unexpected={}".format(
                    disallowed_missing, incompatible.unexpected_keys
                )
            )
        with torch.no_grad():
            trainer.model.actor.pose_residual_head.weight.zero_()
            trainer.model.actor.pose_residual_head.bias.zero_()
    elif source_policy_contract == JAILER_POLICY_CONTRACT:
        trainer.model.actor.load_state_dict(actor_state)
    else:
        incompatible = trainer.model.actor.load_state_dict(
            actor_state, strict=False
        )
        disallowed_missing = [
            name
            for name in incompatible.missing_keys
            if not (
                name == "pose_residual_gate"
                or name.startswith("pose_residual_features.")
                or name.startswith("pose_residual_head.")
            )
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Legacy Actor cannot be mapped into the pose-residual policy: "
                "missing={}, unexpected={}".format(
                    disallowed_missing, incompatible.unexpected_keys
                )
            )
    trainer.state_normalizer.load_state_dict(checkpoint["state_normalizer"])
    legacy_orientation_adapter = bool(
        source_state_contract == JAILER_LEGACY_POSITION_MASKED_STATE_CONTRACT
    )
    if legacy_orientation_adapter:
        # In the legacy position curriculum state[3:6] was identically zero,
        # so these first-layer columns retained arbitrary initialization and
        # the recorded variance collapsed to almost zero.  Zeroing only those
        # columns preserves the learned position policy exactly while allowing
        # subsequent PPO gradients to learn orientation control from a neutral
        # starting point.
        first_actor_layer = next(
            module
            for module in trainer.model.actor.features
            if isinstance(module, torch.nn.Linear)
        )
        with torch.no_grad():
            first_actor_layer.weight[:, 3:6].zero_()
        orientation_std_rad = np.deg2rad(
            float(legacy_orientation_input_std_deg)
        )
        if orientation_std_rad <= 0.0:
            raise ValueError(
                "Legacy orientation input standard deviation must be positive"
            )
        trainer.state_normalizer.mean[3:6] = 0.0
        trainer.state_normalizer.variance[3:6] = orientation_std_rad ** 2
    trainer.reset_critic()
    trainer.rebuild_optimizer()
    trainer.jailer_training_initialization = {
        "mode": "actor_and_normalizer_warm_start_new_critic",
        "source": str(path),
        "source_completed_updates": int(checkpoint["completed_updates"]),
        "source_state_contract": str(source_state_contract),
        "source_policy_contract": source_policy_contract,
        "target_policy_contract": _configured_policy_contract(trainer),
        "position_actor_only_import": bool(target_separated or target_joint),
        "frozen_position_reference_initialized": bool(
            trainer.action_composition_mode == JOINT_REFERENCE_ACTION
        ),
        "joint_actor_initialized_without_reference": bool(
            trainer.action_composition_mode == JOINT_PPO_ACTION
        ),
        "legacy_orientation_adapter": legacy_orientation_adapter,
        "legacy_orientation_input_std_deg": (
            float(legacy_orientation_input_std_deg)
            if legacy_orientation_adapter
            else None
        ),
    }
    return int(checkpoint["completed_updates"])


def _prepare_pose_transition(trainer, minimum_log_std):
    """Increase exploration without altering the learned mean position policy."""
    minimum_log_std = float(minimum_log_std)
    active_log_std = trainer.active_log_std()
    with torch.no_grad():
        before = active_log_std.detach().cpu().numpy().copy()
        active_log_std.copy_(
            torch.maximum(
                active_log_std,
                torch.full_like(active_log_std, minimum_log_std),
            )
        )
        after = active_log_std.detach().cpu().numpy().copy()
    return {
        "pose_transition_log_std_before": before.tolist(),
        "pose_transition_log_std_after": after.tolist(),
    }


def _prepare_masked_to_observed_transition(
    trainer, orientation_input_std_deg
):
    """Keep the position Actor but replace a masked-observation Critic."""
    first_actor_layer = next(
        module
        for module in trainer.model.actor.features
        if isinstance(module, torch.nn.Linear)
    )
    with torch.no_grad():
        first_actor_layer.weight[:, 3:6].zero_()
    orientation_std_rad = np.deg2rad(float(orientation_input_std_deg))
    if orientation_std_rad <= 0.0:
        raise ValueError("Orientation input standard deviation must be positive")
    trainer.state_normalizer.mean[3:6] = 0.0
    trainer.state_normalizer.variance[3:6] = orientation_std_rad ** 2
    trainer.reset_critic()
    trainer.rebuild_optimizer()
    return {
        "stage_transition_mode": "masked_to_observed_new_critic",
        "stage_transition_actor_weights_preserved": True,
        "stage_transition_actor_orientation_columns_zeroed": True,
        "stage_transition_critic_weights_preserved": False,
        "stage_transition_optimizer_state_preserved": False,
        "stage_transition_orientation_input_std_deg": float(
            orientation_input_std_deg
        ),
    }


def _apply_stage_entry_transition(
    trainer, curriculum, previous_stage, next_stage, settings
):
    """Apply an explicit and checkpointed curriculum-boundary contract."""
    mode = str(next_stage.get("entry_transition", "continuous"))
    metrics = {"stage_transition_mode": mode}
    if mode == "masked_to_observed_new_critic":
        metrics = _prepare_masked_to_observed_transition(
            trainer, settings["legacy_orientation_input_std_deg"]
        )
        started = curriculum.begin_stage_critic_alignment(next_stage)
        metrics["critic_alignment_started"] = bool(started)
    elif mode == "preserve_critic_reset_adam_align":
        trainer.set_curriculum_task(str(next_stage["task"]))
        # Preserve both networks, but discard every old Adam moment before
        # fitting V(s) to fresh B1 outcomes under the new action/reward rules.
        trainer.rebuild_optimizer()
        metrics.update(
            _prepare_pose_transition(
                trainer, settings["pose_transition_min_log_std"]
            )
        )
        started = curriculum.begin_stage_critic_alignment(next_stage)
        metrics.update(
            {
                "stage_transition_actor_weights_preserved": True,
                "stage_transition_critic_weights_preserved": True,
                "stage_transition_optimizer_state_preserved": False,
                "critic_alignment_started": bool(started),
            }
        )
    elif mode == "new_critic_reset_adam_align":
        trainer.set_curriculum_task(str(next_stage["task"]))
        trainer.reset_critic()
        trainer.rebuild_optimizer()
        started = curriculum.begin_stage_critic_alignment(next_stage)
        metrics.update(
            {
                "stage_transition_actor_weights_preserved": True,
                "stage_transition_critic_weights_preserved": False,
                "stage_transition_optimizer_state_preserved": False,
                "critic_alignment_started": bool(started),
            }
        )
    elif mode == "continuous":
        if str(previous_stage["task"]) != str(next_stage["task"]):
            trainer.set_curriculum_task(str(next_stage["task"]))
            if trainer.action_composition_mode not in JOINT_ACTION_MODES:
                metrics.update(
                    _prepare_pose_transition(
                        trainer, settings["pose_transition_min_log_std"]
                    )
                )
    else:
        raise ValueError("Unsupported stage entry transition '{}'".format(mode))

    metrics.update(trainer.prepare_training_stage(next_stage, entering=True))

    history = trainer.jailer_training_initialization.setdefault(
        "stage_transition_history", []
    )
    history.append(
        {
            "from": str(previous_stage["name"]),
            "to": str(next_stage["name"]),
            **metrics,
        }
    )
    return metrics


def train_jailer_curriculum(
    config,
    model_path: str,
    tcp_site_name: str,
    output_path: str,
    device: str = "auto",
    initial_pool_path: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    resume_start_stage: Optional[str] = None,
    actor_warm_start_checkpoint: Optional[str] = None,
    actor_warm_start_stage: Optional[str] = None,
):
    settings = config["jailer"]
    pose_only_warm_start = bool(
        settings["curriculum"].get("pose_only_actor_warm_start", False)
    )
    if (
        pose_only_warm_start
        and resume_checkpoint is None
        and actor_warm_start_checkpoint is None
    ):
        raise ValueError(
            "The joint pose-only curriculum requires an Observed P6 Actor "
            "warm start or a compatible full resume"
        )
    if float(settings["control_rate_hz"]) != 100.0:
        raise ValueError("The agreed JAiLeR control frequency is exactly 100 Hz")
    if int(settings["logical_slots"]) <= 0:
        raise ValueError("JAiLeR needs at least one logical slot")
    seed = int(settings.get("seed", config["system"]["seed"]))
    np.random.seed(seed)
    torch.manual_seed(seed)
    ur5e = UR5eControlModel(
        model_path,
        config["robot"],
        tcp_site_name=tcp_site_name,
        control_rate_hz=float(settings["control_rate_hz"]),
    )
    output = Path(output_path)
    generated_pool_path = output.parent / (
        output.stem + "_initial_joint_pool.npz"
    )
    if initial_pool_path is None:
        arrays = generate_initial_joint_pool(
            ur5e,
            settings,
            int(settings["logical_slots"]),
            seed,
        )
        save_initial_joint_pool(str(generated_pool_path), arrays)
        pool_path = str(generated_pool_path)
    else:
        pool_path = str(initial_pool_path)
    pool = load_initial_joint_pool(
        pool_path, ur5e, int(settings["logical_slots"])
    )
    environment = JailerLogicalVectorEnv(ur5e, settings, pool, seed + 1)
    trainer = JailerPPO2Trainer(settings["ppo"], device=device)
    trainer.jailer_training_initialization = {"mode": "random"}
    trainer.jailer_best_key = None
    curriculum = JailerCurriculum(settings["curriculum"])
    start_update = 0
    last_metrics = {}
    if resume_checkpoint is not None and actor_warm_start_checkpoint is not None:
        raise ValueError(
            "Use either full resume or Actor warm start, not both"
        )
    if actor_warm_start_stage is not None and actor_warm_start_checkpoint is None:
        raise ValueError(
            "--actor-warm-start-stage requires --actor-warm-start-checkpoint"
        )
    if resume_start_stage is not None and resume_checkpoint is None:
        raise ValueError(
            "--resume-start-stage requires --resume-checkpoint"
        )
    if resume_start_stage is not None and actor_warm_start_checkpoint is not None:
        raise ValueError(
            "--resume-start-stage is valid only for full resume"
        )
    if resume_checkpoint is not None:
        start_update, last_metrics = _restore_checkpoint(
            resume_checkpoint, trainer, environment, curriculum
        )
        if resume_start_stage is not None:
            curriculum.start_at_stage(resume_start_stage)
            trainer.jailer_training_initialization["latest_resume"] = {
                "mode": "full_state_resume_with_explicit_stage",
                "source": str(resume_checkpoint),
                "source_completed_updates": int(start_update),
                "requested_start_stage": str(resume_start_stage),
                "preserved": [
                    "actor",
                    "critic",
                    *(
                        ["actor_optimizer", "critic_optimizer"]
                        if trainer.action_composition_mode
                        == JOINT_PPO_ACTION
                        else ["optimizer"]
                    ),
                    "state_normalizer",
                ],
            }
            last_metrics["resume_start_stage"] = str(resume_start_stage)
            last_metrics["resume_full_training_state_preserved"] = True
    elif actor_warm_start_checkpoint is not None:
        source_updates = _warm_start_actor(
            actor_warm_start_checkpoint,
            trainer,
            environment,
            float(settings["legacy_orientation_input_std_deg"]),
        )
        if actor_warm_start_stage is not None:
            curriculum.start_at_stage(actor_warm_start_stage)
            if (
                bool(
                    trainer.jailer_training_initialization.get(
                        "legacy_orientation_adapter", False
                    )
                )
                and str(curriculum.stage["task"]) == "pose"
            ):
                raise ValueError(
                    "A legacy position-masked Actor cannot safely skip the "
                    "new orientation-aware position curriculum; warm-start "
                    "from P1 or use a v2 full-pose-observation checkpoint"
                )
            trainer.jailer_training_initialization[
                "requested_start_stage"
            ] = str(actor_warm_start_stage)
        last_metrics = {
            "actor_warm_start_source_completed_updates": int(source_updates)
        }
    trainer.set_curriculum_task(str(curriculum.stage["task"]))
    trainer.prepare_training_stage(
        curriculum.stage,
        entering=bool(
            resume_checkpoint is None or resume_start_stage is not None
        ),
        updates_in_stage=int(curriculum.state.updates_in_stage),
    )
    if resume_checkpoint is None or resume_start_stage is not None:
        curriculum.begin_stage_critic_alignment(curriculum.stage)

    maximum_updates = int(settings["ppo"]["updates"])
    checkpoint_directory = output.parent / (output.stem + "_checkpoints")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    run_snapshot_path = _write_run_snapshot(
        checkpoint_directory,
        config,
        start_update,
        model_path,
        tcp_site_name,
        pool_path,
        resume_checkpoint,
        actor_warm_start_checkpoint,
    )
    trainer.jailer_training_initialization["run_snapshot_manifest"] = str(
        run_snapshot_path
    )
    history_path = checkpoint_directory / "training_history.jsonl"
    success_database_path = (
        checkpoint_directory / "successful_reaches.sqlite3"
    )
    success_csv_path = checkpoint_directory / "successful_reaches.csv.gz"
    success_pdf_path = (
        checkpoint_directory
        / "successful_reach_steps_by_initial_error.pdf"
    )
    movement_csv_path = checkpoint_directory / "move50_move100_vs_error.csv.gz"
    movement_pdf_path = checkpoint_directory / "move50_move100_vs_error.pdf"
    success_connection = _prepare_success_database(
        success_database_path,
        reset=resume_checkpoint is None,
    )
    if resume_checkpoint is None:
        history_path.write_text("", encoding="utf-8")
    checkpoint_interval = int(settings["checkpoint_interval_updates"])
    success_curve_interval = int(settings["success_curve_interval_updates"])
    movement_trend_interval = int(
        settings.get("movement_trend_interval_updates", success_curve_interval)
    )
    update_bar = tqdm(
        range(start_update, maximum_updates),
        total=maximum_updates,
        initial=start_update,
        desc="UR5e JAiLeR curriculum",
        unit="update",
        dynamic_ncols=True,
    )
    for update_index in update_bar:
        stage_index_before = int(curriculum.state.stage_index)
        stage_before = dict(curriculum.stage)
        trainer.prepare_adaptive_exploration(stage_before)
        trainer.prepare_training_stage(
            stage_before,
            entering=False,
            updates_in_stage=int(curriculum.state.updates_in_stage),
        )
        rollout_size = int(stage_before["rollout_steps_per_slot"]) * int(
            settings["logical_slots"]
        )
        rollout_bar = tqdm(
            total=rollout_size,
            desc="{} rollout".format(stage_before["name"]),
            unit="transition",
            leave=False,
            dynamic_ncols=True,
        )
        try:
            rollout, rollout_metrics, successful_cases = collect_logical_rollout(
                trainer,
                environment,
                stage_before,
                progress_bar=rollout_bar,
            )
        finally:
            rollout_bar.close()
        movement_trend_bins = rollout_metrics.pop(
            "_movement_trend_bins", []
        )
        _append_movement_trend_bins(
            success_connection, movement_trend_bins
        )
        critic_alignment_update = bool(curriculum.critic_alignment_active)
        if critic_alignment_update:
            optimizer_metrics = trainer.align_critic(rollout)
            optimizer_metrics.update(
                curriculum.consume_critic_alignment_update(optimizer_metrics)
            )
            promotion = None
            stage_transition_metrics = {}
        else:
            optimizer_metrics = trainer.update(rollout)
            optimizer_metrics.setdefault("optimization_mode", "joint_ppo")
            optimizer_metrics.update(
                trainer.update_adaptive_exploration(
                    stage_before, rollout_metrics, optimizer_metrics
                )
            )
            promotion = curriculum.observe(rollout_metrics)
            stage_transition_metrics = {}
            if promotion is not None and promotion["next"] is not None:
                stage_transition_metrics = _apply_stage_entry_transition(
                    trainer,
                    curriculum,
                    stage_before,
                    dict(curriculum.stage),
                    settings,
                )
        _append_success_cases(
            success_connection,
            update_index + 1,
            stage_before,
            successful_cases,
            float(settings["control_rate_hz"]),
        )
        stage_success_threshold = float(
            stage_before["promotion_success_rate"]
        )
        stage_success = float(rollout_metrics["terminal_success_rate"])
        stage_success_eligible = bool(
            stage_success > stage_success_threshold
            if bool(stage_before.get("promotion_success_rate_strict", False))
            else stage_success >= stage_success_threshold
        )
        best_checkpoint_eligible = bool(
            not critic_alignment_update
            and stage_success_eligible
            and float(rollout_metrics["final_position_error_mean_m"])
            <= float(stage_before["promotion_position_mean_m"])
            and (
                str(stage_before["task"]) == "position"
                or float(
                    rollout_metrics["final_orientation_error_mean_deg"]
                )
                <= float(stage_before["promotion_orientation_mean_deg"])
            )
        )
        last_metrics = {
            **optimizer_metrics,
            **rollout_metrics,
            **stage_transition_metrics,
            "update": int(update_index + 1),
            "curriculum_stage": str(stage_before["name"]),
            "curriculum_task": str(stage_before["task"]),
            "best_checkpoint_eligible": best_checkpoint_eligible,
            "critic_alignment_update": critic_alignment_update,
            "critic_alignment_updates_remaining": int(
                curriculum.state.critic_alignment_updates_remaining
            ),
            "promoted_to": None if promotion is None else promotion["next"],
            "final_curriculum_mastered": bool(
                curriculum.state.final_mastered
            ),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(last_metrics, ensure_ascii=False) + "\n")
        pose_cost = float(rollout_metrics["final_position_error_mean_m"])
        if str(stage_before["task"]) == "pose":
            pose_cost += np.deg2rad(
                float(rollout_metrics["final_orientation_error_mean_deg"])
            ) * float(settings["best_orientation_radian_weight"])
        candidate_key = (stage_index_before, -pose_cost)
        is_new_best = bool(
            best_checkpoint_eligible
            and (
                trainer.jailer_best_key is None
                or candidate_key > trainer.jailer_best_key
            )
        )
        if is_new_best:
            trainer.jailer_best_key = candidate_key
        payload = _checkpoint_payload(
            trainer,
            environment,
            curriculum,
            update_index + 1,
            last_metrics,
        )
        _save_checkpoint(checkpoint_directory / "latest.pt", payload)
        if (update_index + 1) % checkpoint_interval == 0:
            _save_checkpoint(
                checkpoint_directory
                / "update_{:04d}.pt".format(update_index + 1),
                payload,
            )
        if (update_index + 1) % success_curve_interval == 0:
            _export_success_csv(success_connection, success_csv_path)
            _plot_success_curves(
                success_connection,
                success_pdf_path,
                settings,
            )
        if (update_index + 1) % movement_trend_interval == 0:
            _export_movement_trends(
                success_connection, movement_csv_path, movement_pdf_path
            )
        if promotion is not None:
            _save_checkpoint(
                checkpoint_directory
                / "stage_{}_complete.pt".format(promotion["completed"]),
                payload,
            )
        if is_new_best:
            _save_checkpoint(checkpoint_directory / "best.pt", payload)
        update_bar.set_postfix(
            stage=stage_before["name"],
            mode=(
                "critic-align"
                if last_metrics["critic_alignment_update"]
                else "ppo"
            ),
            cap=int(last_metrics["rollout_steps_per_slot"]),
            scale="{:.2f}".format(
                last_metrics.get(
                    "physical_action_velocity_scale_mean",
                    last_metrics["stage_action_velocity_scale"],
                )
            ),
            gain="{:.2f}".format(
                last_metrics.get(
                    "pose_residual_gain_mean",
                    last_metrics["stage_action_velocity_scale"],
                )
            ),
            drain=int(last_metrics.get("drain_transition_count", 0)),
            ent_c="{:.4f}".format(
                float(last_metrics.get("entropy_coefficient", 0.0))
            ),
            reward="{:.3f}".format(last_metrics["rollout_reward_mean"]),
            pos="{:.1f}mm".format(
                1000.0 * last_metrics["final_position_error_mean_m"]
            ),
            angle="{:.1f}deg".format(
                last_metrics["final_orientation_error_mean_deg"]
            ),
            move50="{:.1f}mm".format(
                1000.0 * last_metrics["tcp_50ms_net_displacement_mean_m"]
            ),
            move100=(
                "--"
                if not np.isfinite(
                    last_metrics["tcp_100ms_net_displacement_mean_m"]
                )
                else "{:.1f}mm".format(
                    1000.0
                    * last_metrics[
                        "tcp_100ms_net_displacement_mean_m"
                    ]
                )
            ),
            success="{:.1f}%".format(
                100.0 * last_metrics["terminal_success_rate"]
            ),
            cut=int(last_metrics["rollout_truncated_task_count"]),
            rotclip="{:.1f}%".format(
                100.0
                * last_metrics[
                    "normalized_orientation_input_clip_fraction"
                ]
            ),
            reach_steps=(
                "--"
                if last_metrics["successful_task_steps_mean"] is None
                else "{:.1f}".format(
                    last_metrics["successful_task_steps_mean"]
                )
            ),
            refresh=False,
        )
        if curriculum.state.final_mastered and bool(
            settings["curriculum"]["stop_when_final_stage_mastered"]
        ):
            break

    final_payload = _checkpoint_payload(
        trainer,
        environment,
        curriculum,
        int(last_metrics.get("update", start_update)),
        last_metrics,
    )
    _save_checkpoint(output, final_payload)
    _export_success_csv(success_connection, success_csv_path)
    _plot_success_curves(success_connection, success_pdf_path, settings)
    _export_movement_trends(
        success_connection, movement_csv_path, movement_pdf_path
    )
    successful_reach_count = int(
        success_connection.execute(
            "SELECT COUNT(*) FROM successful_reaches"
        ).fetchone()[0]
    )
    success_connection.close()
    return {
        "output": str(output),
        "checkpoint_directory": str(checkpoint_directory),
        "successful_reach_database": str(success_database_path),
        "successful_reach_csv": str(success_csv_path),
        "successful_reach_pdf": str(success_pdf_path),
        "movement_trend_csv": str(movement_csv_path),
        "movement_trend_pdf": str(movement_pdf_path),
        "successful_reach_count": successful_reach_count,
        "initial_joint_pool": pool_path,
        "run_snapshot_manifest": str(run_snapshot_path),
        "training_initialization": dict(
            trainer.jailer_training_initialization
        ),
        "completed_updates": int(last_metrics.get("update", start_update)),
        "final_stage": str(curriculum.stage["name"]),
        "final_curriculum_mastered": bool(curriculum.state.final_mastered),
        "last_metrics": last_metrics,
        "ur5e_contract": ur5e.contract_dict(),
    }
