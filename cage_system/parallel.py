"""Checkpointed, process-parallel Cage label generation."""

import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np

from cage_system.audit_io import open_audit_text
from cage_system.io import validate_trajectory_detection_period
from cage_system.learning.dataset import CageDatasetArrays
from cage_system.learning.state_arc_dataset import (
    STATE_ARC_DATASET_CONTRACT,
    CageStateArcDatasetArrays,
    load_generated_label_arrays,
)
from cage_system.pipeline import LabelGenerator


_WORKER_GENERATOR = None
_WORKER_THREADPOOL_LIMITER = None


def _empty_arrays(config, nominal_detection_period_s):
    if str(config["labeling"].get("dataset_contract", "")) == STATE_ARC_DATASET_CONTRACT:
        return CageStateArcDatasetArrays.empty(nominal_detection_period_s)
    return CageDatasetArrays(
        np.empty((0, 5, 4)),
        np.empty((0, 6)),
        np.empty((0,), dtype=int),
        np.empty((0,)),
        float(nominal_detection_period_s),
    )


def _part_paths(part_directory, track_id):
    root = Path(part_directory)
    stem = "{:06d}".format(int(track_id))
    return (
        root / ("labels_" + stem + ".npz"),
        root / ("audit_" + stem + ".jsonl.gz"),
        root / ("done_" + stem + ".json"),
    )


def _atomic_array_save(arrays, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    arrays.save(str(temporary))
    os.replace(str(temporary), str(destination))


def _initialize_worker(config):
    """Build one MuJoCo model pair per process and reuse it across tracks."""
    global _WORKER_GENERATOR, _WORKER_THREADPOOL_LIMITER
    # NumPy/OpenBLAS may default to every host core.  Label generation already
    # parallelizes by process, so allowing every worker to create another full
    # BLAS thread pool severely oversubscribes the CPU.  Keep numerical kernels
    # single-threaded inside each independent track worker.
    try:
        from threadpoolctl import threadpool_limits

        _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _WORKER_THREADPOOL_LIMITER = None
    backend_name = config["labeling"]["feasibility_backend"]
    if backend_name == "workspace":
        backend = None
    elif backend_name == "mujoco":
        from cage_system.sim.mujoco_feasibility import MujocoFeasibilityBackend

        backend = MujocoFeasibilityBackend(
            config["labeling"]["mujoco_model_path"],
            config["labeling"]["mujoco_tcp_site"],
            config["robot"],
        )
    else:
        raise ValueError(
            "Checkpointed process workers support workspace or MuJoCo backends; "
            "MoveIt requires one independently configured ROS namespace per worker"
        )
    _WORKER_GENERATOR = LabelGenerator(config, backend)


def _label_one_track(trajectory, part_directory):
    if _WORKER_GENERATOR is None:
        raise RuntimeError("Label worker was not initialized")
    label_path, audit_path, done_path = _part_paths(
        part_directory, trajectory.track_id
    )
    temporary_audit = audit_path.with_name(
        audit_path.name.replace(".jsonl.gz", ".tmp.jsonl.gz")
    )
    temporary_done = done_path.with_suffix(".tmp.json")
    for stale in (temporary_audit, temporary_done):
        if stale.exists():
            stale.unlink()

    started = time.monotonic()
    arrays = _WORKER_GENERATOR.generate([trajectory], str(temporary_audit))
    _atomic_array_save(arrays, label_path)
    os.replace(str(temporary_audit), str(audit_path))
    elapsed = time.monotonic() - started
    marker = {
        "track_id": int(trajectory.track_id),
        "valid_label_count": int(len(arrays.inputs)),
        "elapsed_s": float(elapsed),
    }
    temporary_done.write_text(json.dumps(marker), encoding="utf-8")
    os.replace(str(temporary_done), str(done_path))
    return (
        int(trajectory.track_id),
        str(label_path),
        str(audit_path),
        int(len(arrays.inputs)),
        float(elapsed),
        False,
    )


def _completed_checkpoint(part_directory, trajectory):
    label_path, audit_path, done_path = _part_paths(
        part_directory, trajectory.track_id
    )
    if not (label_path.is_file() and audit_path.is_file() and done_path.is_file()):
        return None
    try:
        marker = json.loads(done_path.read_text(encoding="utf-8"))
        if int(marker["track_id"]) != int(trajectory.track_id):
            return None
        arrays = load_generated_label_arrays(str(label_path))
        declared_period = trajectory.nominal_detection_period_s
        if declared_period is None or not np.isclose(
            float(arrays.nominal_detection_period_s),
            float(declared_period),
            rtol=1e-6,
            atol=1e-9,
        ):
            return None
        if len(arrays.track_ids) and not np.all(
            arrays.track_ids == int(trajectory.track_id)
        ):
            return None
        if int(marker["valid_label_count"]) != len(arrays.inputs):
            return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return (
        int(trajectory.track_id),
        str(label_path),
        str(audit_path),
        int(len(arrays.inputs)),
        float(marker.get("elapsed_s", 0.0)),
        True,
    )


def _print_progress(done, total, valid_count, started, newly_finished, resumed_count):
    elapsed = max(time.monotonic() - started, 1e-9)
    if newly_finished:
        rate = newly_finished / elapsed
        remaining_new = max(total - resumed_count - newly_finished, 0)
        eta = remaining_new / max(rate, 1e-12)
        eta_text = "{:.1f} min".format(eta / 60.0)
    else:
        rate = 0.0
        eta_text = "pending"
    print(
        "[labels] tracks={}/{} ({:.1f}%) valid_labels={} "
        "new_rate={:.3f} tracks/s eta={}".format(
            done,
            total,
            100.0 * done / max(total, 1),
            valid_count,
            rate,
            eta_text,
        ),
        flush=True,
    )


def generate_labels_parallel(
    config,
    trajectories: List,
    output_path: str,
    audit_path: str,
    workers: int,
    resume: bool = True,
    keep_parts: bool = False,
):
    """Generate one atomic checkpoint per track and merge after completion."""
    nominal_detection_period_s = 1.0 / float(
        config["timing"]["detection_rate_hz"]
    )
    # Fail before creating/resuming part files when a legacy 50 ms archive is
    # accidentally supplied to the 100 ms Cage-data configuration.
    validate_trajectory_detection_period(
        trajectories, nominal_detection_period_s
    )
    if int(workers) < 1:
        raise ValueError("workers must be at least one")
    if config["labeling"]["feasibility_backend"] == "moveit":
        raise ValueError(
            "MoveIt multiprocessing requires explicit per-worker ROS namespaces"
        )

    part_directory = Path(output_path).with_suffix(
        Path(output_path).suffix + ".parts"
    )
    if not resume and part_directory.exists():
        shutil.rmtree(part_directory)
    part_directory.mkdir(parents=True, exist_ok=True)

    completed = []
    pending = []
    if resume:
        for trajectory in trajectories:
            checkpoint = _completed_checkpoint(part_directory, trajectory)
            if checkpoint is None:
                pending.append(trajectory)
            else:
                completed.append(checkpoint)
    else:
        pending = list(trajectories)

    total = len(trajectories)
    resumed_count = len(completed)
    valid_count = sum(item[3] for item in completed)
    started = time.monotonic()
    if resumed_count:
        print(
            "[labels] resumed {} completed tracks with {} valid labels".format(
                resumed_count, valid_count
            ),
            flush=True,
        )

    worker_count = min(int(workers), max(len(pending), 1))
    newly_finished = 0
    progress_interval = max(1, total // 100)

    def record(result):
        nonlocal valid_count, newly_finished
        completed.append(result)
        newly_finished += 1
        valid_count += result[3]
        if (
            newly_finished == 1
            or len(completed) == total
            or len(completed) % progress_interval == 0
        ):
            _print_progress(
                len(completed),
                total,
                valid_count,
                started,
                newly_finished,
                resumed_count,
            )

    if pending and worker_count == 1:
        _initialize_worker(config)
        for trajectory in pending:
            record(_label_one_track(trajectory, str(part_directory)))
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(config,),
        ) as executor:
            futures = {
                executor.submit(_label_one_track, trajectory, str(part_directory)):
                trajectory.track_id
                for trajectory in pending
            }
            for future in as_completed(futures):
                record(future.result())

    completed.sort(key=lambda item: item[0])
    arrays = [load_generated_label_arrays(item[1]) for item in completed]
    nonempty = [item for item in arrays if len(item.inputs)]
    if nonempty:
        if any(
            not np.isclose(
                float(item.nominal_detection_period_s),
                nominal_detection_period_s,
                rtol=1e-6,
                atol=1e-9,
            )
            for item in nonempty
        ):
            raise ValueError("Label parts contain mixed detection periods")
        if isinstance(nonempty[0], CageStateArcDatasetArrays):
            if not all(isinstance(item, CageStateArcDatasetArrays) for item in nonempty):
                raise ValueError("Label parts contain mixed dataset contracts")
            merged = CageStateArcDatasetArrays.concatenate(
                nonempty, nominal_detection_period_s
            )
        else:
            if not all(isinstance(item, CageDatasetArrays) for item in nonempty):
                raise ValueError("Label parts contain mixed dataset contracts")
            merged = CageDatasetArrays(
                np.concatenate([item.inputs for item in nonempty]),
                np.concatenate([item.labels for item in nonempty]),
                np.concatenate([item.track_ids for item in nonempty]),
                np.concatenate([item.label_timestamps for item in nonempty]),
                nominal_detection_period_s,
            ).validate()
    else:
        merged = _empty_arrays(config, nominal_detection_period_s)

    output_destination = Path(output_path)
    output_destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_array_save(merged, output_destination)

    audit_destination = Path(audit_path)
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit_destination.with_name(
        audit_destination.name.replace(".jsonl.gz", ".tmp.jsonl.gz")
    )
    with open_audit_text(temporary_audit, "w") as destination:
        for _, _, part_audit, _, _, _ in completed:
            with open_audit_text(part_audit, "r") as source:
                for line in source:
                    destination.write(line)
    os.replace(str(temporary_audit), str(audit_destination))

    if not keep_parts:
        shutil.rmtree(part_directory)
    print(
        "[labels] complete tracks={} valid_labels={} elapsed={:.1f} min".format(
            total, len(merged.inputs), (time.monotonic() - started) / 60.0
        ),
        flush=True,
    )
    return merged, completed
