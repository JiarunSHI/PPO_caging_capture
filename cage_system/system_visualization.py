"""Deterministic audit rendering for the complete UR5e/eye-on-hand/Cage stack."""

from pathlib import Path

import numpy as np

from cage_system.geometry import pose6_to_transform
from cage_system.io import load_generated_trajectories
from cage_system.learning.dataset import CageDatasetArrays
from cage_system.perception.eye_on_hand import MujocoEyeOnHandCamera
from cage_system.sim.mujoco_feasibility import MujocoFeasibilityBackend


def _trajectory_position(trajectory, timestamp):
    return np.asarray(
        [
            np.interp(timestamp, trajectory.timestamps, trajectory.true_positions[:, axis])
            for axis in range(3)
        ],
        dtype=float,
    )


def _add_capsule(scene, mujoco, start, end, radius, rgba):
    """Append one non-physical audit capsule to an already populated MjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.asarray([radius, radius, radius], dtype=float),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    scene.ngeom += 1


def _add_trajectory_to_scene(
    scene, mujoco, trajectory, current_time, radius=0.0024, vivid=False
):
    points = np.asarray(trajectory.true_positions, dtype=float)
    times = np.asarray(trajectory.timestamps, dtype=float)
    if len(points) > 90:
        indices = np.unique(np.linspace(0, len(points) - 1, 90).astype(int))
        points = points[indices]
        times = times[indices]
    for index in range(len(points) - 1):
        if times[index + 1] <= current_time:
            color = (0.10, 0.85, 1.00, 0.92) if vivid else (0.70, 0.75, 0.80, 0.55)
        else:
            color = (1.00, 0.74, 0.08, 0.90)
        _add_capsule(scene, mujoco, points[index], points[index + 1], radius, color)


def _add_trajectory_interval_to_scene(
    scene,
    mujoco,
    trajectory,
    start_time,
    end_time,
    radius=0.0065,
    rgba=(0.95, 0.16, 0.70, 1.0),
):
    """Highlight the complete-trajectory subsection used by a random Cage view."""
    points = np.asarray(trajectory.true_positions, dtype=float)
    times = np.asarray(trajectory.timestamps, dtype=float)
    for index in range(len(points) - 1):
        if times[index + 1] < start_time or times[index] > end_time:
            continue
        segment_start = max(float(times[index]), float(start_time))
        segment_end = min(float(times[index + 1]), float(end_time))
        if segment_end <= segment_start:
            continue
        denominator = max(float(times[index + 1] - times[index]), 1e-12)
        first_weight = (segment_start - float(times[index])) / denominator
        second_weight = (segment_end - float(times[index])) / denominator
        first = (1.0 - first_weight) * points[index] + first_weight * points[index + 1]
        second = (1.0 - second_weight) * points[index] + second_weight * points[index + 1]
        _add_capsule(scene, mujoco, first, second, radius, rgba)


def _random_contiguous_solution_segment(solutions, pose_count, random_state):
    """Select a reproducible contiguous slice from time-ordered IK solutions."""
    count = min(max(1, int(pose_count)), len(solutions))
    maximum_start = len(solutions) - count
    start = 0 if maximum_start <= 0 else int(random_state.randint(maximum_start + 1))
    return solutions[start : start + count]


def _split_time_contiguous_solutions(solutions, label_timestamps, maximum_gap_s):
    """Split feasible labels whenever visual-time continuity is interrupted."""
    if not solutions:
        return []
    runs = [[solutions[0]]]
    for solution in solutions[1:]:
        previous_time = float(label_timestamps[runs[-1][-1][0]])
        current_time = float(label_timestamps[solution[0]])
        if current_time - previous_time <= float(maximum_gap_s) + 1e-9:
            runs[-1].append(solution)
        else:
            runs.append([solution])
    return runs


def _draw_timing_panel(axis, detection_rate_hz):
    vision_ms = 1000.0 / float(detection_rate_hz)
    events = np.asarray([0.0, vision_ms, 2.0 * vision_ms])
    axis.set_xlim(-3.0, 2.0 * vision_ms + 4.0)
    axis.set_ylim(-0.65, 2.35)
    axis.set_yticks([0.0, 1.0, 2.0])
    axis.set_yticklabels(["low-level tracking", "PPO + publish", "20 Hz RGB-D"])
    axis.set_xlabel("visual-event time [ms]")
    axis.grid(axis="x", alpha=0.22)
    axis.axvspan(0.0, vision_ms, ymin=0.08, ymax=0.25, color="#4c78a8", alpha=0.55)
    axis.axvspan(
        vision_ms,
        2.0 * vision_ms,
        ymin=0.08,
        ymax=0.25,
        color="#f58518",
        alpha=0.65,
    )
    axis.text(vision_ms * 0.5, 0.0, "track command u[k]", ha="center", va="center", fontsize=9)
    axis.text(
        1.5 * vision_ms,
        0.0,
        "track command u[k+1]",
        ha="center",
        va="center",
        fontsize=8,
    )
    axis.vlines(events, 1.68, 2.30, color="#e45756", linewidth=2.6)
    axis.scatter(events, np.full_like(events, 2.0), color="#e45756", s=34, zorder=3)
    ppo_events = events + 0.9
    axis.vlines(ppo_events, 0.72, 1.28, color="#2f4b7c", linewidth=2.2)
    axis.scatter(ppo_events, np.ones_like(ppo_events), color="#2f4b7c", s=26, zorder=3)
    axis.annotate(
        "detect → Cage → PPO → publish immediately",
        xy=(vision_ms + 0.9, 1.0),
        xytext=(vision_ms, 1.42),
        arrowprops={"arrowstyle": "->", "color": "#e45756"},
        fontsize=8,
        ha="center",
    )
    axis.set_title("Event-driven loop: exactly one PPO decision per completed visual update")


def render_complete_system(
    config,
    model_path,
    trajectory_path,
    label_path,
    output_path,
    sample_index=-1,
    detector_device="auto",
):
    """Render one real offline label with its complete robot and wrist-camera views."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    trajectories = load_generated_trajectories(trajectory_path)
    labels = CageDatasetArrays.load(label_path)
    if not len(labels.inputs):
        raise ValueError("The label dataset is empty")
    by_track = {int(item.track_id): item for item in trajectories}
    if sample_index >= len(labels.inputs):
        raise IndexError("sample_index is outside the label dataset")

    backend = MujocoFeasibilityBackend(
        model_path, config["labeling"]["mujoco_tcp_site"], config["robot"]
    )
    nominal = np.asarray(config["robot"]["nominal_joint_state_rad"], dtype=float)
    search = [int(sample_index)] if sample_index >= 0 else list(range(len(labels.inputs)))
    selected = None
    ik_result = None
    for index in search:
        result = backend.solve(pose6_to_transform(labels.labels[index]), nominal)
        if result.reachable and result.collision_free and result.joints is not None:
            selected = index
            ik_result = result
            break
    if selected is None:
        raise RuntimeError("No requested label has a collision-free IK solution from the nominal seed")

    track_id = int(labels.track_ids[selected])
    if track_id not in by_track:
        raise KeyError("The selected label track is absent from the trajectory archive")
    trajectory = by_track[track_id]
    timestamp = float(labels.label_timestamps[selected])
    truth = _trajectory_position(trajectory, timestamp)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in config["robot"]["joint_names"]
        ]
    )
    qpos_addresses = model.jnt_qposadr[joint_ids]
    data.qpos[qpos_addresses] = ik_result.joints
    actuator_joint_ids = model.actuator_trnid[:, 0]
    for joint_id, joint_value in zip(joint_ids, ik_result.joints):
        matches = np.flatnonzero(actuator_joint_ids == joint_id)
        if len(matches):
            data.ctrl[matches[0]] = joint_value
    target_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, config["vision"]["target_mocap_body"]
    )
    target_mocap_id = int(model.body_mocapid[target_body])
    data.mocap_pos[target_mocap_id] = truth
    mujoco.mj_forward(model, data)

    vision_config = dict(config["vision"])
    vision_config.update(
        {
            "frame_dropout_probability": 0.0,
            "depth_pixel_dropout_probability": 0.0,
            "depth_noise_std_m": 0.0,
        }
    )
    camera = MujocoEyeOnHandCamera(
        model, vision_config, seed=0, detector_device=detector_device
    )
    try:
        detection = camera.capture(data, timestamp, selected)
    finally:
        camera.close()

    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    # mjVIS_CAMERA draws every camera frustum, including the active 960x720
    # overview camera; its near plane can cover the whole image.  The physical
    # wrist housing and yellow optical-axis geom identify the eye camera instead.
    option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
    overview_renderer = mujoco.Renderer(model, height=720, width=960, max_geom=10000)
    try:
        overview_renderer.update_scene(data, camera="overview", scene_option=option)
        _add_trajectory_to_scene(overview_renderer.scene, mujoco, trajectory, timestamp)
        overview = overview_renderer.render().copy()
        overview_renderer.update_scene(data, camera="cage_closeup", scene_option=option)
        cage_closeup = overview_renderer.render().copy()
    finally:
        overview_renderer.close()

    center_error = (
        float(np.linalg.norm(detection.observation.position - truth))
        if detection.detected
        else float("inf")
    )
    target_pose = labels.labels[selected]
    figure = plt.figure(figsize=(18, 10.5), constrained_layout=True)
    grid = figure.add_gridspec(
        3, 4, width_ratios=[1.35, 1.35, 1.0, 1.0], height_ratios=[1.0, 1.0, 0.72]
    )
    overview_axis = figure.add_subplot(grid[:, :2])
    cage_axis = figure.add_subplot(grid[0, 2])
    rgb_axis = figure.add_subplot(grid[0, 3])
    depth_axis = figure.add_subplot(grid[1, 2])
    mask_axis = figure.add_subplot(grid[1, 3])
    timing_axis = figure.add_subplot(grid[2, 2:])

    overview_axis.imshow(overview)
    overview_axis.set_title("Complete UR5e + eye-on-hand + physical Cage at selected label")
    overview_axis.axis("off")
    overview_axis.text(
        0.015,
        0.02,
        "TCP axes: x red / y green / z blue\n"
        "camera optical axis: yellow\n"
        "Cage interior: translucent cyan\n"
        "trajectory: gray history / yellow future",
        transform=overview_axis.transAxes,
        color="white",
        fontsize=10,
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 7},
    )
    cage_axis.imshow(cage_closeup)
    cage_axis.set_title("Cage close-up: geometry + TCP axes + target")
    rgb_axis.imshow(detection.rgb)
    rgb_axis.set_title("eye-on-hand RGB")
    depth_image = depth_axis.imshow(detection.depth_m, cmap="viridis")
    depth_axis.set_title("eye-on-hand depth [m]")
    figure.colorbar(depth_image, ax=depth_axis, fraction=0.046, pad=0.04)
    mask_axis.imshow(detection.detection_mask, cmap="gray")
    mask_axis.set_title(
        "detector mask\ncenter error {:.1f} mm".format(1000.0 * center_error)
        if detection.detected
        else "detector mask\nnot detected"
    )
    for axis in (cage_axis, rgb_axis, depth_axis, mask_axis):
        axis.axis("off")
    _draw_timing_panel(
        timing_axis,
        config["timing"]["detection_rate_hz"],
    )
    figure.suptitle(
        "track {}  |  t={:.3f}s  |  dataset row {}  |  Cage TCP label "
        "xyz=({:.3f}, {:.3f}, {:.3f}) m, rotvec=({:.2f}, {:.2f}, {:.2f}) rad".format(
            track_id, timestamp, selected, *target_pose
        ),
        fontsize=12,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(destination), dpi=170)
    plt.close(figure)
    return {
        "output": str(destination),
        "dataset_row": int(selected),
        "track_id": track_id,
        "timestamp_s": timestamp,
        "cage_tcp_pose6": target_pose.tolist(),
        "ik_joints_rad": ik_result.joints.tolist(),
        "vision_detected": bool(detection.detected),
        "vision_mask_pixels": int(detection.mask_pixels),
        "vision_center_error_m": center_error,
        "detection_period_s": 1.0 / float(config["timing"]["detection_rate_hz"]),
        "ppo_decision_trigger": "one_immediate_action_per_completed_visual_update",
    }


def render_trajectory_robot_cages(
    config,
    model_path,
    trajectory_path,
    label_path,
    output_path,
    track_id=-1,
    pose_count=6,
    random_segment=False,
    random_seed=42,
    include_eye_on_hand=False,
):
    """Render full UR5e/Cage configurations along one complete or partial track."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    trajectories = load_generated_trajectories(trajectory_path)
    labels = CageDatasetArrays.load(label_path)
    by_track = {int(item.track_id): item for item in trajectories}
    backend = MujocoFeasibilityBackend(
        model_path, config["labeling"]["mujoco_tcp_site"], config["robot"]
    )
    nominal = np.asarray(config["robot"]["nominal_joint_state_rad"], dtype=float)

    requested_tracks = (
        [int(track_id)]
        if int(track_id) >= 0
        else [int(item) for item in np.unique(labels.track_ids)]
    )
    random_state = np.random.RandomState(int(random_seed))
    if random_segment and int(track_id) < 0:
        requested_tracks = random_state.permutation(requested_tracks).tolist()
    solutions_by_track = {}
    for candidate_track in requested_tracks:
        if candidate_track not in by_track:
            continue
        seed = nominal.copy()
        solutions = []
        for row in np.flatnonzero(labels.track_ids == candidate_track):
            result = backend.solve(pose6_to_transform(labels.labels[row]), seed)
            if result.reachable and result.collision_free and result.joints is not None:
                solutions.append((int(row), result.joints.copy()))
                seed = result.joints.copy()
        if solutions:
            solutions_by_track[candidate_track] = solutions
            # For the random command, the shuffled first feasible track is the
            # selection. Avoid solving IK for every track in a large archive.
            if random_segment and int(track_id) < 0:
                break
    if not solutions_by_track:
        raise RuntimeError("No track has a collision-free UR5e IK solution")
    selected_track = (
        next(iter(solutions_by_track))
        if random_segment
        else max(solutions_by_track, key=lambda key: len(solutions_by_track[key]))
    )
    solutions = solutions_by_track[selected_track]
    if random_segment:
        maximum_gap_s = 1.75 / float(config["timing"]["detection_rate_hz"])
        contiguous_runs = _split_time_contiguous_solutions(
            solutions, labels.label_timestamps, maximum_gap_s
        )
        desired_count = max(1, int(pose_count))
        eligible_runs = [run for run in contiguous_runs if len(run) >= desired_count]
        if eligible_runs:
            solutions = eligible_runs[int(random_state.randint(len(eligible_runs)))]
        else:
            maximum_length = max(len(run) for run in contiguous_runs)
            longest_runs = [run for run in contiguous_runs if len(run) == maximum_length]
            solutions = longest_runs[int(random_state.randint(len(longest_runs)))]
        selected = _random_contiguous_solution_segment(
            solutions, pose_count, random_state
        )
    else:
        count = min(max(1, int(pose_count)), len(solutions))
        selected_indices = np.unique(
            np.linspace(0, len(solutions) - 1, count).astype(int)
        )
        selected = [solutions[index] for index in selected_indices]
    trajectory = by_track[selected_track]
    segment_start_time = float(labels.label_timestamps[selected[0][0]])
    segment_end_time = float(labels.label_timestamps[selected[-1][0]])

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in config["robot"]["joint_names"]
        ]
    )
    qpos_addresses = model.jnt_qposadr[joint_ids]
    target_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, config["vision"]["target_mocap_body"]
    )
    target_mocap_id = int(model.body_mocapid[target_body])
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
    renderer = mujoco.Renderer(model, height=480, width=640, max_geom=10000)
    frames = []
    tcp_errors = []
    try:
        for row, joints in selected:
            mujoco.mj_resetData(model, data)
            data.qpos[qpos_addresses] = joints
            timestamp = float(labels.label_timestamps[row])
            truth = _trajectory_position(trajectory, timestamp)
            data.mocap_pos[target_mocap_id] = truth
            mujoco.mj_forward(model, data)
            tcp_site_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                config["labeling"]["mujoco_tcp_site"],
            )
            tcp_errors.append(
                float(
                    np.linalg.norm(
                        data.site_xpos[tcp_site_id]
                        - pose6_to_transform(labels.labels[row])[:3, 3]
                    )
                )
            )
            renderer.update_scene(data, camera="overview", scene_option=option)
            _add_trajectory_to_scene(
                renderer.scene,
                mujoco,
                trajectory,
                timestamp,
                radius=0.005,
                vivid=True,
            )
            if random_segment:
                _add_trajectory_interval_to_scene(
                    renderer.scene,
                    mujoco,
                    trajectory,
                    segment_start_time,
                    segment_end_time,
                )
            overview = renderer.render().copy()
            eye_view = None
            if include_eye_on_hand:
                renderer.update_scene(
                    data,
                    camera=config["vision"]["camera_name"],
                    scene_option=option,
                )
                eye_view = renderer.render().copy()
            frames.append((row, timestamp, overview, eye_view))
    finally:
        renderer.close()

    if include_eye_on_hand:
        figure, axes = plt.subplots(
            len(frames), 2, figsize=(14.5, 4.5 * len(frames)), squeeze=False
        )
        for order, (row, timestamp, overview, eye_view) in enumerate(frames, start=1):
            axes[order - 1, 0].imshow(overview)
            axes[order - 1, 0].set_title(
                "MuJoCo overview {} | t={:.3f}s | label row {}".format(
                    order, timestamp, row
                )
            )
            axes[order - 1, 1].imshow(eye_view)
            axes[order - 1, 1].set_title(
                "rigid eye-on-hand RGB at the same state"
            )
            axes[order - 1, 0].axis("off")
            axes[order - 1, 1].axis("off")
    else:
        if len(frames) in (3, 5, 6):
            columns = 3
        else:
            columns = 2 if len(frames) > 1 else 1
        rows = int(np.ceil(len(frames) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(8.2 * columns, 5.8 * rows))
        axes = np.atleast_1d(axes).reshape(-1)
        for order, (axis, (row, timestamp, overview, _)) in enumerate(
            zip(axes, frames), start=1
        ):
            axis.imshow(overview)
            axis.set_title(
                "pose {}  |  t={:.3f}s  |  label row {}".format(
                    order, timestamp, row
                )
            )
            axis.axis("off")
        for axis in axes[len(frames) :]:
            axis.axis("off")
    selection_title = (
        "random contiguous segment {:.3f}-{:.3f}s (magenta)".format(
            segment_start_time, segment_end_time
        )
        if random_segment
        else "uniform poses over the complete trajectory"
    )
    figure.suptitle(
        "track {}: full UR5e + rigid eye-on-hand + Cage + object + trajectory\n"
        "{}; cyan = history, yellow = future, red sphere = object".format(
            selected_track, selection_title
        ),
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(destination), dpi=160)
    plt.close(figure)
    return {
        "output": str(destination),
        "track_id": int(selected_track),
        "dataset_rows": [int(item[0]) for item in selected],
        "pose_count": int(len(selected)),
        "selection_mode": (
            "random_contiguous_feasible_segment"
            if random_segment
            else "uniform_complete_track"
        ),
        "random_seed": int(random_seed) if random_segment else None,
        "segment_time_range_s": [segment_start_time, segment_end_time],
        "maximum_selected_time_gap_s": (
            float(
                np.max(
                    np.diff(
                        [float(labels.label_timestamps[item[0]]) for item in selected]
                    )
                )
            )
            if len(selected) > 1
            else 0.0
        ),
        "eye_on_hand_views_rendered": bool(include_eye_on_hand),
        "maximum_tcp_position_error_m": float(max(tcp_errors)),
        "rendered_geometry": (
            "full_ur5e_rigid_link6_cage_camera_target_complete_track_and_selected_segment"
        ),
    }
