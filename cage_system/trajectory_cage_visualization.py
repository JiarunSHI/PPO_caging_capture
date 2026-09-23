"""Whole-trajectory audit view with several Cage/object pose pairs."""

from pathlib import Path

import numpy as np

from cage_system.cage.candidates import transform_from_config
from cage_system.geometry import pose6_to_transform
from cage_system.io import load_generated_trajectories
from cage_system.learning.dataset import CageDatasetArrays


def _trajectory_position(trajectory, timestamp):
    return np.asarray(
        [
            np.interp(timestamp, trajectory.timestamps, trajectory.true_positions[:, axis])
            for axis in range(3)
        ],
        dtype=float,
    )


def _draw_leaf(axis, transform_base_endpoint, local_vertices, color, alpha):
    rotation = transform_base_endpoint[:3, :3]
    origin = transform_base_endpoint[:3, 3]
    vertices = np.asarray(local_vertices, dtype=float) @ rotation.T + origin
    closed = np.vstack((vertices, vertices[0]))
    axis.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, alpha=alpha)
    axis.plot_surface(
        vertices[:, 0].reshape(2, 2),
        vertices[:, 1].reshape(2, 2),
        vertices[:, 2].reshape(2, 2),
        color=color,
        alpha=0.08,
        shade=False,
    )


def _draw_cage(axis, transform_base_endpoint, region, color):
    x0, x1 = float(region["rear_x"]), float(region["mouth_x"])
    y0, y1 = float(region["inner_y_min"]), float(region["inner_y_max"])
    z0, z1 = float(region["inner_z_min"]), float(region["inner_z_max"])
    for y in (y0, y1):
        _draw_leaf(
            axis,
            transform_base_endpoint,
            [[x0, y, z0], [x0, y, z1], [x1, y, z0], [x1, y, z1]],
            color,
            0.72,
        )
    _draw_leaf(
        axis,
        transform_base_endpoint,
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1]],
        color,
        0.72,
    )


def visualize_trajectory_cages(
    config,
    trajectory_path,
    label_path,
    output_path,
    track_id=-1,
    pose_count=8,
):
    """Draw a complete target path and uniformly selected Cage label poses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    trajectories = load_generated_trajectories(trajectory_path)
    labels = CageDatasetArrays.load(label_path)
    by_track = {int(item.track_id): item for item in trajectories}
    available_tracks = [
        int(item) for item in np.unique(labels.track_ids) if int(item) in by_track
    ]
    if not available_tracks:
        raise ValueError("No label track is present in the trajectory archive")
    selected_track = available_tracks[0] if int(track_id) < 0 else int(track_id)
    if selected_track not in available_tracks:
        raise ValueError("Requested track has no Cage labels")
    rows = np.flatnonzero(labels.track_ids == selected_track)
    count = min(max(1, int(pose_count)), len(rows))
    selected_rows = rows[np.unique(np.linspace(0, len(rows) - 1, count).astype(int))]
    trajectory = by_track[selected_track]
    transform_tcp_endpoint = transform_from_config(config["gripper"]["T_tcp_endpoint"])
    region = config["gripper"]["internal_region_endpoint_frame"]
    centered = np.asarray(trajectory.true_positions) - np.mean(
        trajectory.true_positions, axis=0
    )
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) >= 2 and singular_values[1] > 1e-4 * singular_values[0]:
        plane_normal = right_vectors[-1]
        plane_reference_source = "plane_fitted_from_complete_curved_trajectory"
    else:
        # A line has infinitely many valid motion planes.  Use the first stable
        # Cage z branch as the reference and audit temporal plane continuity.
        first_transform = (
            pose6_to_transform(labels.labels[selected_rows[0]])
            @ transform_tcp_endpoint
        )
        plane_normal = first_transform[:3, 2]
        plane_reference_source = "first_cage_z_for_straight_line_degeneracy"
    plane_normal = np.asarray(plane_normal, dtype=float)
    plane_normal /= np.linalg.norm(plane_normal)

    figure = plt.figure(figsize=(14, 10))
    axis = figure.add_subplot(111, projection="3d")
    positions = np.asarray(trajectory.true_positions, dtype=float)
    axis.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color="0.28", linewidth=2.4, label="complete target trajectory",
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(selected_rows)))
    alignment_errors = []
    plane_errors = []
    distances = []
    for order, (row, color) in enumerate(zip(selected_rows, colors)):
        timestamp = float(labels.label_timestamps[row])
        object_position = _trajectory_position(trajectory, timestamp)
        transform_base_tcp = pose6_to_transform(labels.labels[row])
        transform_base_endpoint = transform_base_tcp @ transform_tcp_endpoint
        endpoint = transform_base_endpoint[:3, 3]
        rotation = transform_base_endpoint[:3, :3]
        direction = object_position - endpoint
        distance = float(np.linalg.norm(direction))
        unit_direction = direction / max(distance, 1e-12)
        alignment_errors.append(1.0 - float(np.dot(rotation[:, 0], unit_direction)))
        plane_errors.append(
            max(abs(float(np.dot(rotation[:, 0], plane_normal))),
                abs(float(np.dot(rotation[:, 1], plane_normal))))
        )
        distances.append(distance)
        _draw_cage(axis, transform_base_endpoint, region, color)
        axis.scatter(*object_position, color=color, s=44, edgecolors="black", linewidths=0.4)
        axis.plot(
            [endpoint[0], object_position[0]],
            [endpoint[1], object_position[1]],
            [endpoint[2], object_position[2]],
            linestyle=":", color=color, linewidth=1.3,
        )
        axis.quiver(*endpoint, *rotation[:, 0], length=min(0.035, 0.35 * distance), color="r")
        axis.text(*object_position, "  {}".format(order + 1), color=color, fontsize=9)

    all_points = np.vstack((positions, np.asarray([
        _trajectory_position(trajectory, labels.label_timestamps[row])
        for row in selected_rows
    ])))
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    span = max(0.12, float(np.max(all_points.max(axis=0) - all_points.min(axis=0))))
    half = 0.62 * span
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Base x [m]")
    axis.set_ylabel("Base y [m]")
    axis.set_zlabel("Base z [m]")
    axis.set_title(
        "track {}: complete trajectory with {} Cage/object pose pairs".format(
            selected_track, len(selected_rows)
        )
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color="0.28", lw=2.4, label="complete target trajectory"),
            Line2D([0], [0], color="tab:blue", lw=1.4, label="Cage leaves / rear boundary"),
            Line2D([0], [0], color="r", lw=1.4, label="Cage +x toward object"),
        ],
        loc="upper left",
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(destination), dpi=180)
    plt.close(figure)
    return {
        "output": str(destination),
        "track_id": selected_track,
        "dataset_rows": selected_rows.tolist(),
        "pose_count": int(len(selected_rows)),
        "plane_reference_source": plane_reference_source,
        "candidate_distance_m_min_max": [float(min(distances)), float(max(distances))],
        "maximum_x_to_object_alignment_error": float(max(alignment_errors)),
        "maximum_xy_plane_parallel_error": float(max(plane_errors)),
    }
