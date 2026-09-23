import json
from pathlib import Path

import numpy as np

from cage_system.audit_io import open_audit_text


def load_audit_record(path: str, track_id: int, sample_index: int):
    with open_audit_text(path, "r") as handle:
        for line in handle:
            record = json.loads(line)
            if record["track_id"] == track_id and record["sample_index"] == sample_index:
                return record
    raise KeyError("Audit record not found")


def visualize_audit_record(record, config, output_path=None):
    try:
        if output_path:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, registers old Matplotlib 3-D
    except ImportError as error:
        raise RuntimeError("Install matplotlib to use Cage visualization") from error
    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")
    current = np.asarray(record["current_position"])
    tangent = np.asarray(record["tangent"])
    normal = np.asarray(record["plane_normal"])
    radius = float(record["radius"])
    axis.scatter(*current, color="black", s=55, label="current object")
    axis.quiver(*current, *tangent, length=0.7 * radius, color="black", label="tangent")
    axis.quiver(*current, *normal, length=0.5 * radius, color="purple", label="plane normal")

    if "future_curve" in record:
        future = np.asarray(record["future_curve"])
        axis.plot(future[:, 0], future[:, 1], future[:, 2], color="tab:green", label="future curve")
        circle = np.asarray(record["candidate_circle"])
        arc = np.asarray(record["candidate_arc"])
        axis.plot(circle[:, 0], circle[:, 1], circle[:, 2], "--", color="tab:blue", label="candidate circle")
        axis.plot(arc[:, 0], arc[:, 1], arc[:, 2], linewidth=4, color="tab:orange", label="minor arc")
        tangent_point = np.asarray(record["tangent_intersection"])
        future_point = np.asarray(record["future_intersection"])
        axis.scatter(*tangent_point, color="blue", s=45, label="tangent intersection")
        axis.scatter(*future_point, color="green", s=45, label="future intersection")

        for candidate in record.get("candidates", []):
            position = np.asarray(candidate["endpoint_position"])
            rotation = np.asarray(candidate["rotation_matrix"])
            color = "tab:red" if candidate.get("candidate_id") == record.get("best_candidate_id") else "0.65"
            axis.scatter(*position, color=color, s=18)
            for frame_axis, axis_color in zip(rotation.T, ("r", "g", "b")):
                axis.quiver(*position, *frame_axis, length=0.025, color=axis_color, alpha=0.6)

    nodes = np.asarray(record.get("best_est_nodes_base", []), dtype=float)
    statuses = np.asarray(record.get("best_est_node_status", []), dtype=int)
    if len(nodes):
        colors = np.array(["tab:cyan", "tab:red", "tab:purple"])
        parents = np.asarray(
            record.get("best_est_node_parent_indices", np.full(len(nodes), -1)),
            dtype=int,
        )
        for index, parent in enumerate(parents):
            if parent >= 0 and parent < len(nodes):
                axis.plot(
                    nodes[[parent, index], 0],
                    nodes[[parent, index], 1],
                    nodes[[parent, index], 2],
                    color=colors[statuses[index]],
                    linewidth=0.45,
                    alpha=0.22,
                )
        axis.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=colors[statuses], s=8, alpha=0.45, label="EST nodes")
        _draw_internal_region(axis, record, config)

    extent_points = [current[None, :]]
    for key in ("candidate_circle", "future_curve", "best_est_nodes_base"):
        if key in record and len(record[key]):
            extent_points.append(np.asarray(record[key]))
    all_points = np.concatenate(extent_points)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    half = max(0.08, 0.55 * float(np.max(all_points.max(axis=0) - all_points.min(axis=0))))
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.set_xlabel("Base x [m]")
    axis.set_ylabel("Base y [m]")
    axis.set_zlabel("Base z [m]")
    axis.set_title(
        "track={} sample={} valid={} reason={}".format(
            record["track_id"], record["sample_index"], record["valid"], record.get("invalid_reason", "")
        )
    )
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(destination), dpi=180)
        plt.close(figure)
    else:
        plt.show()


def _draw_internal_region(axis, record, config):
    best_id = record.get("best_candidate_id")
    candidate = next(
        (item for item in record.get("candidates", []) if item["candidate_id"] == best_id), None
    )
    if candidate is None:
        return
    region = config["gripper"]["internal_region_endpoint_frame"]
    x0, x1 = region["rear_x"], region["mouth_x"]
    y0, y1 = region["inner_y_min"], region["inner_y_max"]
    z0, z1 = region["inner_z_min"], region["inner_z_max"]
    local = np.array(
        [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=float
    )
    rotation = np.asarray(candidate["rotation_matrix"])
    origin = np.asarray(candidate["endpoint_position"])
    points = local @ rotation.T + origin
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
        (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    for first, second in edges:
        # Mouth rectangle is dashed: it is an escape boundary, not a solid Cap.
        mouth_edge = local[first, 0] == x1 and local[second, 0] == x1
        style = "--" if mouth_edge else "-"
        axis.plot(
            points[[first, second], 0],
            points[[first, second], 1],
            points[[first, second], 2],
            linestyle=style,
            color="tab:brown",
            linewidth=1.5,
        )
