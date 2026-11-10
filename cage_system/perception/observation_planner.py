"""Simulator-only initial observation planning; never supplies state to inference."""
import numpy as np
from cage_system.cage.candidates import transform_from_config
from cage_system.perception.eye_on_hand import MujocoEyeOnHandCamera
from cage_system.sim.mujoco_feasibility import MujocoFeasibilityBackend

class InitialObservationPlanner:
    """Own an independent MJCF state for checking initial camera coverage."""
    def __init__(self, config, model_path: str, detector_device="auto"):
        import mujoco

        self.mujoco = mujoco
        self.config = config
        self.nominal_detection_period_s = 1.0 / float(
            config["timing"]["detection_rate_hz"]
        )
        self.model_path = str(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.camera = MujocoEyeOnHandCamera(
            self.model,
            config["vision"],
            seed=int(config["system"]["seed"]),
            detector_device=detector_device,
        )
        self.ik = MujocoFeasibilityBackend(
            self.model_path,
            config["labeling"]["mujoco_tcp_site"],
            config["robot"],
        )
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in config["robot"]["joint_names"]
            ]
        )
        if np.any(self.joint_ids < 0):
            raise ValueError("Vision MJCF is missing configured UR5e joints")
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        self.actuator_ids = np.asarray(
            [np.flatnonzero(actuator_joint_ids == joint_id)[0] for joint_id in self.joint_ids]
        )
        target_body = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            config["vision"]["target_mocap_body"],
        )
        self.target_mocap_id = int(self.model.body_mocapid[target_body])
        self.transform_tcp_endpoint = transform_from_config(
            config["gripper"]["T_tcp_endpoint"]
        )
        self.last_observation_pose_score = None

    def close(self):
        self.camera.close()

    def _trajectory_view_score(self, trajectory, sample_index, lookahead_frames=None):
        """Rank a camera pose by current and near-future truth-point margins."""

        configured_lookahead = (
            int(self.config["vision"].get("observation_pose_lookahead_frames", 0))
            if lookahead_frames is None
            else int(lookahead_frames)
        )
        start = int(np.clip(sample_index, 0, len(trajectory.true_positions) - 1))
        stop = (
            len(trajectory.true_positions)
            if configured_lookahead == 0
            else start + configured_lookahead
        )
        points = np.asarray(
            trajectory.true_positions[start:stop], dtype=float
        )
        camera_position = self.data.cam_xpos[self.camera.camera_id].copy()
        camera_rotation = self.data.cam_xmat[self.camera.camera_id].reshape(3, 3).copy()
        points_camera = (points - camera_position) @ camera_rotation
        depth = -points_camera[:, 2]
        safe_depth = np.maximum(depth, 1e-9)
        columns = self.camera.intrinsics.cx + self.camera.intrinsics.fx * (
            points_camera[:, 0] / safe_depth
        )
        rows = self.camera.intrinsics.cy - self.camera.intrinsics.fy * (
            points_camera[:, 1] / safe_depth
        )
        target_pixel_radius = self.camera.intrinsics.fx * float(
            self.config["vision"]["target_radius_m"]
        ) / safe_depth
        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= float(self.config["vision"]["maximum_depth_m"]))
            & (columns - target_pixel_radius >= 0.0)
            & (columns + target_pixel_radius < float(self.camera.width))
            & (rows - target_pixel_radius >= 0.0)
            & (rows + target_pixel_radius < float(self.camera.height))
        )
        edge_margin = np.minimum.reduce(
            (
                (columns - target_pixel_radius)
                / max(float(self.camera.width - 1), 1.0),
                (float(self.camera.width - 1) - columns - target_pixel_radius)
                / max(float(self.camera.width - 1), 1.0),
                (rows - target_pixel_radius)
                / max(float(self.camera.height - 1), 1.0),
                (float(self.camera.height - 1) - rows - target_pixel_radius)
                / max(float(self.camera.height - 1), 1.0),
            )
        )
        visible_margins = edge_margin[valid]
        minimum_margin = (
            float(np.min(visible_margins)) if len(visible_margins) else -1.0
        )
        mean_margin = (
            float(np.mean(visible_margins)) if len(visible_margins) else -1.0
        )
        return (
            int(bool(valid[0])),
            float(np.mean(valid)),
            minimum_margin,
            mean_margin,
        )
