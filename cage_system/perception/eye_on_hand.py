from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from cage_system.geometry import normalize
from cage_system.types import PointObservation


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class VisionDetection:
    observation: PointObservation
    detected: bool
    confidence: float
    mask_pixels: int
    center_pixel_uv: Optional[np.ndarray]
    rgb: np.ndarray
    depth_m: np.ndarray
    detection_mask: np.ndarray
    simulation_target_mask: Optional[np.ndarray]
    camera_position_base: np.ndarray
    camera_rotation_base: np.ndarray
    dropout_injected: bool
    target_visible_truth: bool


class RedColorMaskDetector:
    """RGB-only detector for the red point-mass proxy used in simulation."""

    def detect(self, rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        image = np.asarray(rgb, dtype=np.float32)
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        mask = (
            (red >= 90.0)
            & (red >= 1.35 * np.maximum(green, 1.0))
            & (red >= 1.35 * np.maximum(blue, 1.0))
            & ((red - np.maximum(green, blue)) >= 28.0)
        )
        if not np.any(mask):
            return mask, 0.0
        dominance = (red[mask] - np.maximum(green[mask], blue[mask])) / 255.0
        return mask, float(np.clip(np.mean(dominance), 0.0, 1.0))


class UltralyticsMaskDetector:
    """Optional COCO segmentation frontend using a local Ultralytics checkpoint."""

    def __init__(self, weights_path: str, class_id: int = -1, confidence: float = 0.35, device="auto"):
        if not weights_path:
            raise ValueError("vision.coco_weights_path is required for the ultralytics backend")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install ultralytics to use the COCO detector backend") from error
        self.model = YOLO(weights_path)
        self.class_id = int(class_id)
        self.confidence = float(confidence)
        self.device = device

    def detect(self, rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        import cv2

        result = self.model.predict(
            source=np.asarray(rgb),
            conf=self.confidence,
            device=None if self.device == "auto" else self.device,
            verbose=False,
        )[0]
        height, width = rgb.shape[:2]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((height, width), dtype=bool), 0.0
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        valid = np.arange(len(classes))
        if self.class_id >= 0:
            valid = valid[classes == self.class_id]
        if len(valid) == 0:
            return np.zeros((height, width), dtype=bool), 0.0
        selected = int(valid[np.argmax(confidences[valid])])
        if result.masks is not None and selected < len(result.masks.data):
            low_resolution = result.masks.data[selected].detach().cpu().numpy()
            mask = cv2.resize(
                low_resolution, (width, height), interpolation=cv2.INTER_NEAREST
            ) > 0.5
        else:
            # Detection-only COCO checkpoints expose boxes but no masks.  The
            # depth validity gate suppresses background pixels inside the box.
            mask = np.zeros((height, width), dtype=bool)
            x0, y0, x1, y1 = result.boxes.xyxy[selected].detach().cpu().numpy()
            x0 = int(np.clip(np.floor(x0), 0, width - 1))
            x1 = int(np.clip(np.ceil(x1), x0 + 1, width))
            y0 = int(np.clip(np.floor(y0), 0, height - 1))
            y1 = int(np.clip(np.ceil(y1), y0 + 1, height))
            mask[y0:y1, x0:x1] = True
        return mask, float(confidences[selected])


def intrinsics_from_vertical_fov(width: int, height: int, vertical_fov_deg: float) -> CameraIntrinsics:
    fy = 0.5 * float(height) / np.tan(0.5 * np.deg2rad(float(vertical_fov_deg)))
    return CameraIntrinsics(
        int(width), int(height), float(fy), float(fy),
        0.5 * (float(width) - 1.0), 0.5 * (float(height) - 1.0),
    )


def unproject_depth_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_position_base: np.ndarray,
    camera_rotation_base: np.ndarray,
) -> np.ndarray:
    """Unproject selected MuJoCo pixels to Base; cameras look along local -z."""

    depth = np.asarray(depth_m, dtype=float)
    selected = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(selected)
    if not len(rows):
        return np.empty((0, 3), dtype=float)
    z_depth = depth[rows, columns]
    x_camera = (columns.astype(float) - intrinsics.cx) * z_depth / intrinsics.fx
    y_camera = -(rows.astype(float) - intrinsics.cy) * z_depth / intrinsics.fy
    points_camera = np.column_stack((x_camera, y_camera, -z_depth))
    return points_camera @ np.asarray(camera_rotation_base, dtype=float).T + np.asarray(
        camera_position_base, dtype=float
    )


class MujocoEyeOnHandCamera:
    """MuJoCo RGB-D renderer whose output is converted to PointObservation."""

    def __init__(self, model, vision_config, seed: int = 0, detector_device="auto"):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("MuJoCo is required for rendered eye-on-hand perception") from error
        self.mujoco = mujoco
        self.model = model
        self.config = vision_config
        self.rng = np.random.RandomState(int(seed))
        self.width = int(vision_config["image_width"])
        self.height = int(vision_config["image_height"])
        self.camera_name = str(vision_config["camera_name"])
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        self.target_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, str(vision_config["target_geom"])
        )
        if self.camera_id < 0 or self.target_geom_id < 0:
            raise ValueError("MJCF is missing the configured eye-on-hand camera or target geom")
        self.intrinsics = intrinsics_from_vertical_fov(
            self.width, self.height, float(model.cam_fovy[self.camera_id])
        )
        backend = str(vision_config["detector_backend"])
        if backend == "red_color":
            self.detector = RedColorMaskDetector()
        elif backend == "ultralytics":
            self.detector = UltralyticsMaskDetector(
                vision_config["coco_weights_path"],
                vision_config["coco_class_id"],
                vision_config["coco_confidence_threshold"],
                detector_device,
            )
        elif backend == "mujoco_segmentation":
            self.detector = None
        else:
            raise ValueError("Unknown vision.detector_backend '{}'".format(backend))
        self.backend = backend
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _render_segmentation(self, data):
        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(data, camera=self.camera_name)
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()

    def _target_center_inside_camera_frustum(
        self, data, camera_position, camera_rotation
    ):
        """Cheap truth audit used when a third segmentation render is disabled."""
        target_position = data.geom_xpos[self.target_geom_id]
        target_camera = np.asarray(camera_rotation, dtype=float).T @ (
            np.asarray(target_position, dtype=float)
            - np.asarray(camera_position, dtype=float)
        )
        depth = -float(target_camera[2])
        if (
            not np.isfinite(depth)
            or depth <= 0.0
            or depth > float(self.config["maximum_depth_m"])
        ):
            return False
        column = self.intrinsics.cx + self.intrinsics.fx * (
            float(target_camera[0]) / depth
        )
        row = self.intrinsics.cy - self.intrinsics.fy * (
            float(target_camera[1]) / depth
        )
        return bool(
            0.0 <= column < float(self.width)
            and 0.0 <= row < float(self.height)
        )

    def capture(self, data, timestamp: float, source_index: int = -1) -> VisionDetection:
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(data, camera=self.camera_name)
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        render_truth = self.backend == "mujoco_segmentation" or bool(
            self.config.get("render_segmentation_for_audit", False)
        )
        segmentation = self._render_segmentation(data) if render_truth else None
        simulation_mask = None
        if segmentation is not None:
            simulation_mask = (
                (segmentation[..., 0] == self.target_geom_id)
                & (segmentation[..., 1] == int(self.mujoco.mjtObj.mjOBJ_GEOM))
            )
        if self.detector is None:
            mask = simulation_mask.copy()
            confidence = 1.0 if np.any(mask) else 0.0
        else:
            mask, confidence = self.detector.detect(rgb)

        camera_position = data.cam_xpos[self.camera_id].copy()
        camera_rotation = data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        target_visible_truth = (
            bool(np.any(simulation_mask))
            if simulation_mask is not None
            else self._target_center_inside_camera_frustum(
                data, camera_position, camera_rotation
            )
        )
        dropped = self.rng.rand() < float(self.config["frame_dropout_probability"])
        valid_depth = (
            mask
            & np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= float(self.config["maximum_depth_m"]))
        )
        if np.any(valid_depth):
            foreground_depth = float(np.percentile(depth[valid_depth], 15.0))
            valid_depth &= depth <= foreground_depth + float(
                self.config["depth_foreground_band_m"]
            )
        if np.any(valid_depth):
            random_drop = self.rng.rand(*depth.shape) < float(
                self.config["depth_pixel_dropout_probability"]
            )
            valid_depth &= ~random_drop
        noisy_depth = depth.copy()
        count = int(np.sum(valid_depth))
        if count:
            noisy_depth[valid_depth] += self.rng.normal(
                scale=float(self.config["depth_noise_std_m"]), size=count
            )
        detected = not dropped and count >= int(self.config["minimum_mask_pixels"])
        center_uv = None
        position = np.zeros(3, dtype=float)
        if detected:
            points = unproject_depth_mask(
                noisy_depth,
                valid_depth,
                self.intrinsics,
                camera_position,
                camera_rotation,
            )
            surface_center = np.median(points, axis=0)
            view_direction = normalize(surface_center - camera_position)
            position = surface_center + float(
                self.config["center_offset_along_view_m"]
            ) * view_direction
            rows, columns = np.nonzero(valid_depth)
            center_uv = np.array([np.median(columns), np.median(rows)], dtype=float)
        observation = PointObservation(
            timestamp=float(timestamp),
            position=position,
            valid=bool(detected),
            source_index=int(source_index),
        )
        return VisionDetection(
            observation,
            bool(detected),
            float(confidence),
            count,
            center_uv,
            rgb,
            noisy_depth,
            valid_depth,
            simulation_mask,
            camera_position,
            camera_rotation,
            bool(dropped),
            bool(target_visible_truth),
        )
