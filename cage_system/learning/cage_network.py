import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from cage_system.learning.dataset import (
    CAGE_INPUT_DIM,
    CAGE_GEOMETRY_CONTRACT,
    HISTORY_CANONICALIZATION_CONTRACT,
    CageDatasetArrays,
    FeatureNormalizer,
    canonicalize_cage_batch,
    history_reference_transform,
    split_by_track,
)
from cage_system.geometry import pose6_to_transform, transform_to_pose6
from cage_system.learning.history_quality import HISTORY_QUALITY_CONTRACT
from cage_system.learning.device import resolve_torch_device, seed_learning
from cage_system.learning.state_arc_dataset import (
    STATE_ARC_DATASET_CONTRACT,
    STATE_ARC_INPUT_DIM,
    CageStateArcDatasetArrays,
    load_generated_label_arrays,
)


CAGE_CHECKPOINT_FORMAT_VERSION = 12
STATE_ARC_CHECKPOINT_FORMAT_VERSION = 1
STATE_ARC_MODEL_CONTRACT = "state_arc_continuous_cage_mlp_v1"


def rotvec_to_matrix(rotvec):
    angle = torch.norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / torch.clamp(angle, min=1e-7)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(rotvec.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    identity = identity.expand(rotvec.shape[:-1] + (3, 3))
    angle_matrix = angle.unsqueeze(-1)
    matrix = identity + torch.sin(angle_matrix) * skew + (1.0 - torch.cos(angle_matrix)) * (skew @ skew)
    small = angle.squeeze(-1) < 1e-5
    if torch.any(small):
        raw_x, raw_y, raw_z = rotvec.unbind(-1)
        raw_skew = torch.stack(
            [zero, -raw_z, raw_y, raw_z, zero, -raw_x, -raw_y, raw_x, zero], dim=-1
        ).reshape(rotvec.shape[:-1] + (3, 3))
        matrix = torch.where(small[..., None, None], identity + raw_skew, matrix)
    return matrix


def orientation_geodesic_squared(predicted_rotvec, target_rotvec):
    predicted = rotvec_to_matrix(predicted_rotvec)
    target = rotvec_to_matrix(target_rotvec)
    relative = target.transpose(-1, -2) @ predicted
    cosine = torch.clamp((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5, -0.999999, 0.999999)
    return torch.acos(cosine) ** 2


def validation_pose_metrics(
    predicted,
    target,
    position_threshold_m: float,
    orientation_threshold_deg: float,
):
    """Return interpretable regression metrics for one validation tensor."""

    position_error = torch.linalg.vector_norm(
        predicted[:, :3] - target[:, :3], dim=-1
    )
    orientation_error_rad = torch.sqrt(
        torch.clamp(
            orientation_geodesic_squared(predicted[:, 3:], target[:, 3:]),
            min=0.0,
        )
    )
    orientation_error_deg = torch.rad2deg(orientation_error_rad)
    position_success = position_error <= float(position_threshold_m)
    orientation_success = orientation_error_deg <= float(
        orientation_threshold_deg
    )
    return {
        "validation_position_mean_mm": float(position_error.mean().item() * 1000.0),
        "validation_position_median_mm": float(position_error.median().item() * 1000.0),
        "validation_position_p95_mm": float(
            torch.quantile(position_error, 0.95).item() * 1000.0
        ),
        "validation_orientation_mean_deg": float(
            orientation_error_deg.mean().item()
        ),
        "validation_orientation_median_deg": float(
            orientation_error_deg.median().item()
        ),
        "validation_orientation_p95_deg": float(
            torch.quantile(orientation_error_deg, 0.95).item()
        ),
        "validation_position_success_rate": float(position_success.float().mean().item()),
        "validation_orientation_success_rate": float(
            orientation_success.float().mean().item()
        ),
        "validation_pose_success_rate": float(
            (position_success & orientation_success).float().mean().item()
        ),
    }


class CageMLP(nn.Module):
    def __init__(self, hidden_dims: Sequence[int]):
        super().__init__()
        layers = []
        input_dim = CAGE_INPUT_DIM
        for hidden in hidden_dims:
            layers.extend([nn.Linear(input_dim, int(hidden)), nn.LayerNorm(int(hidden)), nn.SiLU()])
            input_dim = int(hidden)
        layers.append(nn.Linear(input_dim, 6))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        if inputs.ndim == 3:
            if tuple(inputs.shape[-2:]) != (5, 4):
                raise ValueError("CageMLP expects batched [5,4] point-history inputs")
            inputs = inputs.flatten(start_dim=1)
        if inputs.ndim != 2 or inputs.shape[-1] != CAGE_INPUT_DIM:
            raise ValueError("CageMLP expects a 20-D flattened input per sample")
        return self.network(inputs)


class StateArcCageMLP(nn.Module):
    """Direct current-state to normalized minor-arc-coordinate regressor."""

    def __init__(self, hidden_dims: Sequence[int] = (128, 128, 64)):
        super().__init__()
        layers = []
        input_dim = STATE_ARC_INPUT_DIM
        for hidden in hidden_dims:
            layers.extend([nn.Linear(input_dim, int(hidden)), nn.SiLU()])
            input_dim = int(hidden)
        layers.extend([nn.Linear(input_dim, 1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[-1] != STATE_ARC_INPUT_DIM:
            raise ValueError("StateArcCageMLP expects a batched 13-D state")
        return self.network(inputs)


def state_arc_physical_loss(
    predicted_alpha,
    target_alpha,
    candidate_radius,
    arc_start_angle,
    arc_end_angle,
    *,
    arc_huber_beta: float,
    position_error_scale_m: float,
    orientation_error_scale_rad: float,
    arc_loss_weight: float,
    position_loss_weight: float,
    orientation_loss_weight: float,
):
    """Loss on alpha plus its exact endpoint chord and in-plane angle errors."""

    alpha_error = predicted_alpha - target_alpha
    arc_span = torch.abs(arc_end_angle - arc_start_angle).reshape(-1, 1)
    angle_error = torch.abs(alpha_error) * arc_span
    radius = candidate_radius.reshape(-1, 1)
    endpoint_error = 2.0 * radius * torch.sin(0.5 * angle_error)
    zero = torch.zeros_like(alpha_error)
    arc_loss = F.smooth_l1_loss(
        alpha_error, zero, beta=float(arc_huber_beta), reduction="mean"
    )
    position_loss = F.smooth_l1_loss(
        endpoint_error / float(position_error_scale_m),
        zero,
        beta=1.0,
        reduction="mean",
    )
    orientation_loss = F.smooth_l1_loss(
        angle_error / float(orientation_error_scale_rad),
        zero,
        beta=1.0,
        reduction="mean",
    )
    total = (
        float(arc_loss_weight) * arc_loss
        + float(position_loss_weight) * position_loss
        + float(orientation_loss_weight) * orientation_loss
    )
    return total, arc_loss, position_loss, orientation_loss, endpoint_error, angle_error


def _state_arc_validation_metrics(
    predicted,
    target,
    radii,
    arc_start,
    arc_end,
    position_threshold_m: float,
    orientation_threshold_deg: float,
):
    alpha_error = torch.abs(predicted - target).reshape(-1)
    angle_error = alpha_error * torch.abs(arc_end - arc_start).reshape(-1)
    position_error = 2.0 * radii.reshape(-1) * torch.sin(0.5 * angle_error)
    orientation_error_deg = torch.rad2deg(angle_error)
    position_success = position_error <= float(position_threshold_m)
    orientation_success = orientation_error_deg <= float(orientation_threshold_deg)
    return {
        "validation_arc_alpha_mae": float(alpha_error.mean().item()),
        "validation_arc_alpha_median": float(alpha_error.median().item()),
        "validation_arc_alpha_p95": float(torch.quantile(alpha_error, 0.95).item()),
        "validation_position_mean_mm": float(position_error.mean().item() * 1000.0),
        "validation_position_median_mm": float(position_error.median().item() * 1000.0),
        "validation_position_p95_mm": float(torch.quantile(position_error, 0.95).item() * 1000.0),
        "validation_orientation_mean_deg": float(orientation_error_deg.mean().item()),
        "validation_orientation_median_deg": float(orientation_error_deg.median().item()),
        "validation_orientation_p95_deg": float(torch.quantile(orientation_error_deg, 0.95).item()),
        "validation_position_success_rate": float(position_success.float().mean().item()),
        "validation_orientation_success_rate": float(orientation_success.float().mean().item()),
        "validation_pose_success_rate": float((position_success & orientation_success).float().mean().item()),
    }


def _train_state_arc_cage_network(
    config,
    arrays: CageStateArcDatasetArrays,
    output_path: str,
    device: str,
):
    nominal_period = 1.0 / float(config["timing"]["detection_rate_hz"])
    arrays.require_detection_period(nominal_period)
    if len(arrays.inputs) == 0:
        raise ValueError("The state-arc Cage dataset contains no valid labels")
    cage_config = config["cage_network"]
    if cage_config.get("training_contract") != STATE_ARC_MODEL_CONTRACT:
        raise ValueError(
            "State-arc labels require the state-arc Cage training config; "
            "use configs/ur5e_cage_state_arc_train.yaml"
        )
    resolved_device = resolve_torch_device(device)
    seed_learning(int(config["system"]["seed"]))
    train_mask, validation_mask = split_by_track(
        arrays.track_ids,
        float(cage_config["validation_fraction"]),
        int(config["system"]["seed"]),
    )
    normalizer = FeatureNormalizer.fit(arrays.inputs[train_mask])
    train_x = normalizer.transform(arrays.inputs[train_mask]).astype(np.float32)
    validation_x = normalizer.transform(arrays.inputs[validation_mask]).astype(np.float32)
    train_y = arrays.targets[train_mask].astype(np.float32)
    validation_y = arrays.targets[validation_mask].astype(np.float32)
    train_radius = arrays.candidate_radii[train_mask].astype(np.float32)
    validation_radius = arrays.candidate_radii[validation_mask].astype(np.float32)
    train_start = arrays.arc_start_angles[train_mask].astype(np.float32)
    validation_start = arrays.arc_start_angles[validation_mask].astype(np.float32)
    train_end = arrays.arc_end_angles[train_mask].astype(np.float32)
    validation_end = arrays.arc_end_angles[validation_mask].astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
            torch.from_numpy(train_radius),
            torch.from_numpy(train_start),
            torch.from_numpy(train_end),
        ),
        batch_size=int(cage_config["batch_size"]),
        shuffle=True,
        pin_memory=resolved_device.type == "cuda",
    )
    hidden_dims = [int(item) for item in cage_config["hidden_dims"]]
    model = StateArcCageMLP(hidden_dims).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cage_config["learning_rate"]),
        weight_decay=float(cage_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cage_config["lr_decay_factor"]),
        patience=int(cage_config["lr_plateau_patience"]),
        min_lr=float(cage_config["minimum_learning_rate"]),
    )
    loss_settings = {
        "arc_huber_beta": float(cage_config["arc_huber_beta"]),
        "position_error_scale_m": float(cage_config["position_error_scale_m"]),
        "orientation_error_scale_rad": float(
            np.deg2rad(cage_config["orientation_error_scale_deg"])
        ),
        "arc_loss_weight": float(cage_config["arc_loss_weight"]),
        "position_loss_weight": float(cage_config["position_loss_weight"]),
        "orientation_loss_weight": float(cage_config["orientation_loss_weight"]),
    }
    maximum_epochs = int(cage_config["epochs"])
    early_stopping_patience = int(cage_config["early_stopping_patience"])
    early_stopping_min_delta = float(cage_config["early_stopping_min_delta"])
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    early_stopped = False
    history = []
    progress = tqdm(
        range(maximum_epochs),
        total=maximum_epochs,
        desc="State-arc Cage epochs",
        unit="epoch",
        dynamic_ncols=True,
    )
    validation_tensors = tuple(
        torch.from_numpy(item).to(resolved_device)
        for item in (
            validation_x,
            validation_y,
            validation_radius,
            validation_start,
            validation_end,
        )
    )
    for epoch in progress:
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for features, targets, radii, starts, ends in loader:
            features = features.to(resolved_device, non_blocking=True)
            targets = targets.to(resolved_device, non_blocking=True)
            radii = radii.to(resolved_device, non_blocking=True)
            starts = starts.to(resolved_device, non_blocking=True)
            ends = ends.to(resolved_device, non_blocking=True)
            prediction = model(features)
            loss, *_ = state_arc_physical_loss(
                prediction, targets, radii, starts, ends, **loss_settings
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(features)
            train_sample_count += len(features)
        model.eval()
        with torch.no_grad():
            features, targets, radii, starts, ends = validation_tensors
            prediction = model(features)
            (
                validation_total,
                validation_arc,
                validation_position,
                validation_orientation,
                _,
                _,
            ) = state_arc_physical_loss(
                prediction, targets, radii, starts, ends, **loss_settings
            )
            validation_metrics = _state_arc_validation_metrics(
                prediction,
                targets,
                radii,
                starts,
                ends,
                float(cage_config["validation_position_threshold_m"]),
                float(cage_config["validation_orientation_threshold_deg"]),
            )
        validation_loss = float(validation_total.item())
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / max(train_sample_count, 1),
            "validation_loss": validation_loss,
            "validation_arc_loss": float(validation_arc.item()),
            "validation_position_loss": float(validation_position.item()),
            "validation_orientation_loss": float(validation_orientation.item()),
            **validation_metrics,
        }
        if validation_loss < best_validation - early_stopping_min_delta:
            best_validation = validation_loss
            best_epoch = epoch + 1
            best_metrics = dict(epoch_metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(validation_loss)
        epoch_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        epoch_metrics["best_epoch"] = int(best_epoch)
        epoch_metrics["epochs_without_improvement"] = int(epochs_without_improvement)
        history.append(epoch_metrics)
        progress.set_postfix(
            val=f"{validation_loss:.4f}",
            alpha=f"{validation_metrics['validation_arc_alpha_mae']:.3f}",
            pos=f"{validation_metrics['validation_position_mean_mm']:.1f}mm",
            angle=f"{validation_metrics['validation_orientation_mean_deg']:.1f}deg",
            success=f"{100.0 * validation_metrics['validation_pose_success_rate']:.1f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )
        tqdm.write("[state-arc-cage][validation] " + json.dumps(epoch_metrics, sort_keys=True))
        if epochs_without_improvement >= early_stopping_patience:
            early_stopped = True
            progress.set_description("State-arc Cage epochs (early stopped)")
            break
    progress.close()
    if best_state is None:
        raise RuntimeError("State-arc Cage training produced no checkpoint")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "format_version": STATE_ARC_CHECKPOINT_FORMAT_VERSION,
            "model_contract": STATE_ARC_MODEL_CONTRACT,
            "dataset_contract": STATE_ARC_DATASET_CONTRACT,
            "hidden_dims": hidden_dims,
            "input_mean": normalizer.mean.astype(np.float32),
            "input_std": normalizer.std.astype(np.float32),
            "input_shape": [STATE_ARC_INPUT_DIM],
            "input_dim": STATE_ARC_INPUT_DIM,
            "output_dim": 1,
            "input_contract": "current Base position3 velocity3 acceleration3 positive plane normal3 signed curvature1",
            "output_contract": "sigmoid normalized signed-minor-arc coordinate; tangent=0 future-intersection=1",
            "object_model": "point_mass",
            "nominal_detection_period_s": nominal_period,
            "loss_contract": "arc_huber_plus_endpoint_chord_huber_plus_in_plane_orientation_huber_v1",
            "loss_settings": {
                **loss_settings,
                "orientation_error_scale_deg": float(cage_config["orientation_error_scale_deg"]),
            },
            "validation_thresholds": {
                "position_m": float(cage_config["validation_position_threshold_m"]),
                "orientation_deg": float(cage_config["validation_orientation_threshold_deg"]),
            },
            "best_validation_loss": best_validation,
            "best_epoch": int(best_epoch),
            "best_validation_metrics": best_metrics,
            "training_device": str(resolved_device),
            "training_schedule": {
                "maximum_epochs": maximum_epochs,
                "actual_epochs": len(history),
                "early_stopped": early_stopped,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
            },
            "history": history,
        },
        destination,
    )
    return {
        "best_validation_loss": best_validation,
        "best_epoch": int(best_epoch),
        "best_validation_metrics": best_metrics,
        "model_contract": STATE_ARC_MODEL_CONTRACT,
        "device": str(resolved_device),
        "epochs": len(history),
        "maximum_epochs": maximum_epochs,
        "early_stopped": early_stopped,
        "training_samples": int(np.sum(train_mask)),
        "validation_samples": int(np.sum(validation_mask)),
    }


def train_cage_network(config, dataset_path: str, output_path: str, device: str = "auto"):
    generated_arrays = load_generated_label_arrays(dataset_path)
    if isinstance(generated_arrays, CageStateArcDatasetArrays):
        return _train_state_arc_cage_network(
            config, generated_arrays, output_path, device
        )
    arrays = CageDatasetArrays.load(dataset_path)
    nominal_detection_period_s = 1.0 / float(
        config["timing"]["detection_rate_hz"]
    )
    arrays.require_detection_period(nominal_detection_period_s)
    if len(arrays.inputs) == 0:
        raise ValueError("The Cage dataset contains no valid labels")
    vision_config = config.get("vision", {})
    if (
        vision_config.get("enabled", False)
        and vision_config.get("require_rendered_cage_training_inputs", False)
        and arrays.input_source != "mujoco_eye_on_hand_rgbd"
    ):
        raise ValueError(
            "Vision mode requires a dataset produced by generate-vision-dataset"
        )
    cage_config = config["cage_network"]
    resolved_device = resolve_torch_device(device)
    seed_learning(int(config["system"]["seed"]))
    train_mask, validation_mask = split_by_track(
        arrays.track_ids, float(cage_config["validation_fraction"]), int(config["system"]["seed"])
    )
    base_reference = np.asarray(config["plane"]["base_reference_direction"], dtype=float)
    canonical_inputs, canonical_labels, _ = canonicalize_cage_batch(
        arrays.inputs, arrays.labels, base_reference
    )
    normalizer = FeatureNormalizer.fit(canonical_inputs[train_mask])
    train_x = normalizer.transform(canonical_inputs[train_mask]).astype(np.float32)
    validation_x = normalizer.transform(canonical_inputs[validation_mask]).astype(np.float32)
    train_y = canonical_labels[train_mask].astype(np.float32)
    validation_y = canonical_labels[validation_mask].astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(cage_config["batch_size"]),
        shuffle=True,
        pin_memory=resolved_device.type == "cuda",
    )
    model = CageMLP(cage_config["hidden_dims"]).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cage_config["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cage_config["lr_decay_factor"]),
        patience=int(cage_config["lr_plateau_patience"]),
        min_lr=float(cage_config["minimum_learning_rate"]),
    )
    position_weight = float(cage_config["position_loss_weight"])
    orientation_weight = float(cage_config["orientation_loss_weight"])
    position_absolute_weight = float(
        cage_config.get("position_absolute_loss_weight", 0.0)
    )
    orientation_absolute_weight = float(
        cage_config.get("orientation_absolute_loss_weight", 0.0)
    )
    maximum_epochs = int(cage_config["epochs"])
    early_stopping_patience = int(cage_config["early_stopping_patience"])
    early_stopping_min_delta = float(cage_config["early_stopping_min_delta"])
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    early_stopped = False
    history = []
    progress = tqdm(
        range(maximum_epochs),
        total=maximum_epochs,
        desc="Cage epochs",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for features, targets in loader:
            features = features.to(resolved_device, non_blocking=True)
            targets = targets.to(resolved_device, non_blocking=True)
            prediction = model(features)
            position_squared = torch.sum(
                (prediction[:, :3] - targets[:, :3]) ** 2, dim=-1
            )
            orientation_squared = orientation_geodesic_squared(
                prediction[:, 3:], targets[:, 3:]
            )
            position_loss = torch.mean(position_squared)
            orientation_loss = torch.mean(orientation_squared)
            position_absolute_loss = torch.mean(torch.sqrt(position_squared + 1e-12))
            orientation_absolute_loss = torch.mean(
                torch.sqrt(orientation_squared + 1e-12)
            )
            loss = (
                position_weight * position_loss
                + orientation_weight * orientation_loss
                + position_absolute_weight * position_absolute_loss
                + orientation_absolute_weight * orientation_absolute_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(features)
            train_sample_count += len(features)
        model.eval()
        with torch.no_grad():
            features = torch.from_numpy(validation_x).to(resolved_device)
            targets = torch.from_numpy(validation_y).to(resolved_device)
            prediction = model(features)
            validation_position_squared = torch.sum(
                (prediction[:, :3] - targets[:, :3]) ** 2, dim=-1
            )
            validation_orientation_squared = orientation_geodesic_squared(
                prediction[:, 3:], targets[:, 3:]
            )
            validation_position_loss = torch.mean(validation_position_squared)
            validation_orientation_loss = torch.mean(validation_orientation_squared)
            validation_position_absolute_loss = torch.mean(
                torch.sqrt(validation_position_squared + 1e-12)
            )
            validation_orientation_absolute_loss = torch.mean(
                torch.sqrt(validation_orientation_squared + 1e-12)
            )
            validation_loss = float(
                (
                    position_weight * validation_position_loss
                    + orientation_weight * validation_orientation_loss
                    + position_absolute_weight * validation_position_absolute_loss
                    + orientation_absolute_weight * validation_orientation_absolute_loss
                ).item()
            )
            pose_metrics = validation_pose_metrics(
                prediction,
                targets,
                float(cage_config["validation_position_threshold_m"]),
                float(cage_config["validation_orientation_threshold_deg"]),
            )
        train_loss = train_loss_sum / max(train_sample_count, 1)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_position_component": float(validation_position_loss.item()),
            "validation_orientation_component": float(
                validation_orientation_loss.item()
            ),
            "validation_position_absolute_component": float(
                validation_position_absolute_loss.item()
            ),
            "validation_orientation_absolute_component": float(
                validation_orientation_absolute_loss.item()
            ),
            **pose_metrics,
        }
        improved = validation_loss < best_validation - early_stopping_min_delta
        if improved:
            best_validation = validation_loss
            best_epoch = epoch + 1
            best_metrics = dict(epoch_metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(validation_loss)
        epoch_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        epoch_metrics["best_epoch"] = int(best_epoch)
        epoch_metrics["epochs_without_improvement"] = int(
            epochs_without_improvement
        )
        history.append(epoch_metrics)
        progress.set_postfix(
            val=f"{validation_loss:.4f}",
            pos=f"{pose_metrics['validation_position_mean_mm']:.1f}mm",
            angle=f"{pose_metrics['validation_orientation_mean_deg']:.1f}deg",
            success=f"{100.0 * pose_metrics['validation_pose_success_rate']:.1f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )
        tqdm.write(
            "[cage][validation] "
            + json.dumps(epoch_metrics, sort_keys=True),
        )
        if epochs_without_improvement >= early_stopping_patience:
            early_stopped = True
            progress.set_description("Cage epochs (early stopped)")
            break
    progress.close()
    if best_state is None:
        raise RuntimeError("Cage training produced no checkpoint")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "format_version": CAGE_CHECKPOINT_FORMAT_VERSION,
            "hidden_dims": list(cage_config["hidden_dims"]),
            "input_mean": normalizer.mean.astype(np.float32),
            "input_std": normalizer.std.astype(np.float32),
            "input_contract": "five strictly previous xyz+scalar_speed frames predict current Cage pose6",
            "object_model": "point_mass",
            "training_input_source": arrays.input_source,
            "input_shape": [5, 4],
            "input_dim": CAGE_INPUT_DIM,
            "output_dim": 6,
            "canonicalization_contract": HISTORY_CANONICALIZATION_CONTRACT,
            "canonicalization_base_reference": base_reference.astype(np.float32),
            "geometry_contract": CAGE_GEOMETRY_CONTRACT,
            "decision_trigger": "one_action_per_completed_visual_update",
            "history_quality_contract": HISTORY_QUALITY_CONTRACT,
            "history_quality_config": dict(config["history_quality"]),
            "kalman_config": dict(config["kalman"]),
            "nominal_detection_period_s": nominal_detection_period_s,
            "best_validation_loss": best_validation,
            "best_epoch": int(best_epoch),
            "best_validation_metrics": best_metrics,
            "pose_representation": "xyz_rotvec",
            "training_device": str(resolved_device),
            "training_schedule": {
                "maximum_epochs": maximum_epochs,
                "actual_epochs": len(history),
                "early_stopped": early_stopped,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "position_loss_weight": position_weight,
                "orientation_loss_weight": orientation_weight,
                "position_absolute_loss_weight": position_absolute_weight,
                "orientation_absolute_loss_weight": orientation_absolute_weight,
            },
            "history": history,
        },
        destination,
    )
    return {
        "best_validation_loss": best_validation,
        "best_epoch": int(best_epoch),
        "best_validation_metrics": best_metrics,
        "device": str(resolved_device),
        "epochs": len(history),
        "maximum_epochs": maximum_epochs,
        "early_stopped": early_stopped,
        "training_samples": int(np.sum(train_mask)),
        "validation_samples": int(np.sum(validation_mask)),
    }


class FrozenCagePredictor:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        resolved_device = resolve_torch_device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            int(checkpoint.get("format_version", 0))
            != CAGE_CHECKPOINT_FORMAT_VERSION
            or checkpoint.get("object_model") != "point_mass"
            or list(checkpoint.get("input_shape", [])) != [5, 4]
            or int(checkpoint.get("input_dim", -1)) != CAGE_INPUT_DIM
            or checkpoint.get("geometry_contract") != CAGE_GEOMETRY_CONTRACT
            or checkpoint.get("decision_trigger")
            != "one_action_per_completed_visual_update"
            or checkpoint.get("history_quality_contract")
            != HISTORY_QUALITY_CONTRACT
            or not isinstance(checkpoint.get("kalman_config"), dict)
            or checkpoint.get("canonicalization_contract")
            != HISTORY_CANONICALIZATION_CONTRACT
            or not np.isfinite(
                float(checkpoint.get("nominal_detection_period_s", np.nan))
            )
            or float(checkpoint.get("nominal_detection_period_s", 0.0)) <= 0.0
        ):
            raise ValueError(
                "Checkpoint does not implement the filtered point-mass [5,4] Cage contract"
            )
        self.model = CageMLP(checkpoint["hidden_dims"]).to(resolved_device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.as_tensor(checkpoint["input_mean"], device=resolved_device)
        self.std = torch.as_tensor(checkpoint["input_std"], device=resolved_device)
        self.history_quality_config = dict(checkpoint["history_quality_config"])
        self.kalman_config = dict(checkpoint["kalman_config"])
        self.nominal_detection_period_s = float(
            checkpoint["nominal_detection_period_s"]
        )
        self.base_reference = np.asarray(
            checkpoint["canonicalization_base_reference"], dtype=float
        )
        self.device = resolved_device

    def predict(self, history_matrix: np.ndarray) -> np.ndarray:
        history_matrix = np.asarray(history_matrix, dtype=np.float32)
        if history_matrix.shape != (5, 4):
            raise ValueError("Inference requires five prior xyz+scalar_speed rows")
        reference = history_reference_transform(history_matrix, self.base_reference)
        canonical, _, _ = canonicalize_cage_batch(
            history_matrix[None, ...], None, self.base_reference
        )
        flattened = torch.from_numpy(canonical.astype(np.float32).reshape(1, -1)).to(self.device)
        normalized = (flattened - self.mean) / self.std
        with torch.no_grad():
            relative_pose = self.model(normalized).cpu().numpy()[0]
        return transform_to_pose6(reference @ pose6_to_transform(relative_pose))
