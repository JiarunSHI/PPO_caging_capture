"""Shared MuJoCo/UR5e bindings for 100 Hz learned and Jacobian control.

The module deliberately resolves every robot index by name.  Both controller
branches therefore use the exact same six joints and the exact same TCP site
from the selected MJCF instead of relying on an assumed qpos layout.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from cage_system.geometry import rotation_from_matrix


@dataclass(frozen=True)
class UR5eModelContract:
    joint_names: tuple
    qpos_addresses: tuple
    dof_addresses: tuple
    actuator_ids: tuple
    tcp_site_name: str
    control_rate_hz: float
    physics_timestep_s: float
    physics_steps_per_control: int


class UR5eControlModel:
    """Name-resolved position-servo interface to one UR5e MJCF model."""

    def __init__(
        self,
        model_path: str,
        robot_config: Dict,
        tcp_site_name: str = "attachment_site",
        control_rate_hz: float = 100.0,
    ):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError(
                "MuJoCo is required for the UR5e control branches"
            ) from error

        self.mujoco = mujoco
        self.model_path = str(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.robot = robot_config
        self.control_rate_hz = float(control_rate_hz)
        if self.control_rate_hz <= 0.0:
            raise ValueError("UR5e control rate must be positive")
        self.control_dt = 1.0 / self.control_rate_hz
        physics_ratio = self.control_dt / float(self.model.opt.timestep)
        self.physics_steps_per_control = int(round(physics_ratio))
        if self.physics_steps_per_control < 1 or not np.isclose(
            physics_ratio,
            self.physics_steps_per_control,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "The MJCF timestep ({:.6f}s) must divide the {:.6f}s control "
                "period exactly".format(self.model.opt.timestep, self.control_dt)
            )

        self.joint_names = tuple(str(name) for name in robot_config["joint_names"])
        if len(self.joint_names) != 6 or len(set(self.joint_names)) != 6:
            raise ValueError("The UR5e contract requires six unique joint names")
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in self.joint_names
            ],
            dtype=int,
        )
        if np.any(self.joint_ids < 0):
            missing = [
                name
                for name, joint_id in zip(self.joint_names, self.joint_ids)
                if joint_id < 0
            ]
            raise ValueError(
                "The selected MJCF is missing UR5e joints: {}".format(
                    ", ".join(missing)
                )
            )
        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        if any(int(self.model.jnt_type[joint_id]) != hinge_type for joint_id in self.joint_ids):
            raise ValueError("Every controlled UR5e joint must be a 1-DoF hinge")

        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids].astype(int)
        if len(set(self.qpos_addresses.tolist())) != 6 or len(
            set(self.dof_addresses.tolist())
        ) != 6:
            raise ValueError("UR5e joints do not map to six independent coordinates")

        self.tcp_site_name = str(tcp_site_name)
        self.tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site_name
        )
        if self.tcp_site_id < 0:
            raise ValueError(
                "The selected MJCF is missing TCP site '{}'".format(
                    self.tcp_site_name
                )
            )

        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        actuator_ids = []
        for name, joint_id in zip(self.joint_names, self.joint_ids):
            matches = np.flatnonzero(actuator_joint_ids == joint_id)
            if len(matches) != 1:
                raise ValueError(
                    "Joint '{}' must map to exactly one actuator, found {}".format(
                        name, len(matches)
                    )
                )
            actuator_ids.append(int(matches[0]))
        self.actuator_ids = np.asarray(actuator_ids, dtype=int)

        configured_lower = np.asarray(robot_config["joint_lower_rad"], dtype=float)
        configured_upper = np.asarray(robot_config["joint_upper_rad"], dtype=float)
        self.velocity_limits = np.asarray(
            robot_config["joint_velocity_limit_radps"], dtype=float
        )
        if (
            configured_lower.shape != (6,)
            or configured_upper.shape != (6,)
            or self.velocity_limits.shape != (6,)
            or np.any(configured_lower >= configured_upper)
            or np.any(self.velocity_limits <= 0.0)
        ):
            raise ValueError("UR5e joint position/velocity limits must contain six valid values")
        model_lower = np.full(6, -np.inf, dtype=float)
        model_upper = np.full(6, np.inf, dtype=float)
        for index, joint_id in enumerate(self.joint_ids):
            if bool(self.model.jnt_limited[joint_id]):
                model_lower[index], model_upper[index] = self.model.jnt_range[joint_id]
        self.lower = np.maximum(configured_lower, model_lower)
        self.upper = np.minimum(configured_upper, model_upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Configured and MJCF joint ranges have an empty intersection")

        self.compensate_bias = bool(
            robot_config.get("compensate_bias_forces_in_position_control", True)
        )
        self.contract = UR5eModelContract(
            joint_names=self.joint_names,
            qpos_addresses=tuple(int(value) for value in self.qpos_addresses),
            dof_addresses=tuple(int(value) for value in self.dof_addresses),
            actuator_ids=tuple(int(value) for value in self.actuator_ids),
            tcp_site_name=self.tcp_site_name,
            control_rate_hz=self.control_rate_hz,
            physics_timestep_s=float(self.model.opt.timestep),
            physics_steps_per_control=self.physics_steps_per_control,
        )

    def new_data(self):
        return self.mujoco.MjData(self.model)

    def hide_mocap_targets(self, data) -> None:
        """Move task-irrelevant visual mocap bodies outside the robot scene."""
        if self.model.nmocap:
            data.mocap_pos[:] = np.array([0.0, 0.0, -5.0], dtype=float)

    def reset_data(self, data, joints, joint_velocities=None) -> None:
        joints = np.asarray(joints, dtype=float)
        velocities = (
            np.zeros(6, dtype=float)
            if joint_velocities is None
            else np.asarray(joint_velocities, dtype=float)
        )
        if joints.shape != (6,) or velocities.shape != (6,):
            raise ValueError("UR5e reset expects q6 and qdot6")
        if np.any(joints < self.lower) or np.any(joints > self.upper):
            raise ValueError("UR5e reset joint position is outside the common limits")
        self.mujoco.mj_resetData(self.model, data)
        self.hide_mocap_targets(data)
        data.qpos[self.qpos_addresses] = joints
        data.qvel[self.dof_addresses] = velocities
        data.ctrl[self.actuator_ids] = joints
        self.mujoco.mj_forward(self.model, data)

    def joints(self, data):
        return data.qpos[self.qpos_addresses].copy()

    def joint_velocities(self, data):
        return data.qvel[self.dof_addresses].copy()

    def tcp_pose6(self, data):
        return np.concatenate(
            [
                data.site_xpos[self.tcp_site_id].copy(),
                rotation_from_matrix(
                    data.site_xmat[self.tcp_site_id].reshape(3, 3)
                ).as_rotvec(),
            ]
        )

    def tcp_jacobian(self, data):
        jacobian_position = np.zeros((3, self.model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=float)
        self.mujoco.mj_jacSite(
            self.model,
            data,
            jacobian_position,
            jacobian_rotation,
            self.tcp_site_id,
        )
        return np.vstack(
            [
                jacobian_position[:, self.dof_addresses],
                jacobian_rotation[:, self.dof_addresses],
            ]
        )

    def collision_count(self, data) -> int:
        return int(data.ncon)

    def step_position_target(self, data, joint_target) -> bool:
        """Advance one control period and report any MuJoCo contact."""
        target = np.clip(np.asarray(joint_target, dtype=float), self.lower, self.upper)
        if target.shape != (6,):
            raise ValueError("UR5e position command must contain six joints")
        data.ctrl[self.actuator_ids] = target
        contact = False
        for _ in range(self.physics_steps_per_control):
            if self.compensate_bias:
                data.qfrc_applied[self.dof_addresses] = data.qfrc_bias[
                    self.dof_addresses
                ]
            else:
                data.qfrc_applied[self.dof_addresses] = 0.0
            self.mujoco.mj_step(self.model, data)
            contact = contact or bool(data.ncon)
        return contact

    def contract_dict(self):
        return {
            "joint_names": list(self.contract.joint_names),
            "qpos_addresses": list(self.contract.qpos_addresses),
            "dof_addresses": list(self.contract.dof_addresses),
            "actuator_ids": list(self.contract.actuator_ids),
            "tcp_site_name": self.contract.tcp_site_name,
            "control_rate_hz": self.contract.control_rate_hz,
            "physics_timestep_s": self.contract.physics_timestep_s,
            "physics_steps_per_control": self.contract.physics_steps_per_control,
        }
