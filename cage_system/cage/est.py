"""Kinodynamic EST expansion using MuJoCo rigid-body forward simulation."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from cage_system.cage.internal_region import CageInternalRegion
from cage_system.geometry import rotation_from_matrix
from cage_system.types import ESTMetrics, ObjectState


@dataclass
class DisturbanceTape:
    """Candidate-independent wrench samples reused for fair EST comparison."""

    forces_base: np.ndarray
    torques_base: np.ndarray
    durations: np.ndarray
    selection_quantiles: np.ndarray

    @classmethod
    def generate(cls, config: Dict, seed: int, arc=None):
        rng = np.random.RandomState(int(seed))
        count = int(config["node_count"])
        force_limit = np.asarray(config["force_limit_n"], dtype=float)
        torque_limit = np.asarray(config["torque_limit_nm"], dtype=float)
        durations = rng.uniform(
            float(config["action_duration_s"][0]),
            float(config["action_duration_s"][1]),
            size=count,
        )
        forces = rng.uniform(-force_limit, force_limit, size=(count, 3))
        torques = rng.uniform(-torque_limit, torque_limit, size=(count, 3))
        selection_quantiles = rng.uniform(0.0, 1.0, size=count)
        sampling_mode = str(
            config.get("disturbance_direction_sampling_mode", "uniform_box_v1")
        )
        if sampling_mode == "motion_arc_mixture_v1":
            if arc is None:
                raise ValueError("Arc-directed EST sampling requires a candidate arc")
            probability = float(config["arc_direction_probability"])
            directional_mask = rng.uniform(0.0, 1.0, size=count) < probability
            lower, upper = sorted((float(arc.angle_start), float(arc.angle_end)))
            planar_angles = rng.uniform(lower, upper, size=count)
            out_of_plane = rng.normal(
                0.0,
                float(config["arc_direction_out_of_plane_std"]),
                size=count,
            )
            directions = (
                np.cos(planar_angles)[:, None] * np.asarray(arc.u_axis)[None, :]
                + np.sin(planar_angles)[:, None] * np.asarray(arc.v_axis)[None, :]
                + out_of_plane[:, None] * np.asarray(arc.normal)[None, :]
            )
            directions /= np.maximum(
                np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
            )
            magnitude_bounds = np.asarray(
                config["arc_direction_force_magnitude_fraction"], dtype=float
            )
            magnitude = rng.uniform(
                magnitude_bounds[0], magnitude_bounds[1], size=(count, 1)
            )
            directional_forces = directions * force_limit[None, :] * magnitude
            forces[directional_mask] = directional_forces[directional_mask]
        elif sampling_mode != "uniform_box_v1":
            raise ValueError("Unsupported EST disturbance direction sampling mode")
        return cls(
            forces_base=forces,
            torques_base=torques,
            durations=durations,
            selection_quantiles=selection_quantiles,
        )


@dataclass
class _Node:
    qpos: np.ndarray
    qvel: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    time: float
    parent: int
    status: int
    opening_escape: bool


class MujocoESTEvaluator:
    """Fixed-Cage EST whose tree is propagated by MuJoCo dynamics.

    Capture membership is a post-expansion node label, not an EST validity
    condition. A root or child outside the Cage can therefore remain a parent
    and later descendants can enter the true Cage region. Only non-finite,
    excessive-penetration, or environment-bound violations stop expansion.
    """

    INSIDE = 0
    OUTSIDE = 1
    ILLEGAL = 2

    def __init__(
        self,
        est_config,
        gripper_config,
        vision_config,
        internal_region: CageInternalRegion,
    ):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("MuJoCo is required for EST dynamics expansion") from error
        self.mujoco = mujoco
        self.config = est_config
        self.region = internal_region
        self.environment_min = np.asarray(est_config["environment_min_m"], dtype=float)
        self.environment_max = np.asarray(est_config["environment_max_m"], dtype=float)
        local_bounds = est_config.get("local_expansion_bounds_endpoint_frame")
        if local_bounds is None:
            self.local_expansion_min = None
            self.local_expansion_max = None
        else:
            self.local_expansion_min = np.asarray(local_bounds["min_m"], dtype=float)
            self.local_expansion_max = np.asarray(local_bounds["max_m"], dtype=float)
        self.maximum_penetration = float(est_config["maximum_contact_penetration_m"])
        xml = self._build_model_xml(est_config, gripper_config, vision_config)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "est_object"
        )
        cage_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "est_cage"
        )
        self.cage_mocap_id = int(self.model.body_mocapid[cage_body_id])
        self.free_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "est_object_free"
        )
        self.qpos_address = int(self.model.jnt_qposadr[self.free_joint_id])
        self.qvel_address = int(self.model.jnt_dofadr[self.free_joint_id])

    @staticmethod
    def _build_model_xml(est_config, gripper_config, vision_config):
        region = gripper_config["internal_region_endpoint_frame"]
        rear = float(region["rear_x"])
        mouth = float(region["mouth_x"])
        y_min = float(region["inner_y_min"])
        y_max = float(region["inner_y_max"])
        z_min = float(region["inner_z_min"])
        z_max = float(region["inner_z_max"])
        thickness = float(gripper_config["finger_thickness_m"])
        finger_x = 0.5 * (rear + mouth)
        finger_half_x = 0.5 * (mouth - rear)
        finger_half_z = 0.5 * (z_max - z_min)
        finger_z = 0.5 * (z_min + z_max)
        left_y = y_max + 0.5 * thickness
        right_y = y_min - 0.5 * thickness
        palm_half_y = max(abs(y_min), abs(y_max)) + thickness
        friction = " ".join(str(float(item)) for item in est_config["contact_friction"])
        gravity = " ".join(str(float(item)) for item in est_config["gravity_base_mps2"])
        timestep = float(est_config["integration_dt_s"])
        radius = float(vision_config["target_radius_m"])
        mass = float(gripper_config["object_mass_kg"])
        restitution = float(gripper_config["contact_restitution"])
        damping_ratio = max(0.15, 1.0 - restitution)
        return f"""
<mujoco model="cage_est_dynamics">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" gravity="{gravity}" integrator="implicitfast"/>
  <default>
    <geom friction="{friction}" solref="0.008 {damping_ratio}" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="est_floor" type="plane" size="2 2 0.05"/>
    <body name="est_cage" mocap="true">
      <geom name="est_left_finger" type="box"
        pos="{finger_x} {left_y} {finger_z}"
        size="{finger_half_x} {0.5 * thickness} {finger_half_z}"/>
      <geom name="est_right_finger" type="box"
        pos="{finger_x} {right_y} {finger_z}"
        size="{finger_half_x} {0.5 * thickness} {finger_half_z}"/>
      <geom name="est_rear_palm" type="box"
        pos="{0.5 * rear} {0.5 * (y_min + y_max)} {finger_z}"
        size="{0.5 * rear} {palm_half_y} {finger_half_z}"/>
    </body>
    <body name="est_object" pos="0 0 0.5">
      <freejoint name="est_object_free"/>
      <geom name="est_object_geom" type="sphere" size="{radius}" mass="{mass}"/>
    </body>
  </worldbody>
</mujoco>
"""

    def evaluate(
        self,
        object_state: ObjectState,
        transform_base_endpoint: np.ndarray,
        tape: DisturbanceTape,
    ) -> ESTMetrics:
        self.mujoco.mj_resetData(self.model, self.data)
        self._set_cage_pose(transform_base_endpoint)
        root_qpos = self.data.qpos.copy()
        root_qvel = self.data.qvel.copy()
        root_qpos[self.qpos_address : self.qpos_address + 3] = object_state.position
        root_qpos[self.qpos_address + 3 : self.qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
        root_qvel[self.qvel_address : self.qvel_address + 3] = object_state.velocity
        root_qvel[self.qvel_address + 3 : self.qvel_address + 6] = 0.0
        self.data.qpos[:] = root_qpos
        self.data.qvel[:] = root_qvel
        self.mujoco.mj_forward(self.model, self.data)
        root_membership = self.region.evaluate(
            transform_base_endpoint, object_state.position
        )
        root = _Node(
            root_qpos,
            root_qvel,
            object_state.position.copy(),
            object_state.velocity.copy(),
            0.0,
            -1,
            self.INSIDE if root_membership.inside else self.OUTSIDE,
            bool(root_membership.escaped_opening),
        )
        nodes: List[_Node] = [root]
        horizon = float(self.config["horizon_s"])
        integration_dt = float(self.model.opt.timestep)
        target_count = int(self.config["node_count"])
        # A node's density-bin key and its eligibility never change during one
        # EST expansion.  The former implementation rebuilt every existing
        # node's transformed 9-D key for every new node, making one tree
        # quadratic with a large Python/NumPy constant.  Keep the occupied
        # bins incrementally instead.  Choosing a bin uniformly and then a
        # node uniformly inside it is exactly the intended inverse-density
        # distribution: a bin containing k nodes has k * (1 / k) total mass.
        inverse_endpoint = np.linalg.inv(transform_base_endpoint)
        density_bins = {}
        active_bin_keys = []
        if self._node_is_selectable(root, horizon, integration_dt):
            self._add_density_node(
                density_bins,
                active_bin_keys,
                self._density_key(root, inverse_endpoint),
                0,
            )
        for tape_index in range(target_count - 1):
            if not active_bin_keys:
                break
            parent_index = self._select_low_density(
                density_bins,
                active_bin_keys,
                tape.selection_quantiles[tape_index],
            )
            parent = nodes[parent_index]
            duration = min(float(tape.durations[tape_index]), horizon - parent.time)
            remaining_steps = int(
                np.floor((horizon - parent.time + 1e-12) / integration_dt)
            )
            requested_steps = max(
                1, int(np.ceil(duration / integration_dt - 1e-12))
            )
            integration_steps = min(requested_steps, remaining_steps)
            if integration_steps <= 0:
                break
            child = self._propagate(
                parent,
                parent_index,
                integration_steps,
                tape.forces_base[tape_index],
                tape.torques_base[tape_index],
                transform_base_endpoint,
            )
            nodes.append(child)
            child_index = len(nodes) - 1
            if self._node_is_selectable(child, horizon, integration_dt):
                self._add_density_node(
                    density_bins,
                    active_bin_keys,
                    self._density_key(child, inverse_endpoint),
                    child_index,
                )

        statuses = np.asarray([node.status for node in nodes], dtype=int)
        inside = int(np.sum(statuses == self.INSIDE))
        outside = int(np.sum(statuses == self.OUTSIDE))
        illegal = int(np.sum(statuses == self.ILLEGAL))
        total = len(nodes)
        terminal_indices = self._terminal_indices(nodes)
        terminal_inside = sum(nodes[index].status == self.INSIDE for index in terminal_indices)
        retained_paths = self._retained_path_flags(nodes)
        positions = np.asarray([node.position for node in nodes])
        return ESTMetrics(
            inside_nodes=inside,
            escaped_nodes=outside,
            illegal_nodes=illegal,
            total_nodes=total,
            occupancy_ratio=float(inside / max(total, 1)),
            always_retained_ratio=float(np.mean(retained_paths)),
            terminal_retained_ratio=float(terminal_inside / max(len(terminal_indices), 1)),
            opening_escape_ratio=float(np.mean([node.opening_escape for node in nodes])),
            illegal_ratio=float(illegal / max(total, 1)),
            nodes_base=positions,
            node_status=statuses,
            node_parent_indices=np.asarray([node.parent for node in nodes], dtype=int),
            node_times=np.asarray([node.time for node in nodes], dtype=float),
        )

    def _set_cage_pose(self, transform):
        rotation_xyzw = rotation_from_matrix(transform[:3, :3]).as_quat()
        quaternion_wxyz = np.array(
            [rotation_xyzw[3], rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2]]
        )
        self.data.mocap_pos[self.cage_mocap_id] = transform[:3, 3]
        self.data.mocap_quat[self.cage_mocap_id] = quaternion_wxyz

    @classmethod
    def _node_is_selectable(cls, node, horizon, integration_dt):
        return bool(
            node.status != cls.ILLEGAL
            and node.time <= horizon - integration_dt + 1e-12
        )

    @staticmethod
    def _add_density_node(density_bins, active_bin_keys, key, node_index):
        if key not in density_bins:
            density_bins[key] = []
            active_bin_keys.append(key)
        density_bins[key].append(int(node_index))

    def _density_key(self, node, inverse_endpoint):
        rotation_endpoint_base = inverse_endpoint[:3, :3]
        local_position = (
            rotation_endpoint_base @ node.position + inverse_endpoint[:3, 3]
        )
        local_velocity = rotation_endpoint_base @ node.velocity
        local_angular = rotation_endpoint_base @ node.qvel[
            self.qvel_address + 3 : self.qvel_address + 6
        ]
        position_bin = float(self.config["density_position_bin_m"])
        velocity_bin = float(self.config["density_velocity_bin_mps"])
        angular_bin = float(self.config["density_angular_velocity_bin_radps"])
        return (
            tuple(np.floor(local_position / position_bin).astype(int))
            + tuple(np.floor(local_velocity / velocity_bin).astype(int))
            + tuple(np.floor(local_angular / angular_bin).astype(int))
        )

    @staticmethod
    def _select_low_density(density_bins, active_bin_keys, quantile):
        # One uniform variate is split into a uniform occupied-bin choice and a
        # uniform node choice inside that bin.  Clamp the endpoint defensively
        # even though DisturbanceTape generates values in [0, 1).
        scaled = min(max(float(quantile), 0.0), np.nextafter(1.0, 0.0)) * len(
            active_bin_keys
        )
        bin_index = min(int(scaled), len(active_bin_keys) - 1)
        within_bin = scaled - bin_index
        indices = density_bins[active_bin_keys[bin_index]]
        node_offset = min(int(within_bin * len(indices)), len(indices) - 1)
        return indices[node_offset]

    def _propagate(self, parent, parent_index, integration_steps, force, torque, transform):
        self.data.qpos[:] = parent.qpos
        self.data.qvel[:] = parent.qvel
        self.data.time = parent.time
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.object_body_id, :3] = force
        self.data.xfrc_applied[self.object_body_id, 3:] = torque
        illegal = False
        for _ in range(integration_steps):
            self.mujoco.mj_step(self.model, self.data)
            if not self._physics_state_valid(transform):
                illegal = True
                break
        self.data.xfrc_applied[:] = 0.0
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        position = qpos[self.qpos_address : self.qpos_address + 3].copy()
        velocity = qvel[self.qvel_address : self.qvel_address + 3].copy()
        membership = self.region.evaluate(transform, position)
        if illegal:
            status = self.ILLEGAL
        else:
            status = self.INSIDE if membership.inside else self.OUTSIDE
        return _Node(
            qpos,
            qvel,
            position,
            velocity,
            parent.time + integration_steps * self.model.opt.timestep,
            parent_index,
            status,
            bool(membership.escaped_opening),
        )

    def _physics_state_valid(self, transform_base_endpoint=None):
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return False
        position = self.data.qpos[self.qpos_address : self.qpos_address + 3]
        if np.any(position < self.environment_min) or np.any(position > self.environment_max):
            return False
        if (
            self.local_expansion_min is not None
            and transform_base_endpoint is not None
        ):
            rotation_base_endpoint = transform_base_endpoint[:3, :3]
            local_position = rotation_base_endpoint.T @ (
                position - transform_base_endpoint[:3, 3]
            )
            if np.any(local_position < self.local_expansion_min) or np.any(
                local_position > self.local_expansion_max
            ):
                return False
        for contact_index in range(self.data.ncon):
            if self.data.contact[contact_index].dist < -self.maximum_penetration:
                return False
        return True

    @classmethod
    def _retained_path_flags(cls, nodes):
        flags = np.zeros(len(nodes), dtype=bool)
        flags[0] = nodes[0].status == cls.INSIDE
        for index, node in enumerate(nodes[1:], start=1):
            flags[index] = node.status == cls.INSIDE and flags[node.parent]
        return flags

    @staticmethod
    def _terminal_indices(nodes: List[_Node]) -> List[int]:
        has_child = np.zeros(len(nodes), dtype=bool)
        for node in nodes[1:]:
            has_child[node.parent] = True
        leaves = [index for index in range(len(nodes)) if not has_child[index]]
        return leaves or [len(nodes) - 1]
