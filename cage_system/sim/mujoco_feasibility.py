from typing import List, Tuple

import numpy as np

from cage_system.cage.feasibility import BackendResult
from cage_system.geometry import rotation_from_matrix
from cage_system.sim.ur5e_analytic_ik import inverse_ur5e, wrap_solutions_to_limits


class MujocoFeasibilityBackend:
    """Analytic multi-branch UR5e IK, polished against the compiled MJCF."""

    name = "mujoco"

    def __init__(self, model_path: str, tcp_site_name: str, robot_config):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("Install MuJoCo before selecting the mujoco feasibility backend") from error
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.names = list(robot_config["joint_names"])
        self.lower = np.asarray(robot_config["joint_lower_rad"], dtype=float)
        self.upper = np.asarray(robot_config["joint_upper_rad"], dtype=float)
        self.nominal = np.asarray(robot_config["nominal_joint_state_rad"], dtype=float)
        self.multi_start_count = int(robot_config.get("ik_multi_start_count", 16))
        self.analytic_empty_multistart_count = int(
            robot_config.get("ik_analytic_empty_multistart_count", 16)
        )
        self.maximum_iterations = int(robot_config.get("ik_max_iterations", 100))
        self.joint_ids = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.names
        ])
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name
        )
        if np.any(self.joint_ids < 0) or self.site_id < 0:
            raise ValueError("MJCF joint names or TCP site do not match the UR5e contract")
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids]

    def _polish(self, target: np.ndarray, initial: np.ndarray) -> Tuple[bool, np.ndarray]:
        joints = np.clip(np.asarray(initial, dtype=float), self.lower, self.upper)
        target_position, target_rotation = target[:3, 3], target[:3, :3]
        for _ in range(self.maximum_iterations):
            self.data.qpos[self.qpos_addresses] = joints
            self.mujoco.mj_forward(self.model, self.data)
            position = self.data.site_xpos[self.site_id].copy()
            rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
            orientation_error = rotation_from_matrix(target_rotation @ rotation.T).as_rotvec()
            error = np.concatenate([target_position - position, orientation_error])
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 2e-3:
                return True, joints.copy()
            jacobian_position = np.zeros((3, self.model.nv))
            jacobian_rotation = np.zeros((3, self.model.nv))
            self.mujoco.mj_jacSite(
                self.model, self.data, jacobian_position, jacobian_rotation, self.site_id
            )
            jacobian = np.vstack([
                jacobian_position[:, self.dof_addresses],
                jacobian_rotation[:, self.dof_addresses],
            ])
            try:
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 2e-3 * np.eye(6), error
                )
            except np.linalg.LinAlgError:
                break
            joints = np.clip(joints + 0.55 * delta, self.lower, self.upper)
        return False, joints

    def _fallback_seeds(self, reference: np.ndarray) -> List[np.ndarray]:
        seeds = [np.asarray(reference, dtype=float), self.nominal.copy()]
        offsets = (
            [np.pi / 2, 0, 0, 0, 0, 0], [-np.pi / 2, 0, 0, 0, 0, 0],
            [0, np.pi / 2, -np.pi / 2, 0, 0, 0], [0, -np.pi / 2, np.pi / 2, 0, 0, 0],
            [0, 0, np.pi / 2, -np.pi / 2, 0, 0], [0, 0, -np.pi / 2, np.pi / 2, 0, 0],
            [0, 0, 0, 0, np.pi, 0], [np.pi, 0, 0, 0, 0, np.pi],
            [-np.pi, 0, 0, 0, 0, -np.pi],
            [np.pi / 2, -np.pi / 2, np.pi / 2, 0, -np.pi / 2, 0],
            [-np.pi / 2, -np.pi / 2, np.pi / 2, 0, np.pi / 2, 0],
            [0, -np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0],
            [0, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0],
            [np.pi, -np.pi / 2, np.pi / 2, 0, 0, 0],
        )
        for offset in offsets:
            seeds.append(np.clip(self.nominal + np.asarray(offset), self.lower, self.upper))
        # A fixed low-discrepancy-like random cover avoids the former single-
        # basin failure while keeping label generation exactly reproducible.
        random_state = np.random.RandomState(73013)
        while len(seeds) < self.multi_start_count:
            seeds.append(random_state.uniform(self.lower, self.upper))
        unique = []
        for seed in seeds:
            if not any(np.linalg.norm(seed - item) < 1e-8 for item in unique):
                unique.append(seed)
        return unique[: self.multi_start_count]

    def _contact_result(self, joints: np.ndarray):
        self.data.qpos[self.qpos_addresses] = joints
        self.mujoco.mj_forward(self.model, self.data)
        if self.data.ncon == 0:
            return True, np.inf, ""
        distances = [float(self.data.contact[index].dist) for index in range(self.data.ncon)]
        contact_index = int(np.argmin(distances))
        contact = self.data.contact[contact_index]
        first = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
        )
        second = self.mujoco.mj_id2name(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
        )
        return False, float(distances[contact_index]), "mujoco_contact_collision:{}:{}".format(
            first or int(contact.geom1), second or int(contact.geom2)
        )

    def solve(self, transform_base_tcp: np.ndarray, seed_joints: np.ndarray) -> BackendResult:
        target = np.asarray(transform_base_tcp, dtype=float)
        reference = np.clip(np.asarray(seed_joints, dtype=float), self.lower, self.upper)
        analytic = wrap_solutions_to_limits(
            inverse_ur5e(target, q6_des=float(reference[5])),
            self.lower, self.upper, reference,
        )
        analytic_seeds: List[Tuple[np.ndarray, str]] = [
            (solution, "ur5e_analytic_polished") for solution in analytic
        ]
        fallback_seeds: List[Tuple[np.ndarray, str]] = []
        fallback_seeds.extend(
            (seed, "deterministic_multistart") for seed in self._fallback_seeds(reference)
        )
        analytic_unique: List[Tuple[np.ndarray, str]] = []
        for seed, source in analytic_seeds:
            if not any(np.linalg.norm(seed - prior[0]) < 1e-7 for prior in analytic_unique):
                analytic_unique.append((seed, source))
        fallback_unique: List[Tuple[np.ndarray, str]] = []
        for seed, source in fallback_seeds:
            if not any(
                np.linalg.norm(seed - prior[0]) < 1e-7
                for prior in analytic_unique + fallback_unique
            ):
                fallback_unique.append((seed, source))
        if not analytic_unique:
            fallback_unique = fallback_unique[: self.analytic_empty_multistart_count]

        attempted = 0
        converged_results = []
        for seed_group in (analytic_unique, fallback_unique):
            for candidate_seed, source in seed_group:
                attempted += 1
                converged, joints = self._polish(target, candidate_seed)
                if not converged:
                    continue
                collision_free, collision_distance, reason = self._contact_result(joints)
                margin = float(np.min(np.minimum(joints - self.lower, self.upper - joints)))
                distance = float(np.linalg.norm(joints - reference))
                converged_results.append(
                    (not collision_free, distance, joints, collision_free,
                     collision_distance, margin, reason, source)
                )
            # Analytic branches cover all UR5e topologies. Deterministic
            # multi-start is needed only if rounded-MJCF polishing did not
            # produce a collision-free analytic branch.
            if any(item[3] for item in converged_results):
                break
        if not converged_results:
            return BackendResult(
                False, False, None, -np.inf, -np.inf,
                "mujoco_ik_failed_after_analytic_and_multistart", "none", attempted
            )
        # Any collision-free branch outranks every colliding branch. Distance
        # to the seed only decides which hard-feasible root is returned.
        result = min(converged_results, key=lambda item: (item[0], item[1]))
        _, _, joints, collision_free, collision_distance, margin, reason, source = result
        return BackendResult(
            True, bool(collision_free), joints.copy(), margin, collision_distance,
            reason, source, attempted
        )
