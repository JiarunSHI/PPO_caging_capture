import numpy as np

from cage_system.cage.feasibility import BackendResult
from cage_system.geometry import matrix_to_quaternion_xyzw


class MoveItFeasibilityBackend:
    """Production IK and collision hard-filter through MoveIt services."""

    name = "moveit"

    def __init__(self, robot_config, labeling_config, base_frame: str):
        try:
            import rospy
            from moveit_msgs.srv import GetPositionIK, GetStateValidity
        except ImportError as error:
            raise RuntimeError("MoveIt Python messages/services are unavailable") from error
        self.rospy = rospy
        self.GetPositionIK = GetPositionIK
        self.GetStateValidity = GetStateValidity
        self.robot = robot_config
        self.labeling = labeling_config
        self.base_frame = base_frame
        self.names = list(robot_config["joint_names"])
        self.lower = np.asarray(robot_config["joint_lower_rad"], dtype=float)
        self.upper = np.asarray(robot_config["joint_upper_rad"], dtype=float)
        rospy.wait_for_service("/compute_ik", timeout=10.0)
        rospy.wait_for_service("/check_state_validity", timeout=10.0)
        self.ik_service = rospy.ServiceProxy("/compute_ik", GetPositionIK, persistent=True)
        self.validity_service = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity, persistent=True
        )

    def solve(self, transform_base_tcp: np.ndarray, seed_joints: np.ndarray) -> BackendResult:
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.msg import MoveItErrorCodes
        from moveit_msgs.srv import GetPositionIKRequest, GetStateValidityRequest

        request = GetPositionIKRequest()
        request.ik_request.group_name = self.labeling["moveit_group_name"]
        request.ik_request.ik_link_name = self.labeling["moveit_tcp_link"]
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout = self.rospy.Duration(
            float(self.labeling["moveit_ik_timeout_s"])
        )
        request.ik_request.robot_state.joint_state.name = self.names
        request.ik_request.robot_state.joint_state.position = list(np.asarray(seed_joints))
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.rospy.Time(0)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = transform_base_tcp[:3, 3]
        quaternion = matrix_to_quaternion_xyzw(transform_base_tcp[:3, :3])
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quaternion
        request.ik_request.pose_stamped = pose
        response = self.ik_service(request)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return BackendResult(False, False, None, -np.inf, -np.inf, "moveit_ik_failed")
        solution_map = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        if any(name not in solution_map for name in self.names):
            return BackendResult(False, False, None, -np.inf, -np.inf, "moveit_solution_missing_joint")
        joints = np.asarray([solution_map[name] for name in self.names], dtype=float)
        validity_request = GetStateValidityRequest()
        validity_request.robot_state = response.solution
        validity_request.group_name = self.labeling["moveit_group_name"]
        validity = self.validity_service(validity_request)
        joint_margin = float(np.min(np.minimum(joints - self.lower, self.upper - joints)))
        return BackendResult(
            reachable=True,
            collision_free=bool(validity.valid),
            joints=joints,
            reachability_margin=joint_margin,
            # The ROS service exposes validity, not continuous FCL distance.
            collision_distance=np.inf if validity.valid else -np.inf,
            reason="" if validity.valid else "moveit_state_collision",
        )

