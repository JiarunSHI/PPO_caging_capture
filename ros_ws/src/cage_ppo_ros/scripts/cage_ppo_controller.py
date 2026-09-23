#!/usr/bin/env python3
import threading

import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
import torch
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from cage_system.config import load_config
from cage_system.geometry import pose6_to_transform, rotation_distance
from cage_system.learning.cage_network import FrozenCagePredictor
from cage_system.learning.history_quality import maximum_history_interval_s
from cage_system.learning.online_interface import HistoricalCageInference
from cage_system.learning.ppo2 import build_ppo_state
from cage_system.learning.residual_ppo import load_frozen_control_policy
from cage_system.types import PointObservation


class CagePPOController:
    def __init__(self):
        self.config = load_config(rospy.get_param("~config"))
        self.robot = self.config["robot"]
        self.timing = self.config["timing"]
        self.joint_names = list(self.robot["joint_names"])
        self.base_frame = self.config["system"]["base_frame"]
        self.tcp_frame = rospy.get_param(
            "~tcp_frame", self.config["labeling"]["moveit_tcp_link"]
        )
        requested_device = rospy.get_param("~device", "auto")
        cage = FrozenCagePredictor(rospy.get_param("~cage_checkpoint"), requested_device)
        self.device = cage.device
        self.inference = HistoricalCageInference(
            cage,
            self.config["trajectory"],
            self.config["kalman"],
            self.timing,
            self.config["history_quality"],
        )
        self.ppo = load_frozen_control_policy(
            rospy.get_param("~ppo_checkpoint"), str(self.device)
        )
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.publisher = rospy.Publisher(
            rospy.get_param("~command_topic"), Float64MultiArray, queue_size=1
        )
        self.lock = threading.Lock()
        self.joints = None
        self.joint_velocities = None
        self.target = None
        self.target_timestamp = None
        self.last_object_stamp = None
        self.last_accepted_target = None
        self.last_accepted_target_time = None
        self.last_action_timestamp = None
        self.history_hold_active = True
        self.maximum_history_interval_s = maximum_history_interval_s(
            self.timing, self.config["history_quality"]
        )
        rospy.Subscriber(
            rospy.get_param("~object_point_topic"), PointStamped, self.object_callback, queue_size=1
        )
        rospy.Subscriber(
            rospy.get_param("~joint_state_topic"), JointState, self.joint_callback, queue_size=1
        )
        self.history_watchdog = rospy.Timer(
            rospy.Duration(0.5 / float(self.timing["detection_rate_hz"])),
            self._history_watchdog,
        )

    def object_callback(self, message):
        try:
            pose = self._to_base(message)
            stamp = pose.header.stamp.to_sec()
            age = (rospy.Time.now() - pose.header.stamp).to_sec()
            if age > float(self.timing["maximum_visual_to_action_latency_s"]):
                rospy.logwarn_throttle(1.0, "Stale visual event rejected; previous setpoint remains held")
                return
            observation = PointObservation(
                timestamp=stamp,
                position=np.array(
                    [pose.point.x, pose.point.y, pose.point.z],
                    dtype=float,
                ),
            )
            command = None
            with self.lock:
                prediction = self.inference.predict_current_then_update(observation)
                self.last_object_stamp = pose.header.stamp
                if prediction is not None and self._target_speed_is_safe(prediction, stamp):
                    self.target = prediction.copy()
                    self.target_timestamp = stamp
                    self.last_accepted_target = prediction.copy()
                    self.last_accepted_target_time = stamp
                    command = self._command_for_visual_event(prediction, stamp)
                    self.history_hold_active = False
                else:
                    command = self._hold_current_joints()
                    self.history_hold_active = True
                    rospy.logwarn_throttle(
                        1.0,
                        "PPO held for abnormal five-frame history: %s",
                        self.inference.last_quality.reason or "unsafe_cage_target",
                    )
            if command is not None:
                # Event-driven decision: publish immediately after this visual
                # observation and its single PPO inference have completed.
                self.publisher.publish(Float64MultiArray(data=command.tolist()))
        except Exception as error:
            rospy.logwarn_throttle(1.0, "Cage observation rejected: %s", error)

    def _history_watchdog(self, _event):
        command = None
        with self.lock:
            if self.last_object_stamp is None:
                return
            age = (rospy.Time.now() - self.last_object_stamp).to_sec()
            if age > self.maximum_history_interval_s and not self.history_hold_active:
                command = self._hold_current_joints()
                self.history_hold_active = True
        if command is not None:
            rospy.logwarn_throttle(
                1.0,
                "PPO held because the visual history interval exceeded %.3f s",
                self.maximum_history_interval_s,
            )
            self.publisher.publish(Float64MultiArray(data=command.tolist()))

    def _to_base(self, message):
        if not message.header.frame_id or message.header.frame_id == self.base_frame:
            result = message
            result.header.frame_id = self.base_frame
            return result
        transform = self.tf_buffer.lookup_transform(
            self.base_frame, message.header.frame_id, message.header.stamp, rospy.Duration(0.03)
        )
        return tf2_geometry_msgs.do_transform_point(message, transform)

    def _target_speed_is_safe(self, target, timestamp):
        workspace_min = np.asarray(self.robot["workspace_min_m"], dtype=float)
        workspace_max = np.asarray(self.robot["workspace_max_m"], dtype=float)
        if np.any(target[:3] < workspace_min) or np.any(target[:3] > workspace_max):
            rospy.logwarn_throttle(1.0, "Cage target is outside configured workspace")
            return False
        if self.last_accepted_target is None:
            return True
        dt = timestamp - self.last_accepted_target_time
        if dt <= 0.0:
            return False
        linear_speed = np.linalg.norm(target[:3] - self.last_accepted_target[:3]) / dt
        first = pose6_to_transform(self.last_accepted_target)
        second = pose6_to_transform(target)
        angular_speed = rotation_distance(first[:3, :3], second[:3, :3]) / dt
        safe = (
            linear_speed
            <= float(self.robot["maximum_tcp_tracking_speed_mps"])
            / float(self.robot["tracking_speed_margin"])
            and angular_speed
            <= float(self.robot["maximum_tcp_tracking_angular_speed_radps"])
            / float(self.robot["tracking_speed_margin"])
        )
        if not safe:
            rospy.logwarn_throttle(
                1.0,
                "Cage target exceeds speed envelope: linear=%.3f angular=%.3f",
                linear_speed,
                angular_speed,
            )
        return safe

    def joint_callback(self, message):
        mapping = {name: index for index, name in enumerate(message.name)}
        if any(name not in mapping for name in self.joint_names):
            return
        with self.lock:
            self.joints = np.array([message.position[mapping[name]] for name in self.joint_names])
            if len(message.velocity) == len(message.name):
                self.joint_velocities = np.array(
                    [message.velocity[mapping[name]] for name in self.joint_names]
                )
            else:
                self.joint_velocities = np.zeros(6)

    def _command_for_visual_event(self, target, timestamp):
        if self.joints is None or self.joint_velocities is None:
            return None
        tcp_pose = self._tcp_pose_in_base()
        state = build_ppo_state(
            target, tcp_pose, self.joints, self.joint_velocities
        )
        state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self.ppo.sample(state_tensor, deterministic=True)
        normalized_action = action.cpu().numpy()[0]
        nominal_period = 1.0 / float(self.timing["detection_rate_hz"])
        if self.last_action_timestamp is None:
            elapsed = nominal_period
        else:
            elapsed = float(timestamp) - float(self.last_action_timestamp)
            if elapsed <= 0.0:
                return None
        configured_delta = np.asarray(
            self.robot["normalized_delta_max_rad_per_vision_update"], dtype=float
        )
        velocity_delta = np.asarray(self.robot["joint_velocity_limit_radps"]) * elapsed
        delta_limit = np.minimum(configured_delta, velocity_delta)
        command = self.joints + delta_limit * np.clip(normalized_action, -1.0, 1.0)
        lower = np.asarray(self.robot["joint_lower_rad"])
        upper = np.asarray(self.robot["joint_upper_rad"])
        self.last_action_timestamp = float(timestamp)
        return np.clip(command, lower, upper)

    def _tcp_pose_in_base(self):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.tcp_frame,
            rospy.Time(0),
            rospy.Duration(0.02),
        ).transform
        quaternion = np.array(
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                np.array(
                    [
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ],
                    dtype=float,
                ),
                Rotation.from_quat(quaternion).as_rotvec(),
            ]
        )

    def _hold_current_joints(self):
        if self.joints is None:
            return None
        self.last_action_timestamp = None
        lower = np.asarray(self.robot["joint_lower_rad"])
        upper = np.asarray(self.robot["joint_upper_rad"])
        return np.clip(self.joints.copy(), lower, upper)


if __name__ == "__main__":
    rospy.init_node("cage_ppo_controller")
    CagePPOController()
    rospy.spin()
