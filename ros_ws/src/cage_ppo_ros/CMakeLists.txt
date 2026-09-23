cmake_minimum_required(VERSION 3.0.2)
project(cage_ppo_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  tf2_ros
)

catkin_package()

catkin_install_python(
  PROGRAMS scripts/cage_ppo_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

