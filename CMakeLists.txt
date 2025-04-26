cmake_minimum_required(VERSION 3.0.2)
project(yolov8_3d)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  tf2_ros
  message_generation
)

find_package(OpenCV REQUIRED)


generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs geometry_msgs cv_bridge image_transport tf2_ros message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


# 安装Python脚本
catkin_install_python(PROGRAMS
  scripts/detector_3d_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装启动文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# 安装配置文件
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)