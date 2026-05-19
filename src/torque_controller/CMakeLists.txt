cmake_minimum_required(VERSION 3.20)
project(torque_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(robot_interfaces REQUIRED)

find_package(ament_index_cpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

add_executable(tactile_ball_tracker_debug src/tactile_ball_tracker_debug.cpp)
target_include_directories(tactile_ball_tracker_debug PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tactile_ball_tracker_debug PUBLIC
  ${OpenCV_LIBRARIES}
  rt
  pthread
  rclcpp::rclcpp
  Qt5::Widgets
)

add_executable(ball_tracker_node src/ball_tracker_node.cpp)
target_include_directories(ball_tracker_node PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ball_tracker_node PUBLIC
  ${rclcpp_TARGETS}
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
)

# === 신규 추가된 Calibration 전용 UI + 비전 통합 노드 ===
add_executable(ball_calibration_node src/ball_calibration_node.cpp)
target_include_directories(ball_calibration_node PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ball_calibration_node PUBLIC
  ${OpenCV_LIBRARIES}
  rt
  pthread
  rclcpp::rclcpp
  Qt5::Widgets
  Eigen3::Eigen
)

add_executable(pub_posvel_sub_torque src/pub_posvel_sub_torque.cpp)
target_link_libraries(pub_posvel_sub_torque PUBLIC
  ${std_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  dynamixel_sdk::dynamixel_sdk
  rclcpp::rclcpp
)

add_executable(spring_damper_cpp src/spring_damper.cpp)
target_include_directories(spring_damper_cpp PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(spring_damper_cpp PUBLIC
  pinocchio::pinocchio
  rclcpp::rclcpp
  ${visualization_msgs_TARGETS}
  ${ament_index_cpp_TARGETS}
)

add_executable(spring_damper_3axis src/spring_damper_3axis.cpp)
target_include_directories(spring_damper_3axis PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(spring_damper_3axis PUBLIC
  pinocchio::pinocchio
  rclcpp::rclcpp
  ${visualization_msgs_TARGETS}
  ${ament_index_cpp_TARGETS}
)

# (B) Python 노드 설치
install(PROGRAMS
  torque_controller/joint_current_tester.py
  torque_controller/position_controller.py
  torque_controller/visual_ball_tracker_node.py
  torque_controller/balance_controller.py
  torque_controller/balance_controller_ball_est.py
  torque_controller/axis_controller.py
  torque_controller/kalman_filter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  pub_posvel_sub_torque
  spring_damper_cpp
  spring_damper_3axis
  ball_tracker_node
  tactile_ball_tracker_debug
  ball_calibration_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()