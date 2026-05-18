import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    pkg_name = 'torque_controller'
    urdf_file = os.path.join(get_package_share_directory(pkg_name), 'urdf', 'hand_0926.urdf')

    with open(urdf_file, 'r') as inf:
        robot_desc = inf.read()

    # Dynamixel 하드웨어 인터페이스 (가정)
    dynamixel_node = Node(
        package=pkg_name,
        executable='pub_posvel_sub_torque',
        name='dynamixel_interface'
    )

    # Spring-Damper 제어기
    control_node = Node(
        package=pkg_name,
        executable='spring_damper_3axis'
    )

    ball_calibration_node = Node(
        package=pkg_name,
        executable='ball_calibration_node',
        name='ball_calibration_node'
    )

    return LaunchDescription([
        dynamixel_node,
        control_node,
        ball_calibration_node
    ])