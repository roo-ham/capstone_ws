import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_name = 'torque_controller'
    urdf_file = os.path.join(get_package_share_directory(pkg_name), 'urdf', 'hand_0926.urdf')

    with open(urdf_file, 'r') as inf:
        robot_desc = inf.read()

    show_gui_arg = DeclareLaunchArgument('show_gui', default_value='false')
    show_gui = LaunchConfiguration('show_gui')

    show_qt_gui_arg = DeclareLaunchArgument('show_qt_gui', default_value='true')
    show_qt_gui = LaunchConfiguration('show_qt_gui')

    # 1. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    # 2. Dynamixel hardware interface
    dynamixel_node = Node(
        package=pkg_name,
        executable='pub_posvel_sub_torque',
        name='dynamixel_interface'
    )

    # 3. Spring-Damper controller
    control_ui_node = Node(
        package=pkg_name,
        executable='spring_damper_3axis'
    )

    ball_tracker_node = Node(
        package=pkg_name,
        executable='tactile_ball_tracker_debug',
        name='ball_tracker_node',
        parameters=[{'show_gui': show_gui, 'show_qt_gui': show_qt_gui, 'fps_limit': 100}]
    )

    balance_controller = Node(
        package=pkg_name,
        executable='balance_controller.py',
        name='balance_controller',
        parameters=[{'show_gui': show_gui}]
    )

    return LaunchDescription([
        show_gui_arg,
        show_qt_gui_arg,
        robot_state_publisher,
        dynamixel_node,
        ball_tracker_node,
        balance_controller,
        control_ui_node
    ])
