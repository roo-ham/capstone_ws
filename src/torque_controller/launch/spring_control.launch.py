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

    # 1. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    # 2. RViz2
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        # 필요한 경우 rviz 설정파일(.rviz) 경로 추가
        output='screen'
    )

    # 3. Dynamixel 하드웨어 인터페이스 (가정)
    dynamixel_node = Node(
        package=pkg_name,
        executable='pub_posvel_sub_torque',
        name='dynamixel_interface'
    )

    # 4. Spring-Damper 제어기 및 UI 노드
    control_ui_node = Node(
        package=pkg_name,
        executable='spring_damper_cpp',
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        rviz2,
        dynamixel_node,
        # 하드웨어가 뜬 후 제어기가 동작하도록 약간의 지연 시간 부여
        TimerAction(period=2.0, actions=[control_ui_node])
    ])