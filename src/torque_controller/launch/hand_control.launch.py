import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 패키지 이름 정의
    package_name = 'torque_controller'

    # 2. Launch Argument 설정 (터미널에서 변경 가능하도록)
    # 예: ros2 launch ... k_gain:=50.0
    k_gain_arg = DeclareLaunchArgument(
        'k_gain',
        default_value='10.0',
        description='Stiffness gain for impedance control'
    )

    # 3. 노드 정의
    
    # (1) Fingertip Server (Python) - 정기구학
    fingertip_server_node = Node(
        package=package_name,
        executable='fingertip_server',
        name='fingertip_server',
        output='screen'
    )

    # (2) Jacobian Server (Python) - 토크 계산
    jacobian_server_node = Node(
        package=package_name,
        executable='jacobian_server',
        name='jacobian_server',
        output='screen'
    )

    # (3) Dynamixel Controller (C++) - 모터 제어
    dynamixel_controller_node = Node(
        package=package_name,
        executable='dynamixel_controller',
        name='dynamixel_controller',
        output='screen',
        parameters=[{
            'k_gain': LaunchConfiguration('k_gain')
        }]
    )

    # 4. LaunchDescription 리턴
    return LaunchDescription([
        k_gain_arg,
        fingertip_server_node,
        jacobian_server_node,
        dynamixel_controller_node
    ])