import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'torque_controller'

    # ---------------------------------------------------------
    # 2. [노드 정의]
    # ---------------------------------------------------------
    
    # (1) 정기구학 서버
    fingertip_server_node = Node(
        package=package_name,
        executable='fingertip_server.py', # setup.py 설정에 따라 .py 제거 필요할 수 있음
        name='fingertip_server',
        output='screen'
    )

    # (2) 동역학 서버
    jacobian_server_node = Node(
        package=package_name,
        executable='jacobian_server.py', # setup.py 설정에 따라 .py 제거 필요할 수 있음
        name='jacobian_server',
        output='screen'
    )

    # (3) 다이나믹셀 하드웨어 제어기
    dynamixel_controller_node = Node(
        package=package_name,
        executable='pub_posvel_sub_torque',
        name='pub_posvel_sub_torque',
        output='screen'
    )

    # (4) 테스크 스페이스 제어기
    task_space_control_node = Node(
        package=package_name,
        executable='task_space_control.py', # setup.py 설정에 따라 .py 제거 필요할 수 있음
        name='task_space_control',
        output='screen'
    )

    # ---------------------------------------------------------
    # 3. [최종 실행] 타이머 없이 리스트로 반환
    # ---------------------------------------------------------
    return LaunchDescription([
        fingertip_server_node,
        jacobian_server_node,
        dynamixel_controller_node,
        task_space_control_node
    ])