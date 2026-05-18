import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'torque_controller'

    # ---------------------------------------------------------
    # [노드 정의]
    # ---------------------------------------------------------
    
    # (1) 다이나믹셀 하드웨어 제어기 (기존 노드 재사용)
    # 실제 모터의 /joint_states를 발행하고, /hand_joint_torque를 수신합니다.
    dynamixel_controller_node = Node(
        package=package_name,
        executable='pub_posvel_sub_torque',
        name='pub_posvel_sub_torque',
        output='screen'
    )

    # (2) Tkinter 기반 Joint 전류 테스트 GUI 노드 (신규 추가)
    # 사용자가 선택한 Joint에 사인파 전류를 인가하고 상태를 모니터링합니다.
    joint_current_tester_node = Node(
        package=package_name,
        executable='joint_current_tester.py', # setup.py 설정에 따라 .py 확장자 제거가 필요할 수 있습니다.
        name='joint_current_tester',
        output='screen'
    )

    # ---------------------------------------------------------
    # [최종 실행] 
    # ---------------------------------------------------------
    return LaunchDescription([
        dynamixel_controller_node,
        joint_current_tester_node
    ])