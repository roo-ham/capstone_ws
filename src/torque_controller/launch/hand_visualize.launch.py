import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
# 1. 파일 경로 설정
    pkg_path = get_package_share_directory('torque_controller')
    urdf_file = os.path.join(pkg_path, 'urdf', 'hand_0926.urdf')

    # URDF 파일 읽기
    with open(urdf_file, 'r') as inf:
        robot_desc = inf.read()

    # 2. [청소 단계] 기존 노드 강제 종료 명령
    kill_existing_nodes = ExecuteProcess(
        cmd=['pkill', '-f', 'robot_state_publisher'],
        output='screen'
    )

    # 3. [실행 단계] 새로 띄울 노드들 정의
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}],
        arguments=[urdf_file]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # 저장해둔 rviz 설정 파일이 있다면 아래 주석 해제 후 경로 연결
        # arguments=['-d', os.path.join(pkg_path, 'rviz', 'hand_config.rviz')]
    )

    # 4. [지연 실행] 청소 후 2초 뒤에 실행하도록 TimerAction으로 감싸기
    start_nodes_with_delay = TimerAction(
        period=2.0,  # 2.0초 대기 (pkill이 완료될 충분한 시간)
        actions=[
            robot_state_publisher_node,
            rviz_node,
        ]
    )

    # 5. 최종 LaunchDescription 반환
    return LaunchDescription([
        kill_existing_nodes,    # 1. 먼저 죽이고
        start_nodes_with_delay  # 2. 2초 뒤에 다 같이 실행
    ])