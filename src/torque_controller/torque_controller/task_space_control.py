#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import time                 # [추가] 시간 측정을 위한 모듈
from functools import partial # [추가] 콜백에 변수를 넘겨주기 위한 모듈

# 메시지 타입 임포트
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# 사용자 정의 서비스 인터페이스
from robot_interfaces.srv import ComputeFingertipPositions, ComputeTorque

class ServiceBasedController(Node):
    def __init__(self):
        super().__init__('service_based_controller')

        # --- [설정] 제어 게인 (P Gain) ---
        self.Kp = 5.0
        self.Kd = [0.05, 0.1, 0.05, 0.07] * 3 # joint 별 damping ratio

        # --- [1] Publisher & Subscriber ---
        self.torque_pub = self.create_publisher(
            Float64MultiArray, 'hand_joint_torque', 10)
        
        self.target_sub = self.create_subscription(
            Point, 'target_position', self.target_callback, 10)
        
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # --- [2] Service Clients ---
        self.fk_client = self.create_client(
            ComputeFingertipPositions, 'compute_fingertip_positions')
        
        self.dynamics_client = self.create_client(
            ComputeTorque, 'compute_torque_from_force')

        # 서비스가 뜰 때까지 대기
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for FK service...')
        while not self.dynamics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Dynamics service...')

        # --- [3] 내부 변수 ---
        self.target_pos = np.array([0.0, 0.0, 0.5])
        self.current_joints = None
        self.current_velocity = None
        
        # 제어 루프 (50Hz)
        self.timer = self.create_timer(0.001, self.control_loop)
        self.get_logger().info("Service Based Controller Started!")

    def target_callback(self, msg):
        self.target_pos = np.array([msg.x, msg.y, msg.z])
        # self.get_logger().info("Target Received")

    def joint_state_callback(self, msg):
        self.current_joints = np.array(msg.position)
        self.current_velocity = np.array(msg.velocity)

    def control_loop(self):
        """ 주기적으로 실행되는 메인 루프 (Step 1 시작) """
        if self.current_joints is None:
            return

        # [시간 측정] 루프 시작 시간 기록
        start_time = time.perf_counter()

        # 1. FK 서비스 요청 생성
        fk_req = ComputeFingertipPositions.Request()
        fk_req.joint_positions = self.current_joints.tolist()

        # 2. 비동기 호출 
        # [수정] partial을 사용하여 start_time을 step2로 전달
        future = self.fk_client.call_async(fk_req)
        future.add_done_callback(
            partial(self.step2_calculate_force, start_time=start_time)
        )

    def step2_calculate_force(self, future, start_time):
        """ 
        Step 2: FK 결과 수신 
        Note: partial을 썼기 때문에 future가 마지막 인자로 들어오거나, 
        함수 정의 순서에 주의해야 합니다. 
        (partial(func, arg1) -> func(arg1, future) 순서로 호출됨)
        """
        # [시간 측정] FK 서비스 종료 시간
        fk_done_time = time.perf_counter()
        
        try:
            fk_res = future.result()
            if not fk_res.success:
                self.get_logger().warn("FK Service returned failure")
                return

            # [DEBUG] FK 소요 시간 출력
            dt_fk = (fk_done_time - start_time) * 1000.0 # ms 단위
            # self.get_logger().info(f"[Timing] FK Service: {dt_fk:.2f} ms")

            # ... 힘 계산 로직 (기존과 동일) ...
            tips_matrix = np.array([
                fk_res.tip_1_position,
                fk_res.tip_2_position,
                fk_res.tip_3_position
            ])
            centroid = np.mean(tips_matrix, axis=0) 
            errors = self.target_pos - tips_matrix
            forces_3d = errors
            moments = np.zeros_like(forces_3d)
            forces_6d_matrix = np.hstack([forces_3d, moments])

            self.target_pos = centroid

            # 힘 크기 균일화
            norms = np.sum(np.linalg.norm(forces_6d_matrix, axis=1, keepdims=True))
            epsilon = 1e-10
            forces_6d_matrix = forces_6d_matrix / (norms + epsilon)
            forces_6d_matrix *= self.Kp

            # 3. Dynamics(Torque) 서비스 요청 생성
            trq_req = ComputeTorque.Request()
            trq_req.joint_positions = self.current_joints.tolist()
            trq_req.task_force_1 = forces_6d_matrix[0].tolist()
            trq_req.task_force_2 = forces_6d_matrix[1].tolist()
            trq_req.task_force_3 = forces_6d_matrix[2].tolist()

            # 4. 비동기 호출
            # [수정] start_time과 fk_done_time을 step3로 전달
            future_trq = self.dynamics_client.call_async(trq_req)
            future_trq.add_done_callback(
                partial(self.step3_publish_torque, 
                        start_time=start_time, 
                        fk_done_time=fk_done_time)
            )

        except Exception as e:
            self.get_logger().error(f'FK Call failed: {e}')

    def step3_publish_torque(self, future, start_time, fk_done_time):
        """ Step 3: 토크 결과 수신 및 발행 """
        # [시간 측정] Dynamics 서비스 종료 및 전체 종료 시간
        end_time = time.perf_counter()

        try:
            trq_res = future.result()
            if not trq_res.success:
                self.get_logger().warn("Torque Service returned failure")
                return

            # 5. 최종 토크 퍼블리시
            msg = Float64MultiArray()
            # Kd 항 추가 여부는 기존 코드 유지
            if self.current_velocity is not None:
                msg.data = trq_res.joint_torques - self.Kd * self.current_velocity 
            
            self.torque_pub.publish(msg)

            # [DEBUG] 시간 계산 및 출력
            dt_fk = (fk_done_time - start_time) * 1000.0
            dt_dyn = (end_time - fk_done_time) * 1000.0
            dt_total = (end_time - start_time) * 1000.0

            # self.get_logger().info(
            #     f"[Timing] FK: {dt_fk:.2f}ms | Dyn: {dt_dyn:.2f}ms | Total: {dt_total:.2f}ms"
            # )

        except Exception as e:
            self.get_logger().error(f'Torque Call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ServiceBasedController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()