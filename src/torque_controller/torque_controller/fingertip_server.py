#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from robot_interfaces.srv import ComputeFingertipPositions

class FingertipServer(Node):

    def __init__(self):
        super().__init__('fingertip_server')

        # 1. URDF 로드 (패키지 이름 확인 필요)
        pkg_name = 'torque_controller'
        urdf_filename = 'hand_0926.urdf'

        try:
            pkg_share = get_package_share_directory(pkg_name)
            urdf_path = os.path.join(pkg_share, 'urdf', urdf_filename)
        except Exception:
            self.get_logger().error(f"Could not find package '{pkg_name}'")
            return

        # 2. 모델 생성
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # 3. 끝점(End-Effector) 프레임 정의
        # URDF 상의 마지막 링크 이름과 정확히 일치해야 합니다.
        self.tip_frames = ['FL1EEF', 'FL2EEF', 'FL3EEF']
        self.tip_ids = []

        # 프레임 ID 찾기 및 검증
        for frame_name in self.tip_frames:
            if not self.model.existFrame(frame_name):
                self.get_logger().error(f"Frame '{frame_name}' not found in URDF!")
            else:
                frame_id = self.model.getFrameId(frame_name)
                self.tip_ids.append(frame_id)
                self.get_logger().info(f"Found frame '{frame_name}' -> ID: {frame_id}")

        self.get_logger().info(f"Fingertip Server Ready. Joints: {self.model.nq}")
        
        # 4. 서비스 생성
        self.srv = self.create_service(
            ComputeFingertipPositions,
            'compute_fingertip_positions',
            self.compute_callback
        )

    def compute_callback(self, request, response):
        """
        Input: joint_positions (12개)
        Output: tip_1_pos, tip_2_pos, tip_3_pos (각 [x, y, z])
        """
        try:
            q = np.array(request.joint_positions)

            # --- 안전 장치: 관절 개수 확인 ---
            if len(q) != self.model.nq:
                self.get_logger().warn(f"Mismatch! Expected {self.model.nq} joints, got {len(q)}")
                response.success = False
                return response

            # --- 정기구학(FK) 계산 ---
            # 1. 모델의 모든 조인트와 프레임 위치 업데이트
            pin.framesForwardKinematics(self.model, self.data, q)

            # 2. 각 손가락 끝의 위치 추출
            # data.oMf[frame_id]는 Global Base(World) 기준의 Transform 행렬입니다.
            # .translation 속성이 [x, y, z] 벡터입니다.
            
            pos_1 = self.data.oMf[self.tip_ids[0]].translation
            pos_2 = self.data.oMf[self.tip_ids[1]].translation
            pos_3 = self.data.oMf[self.tip_ids[2]].translation

            # 3. 결과 반환
            response.tip_1_position = pos_1.tolist()
            response.tip_2_position = pos_2.tolist()
            response.tip_3_position = pos_3.tolist()
            response.success = True
            
            # 디버그용 로그 (필요 시 주석 해제)
            # self.get_logger().info(f"Tip 1: {np.round(pos_1, 3)}")

        except Exception as e:
            self.get_logger().error(f"FK Calculation failed: {e}")
            response.success = False

        return response

def main(args=None):
    rclpy.init(args=args)
    node = FingertipServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()