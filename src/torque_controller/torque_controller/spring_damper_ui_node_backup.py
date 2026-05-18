#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
import os
import threading
from tkinter import *

from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from geometry_msgs.msg import Point

# [추가] 방향(Orientation) 계산을 위한 Scipy Rotation 라이브러리
from scipy.spatial.transform import Rotation as R

class InteractiveSpringDamperNode(Node):
    def __init__(self):
        super().__init__('interactive_spring_node')

        # 1. Pinocchio 모델 로드
        pkg_name = 'torque_controller'
        urdf_filename = 'hand_0926.urdf'
        urdf_path = os.path.join(get_package_share_directory(pkg_name), 'urdf', urdf_filename)
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.tip_ids = [self.model.getFrameId(f) for f in ['FL1EEF', 'FL2EEF', 'FL3EEF']]

        # 2. 제어 변수 (UI와 공유)
        self.K = 15.0
        self.D = 0.8
        self.D_joint = -0.015
        self.D_joint_weight = np.array([2, 1.5, 1, 1] * 3, np.float64)
        
        # [추가] Orientation 제어 변수
        self.K_ori = 0.1          # Orientation Spring Constant
        self.target_roll = 0.0    # RPY (Radian)
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        
        self.q = np.zeros(self.model.nq)
        self.v = np.zeros(self.model.nv)

        # 3. [핵심] q = [0, ..., 0] 일 때의 Task Space 위치 및 방향 계산
        zero_q = np.zeros(self.model.nq)
        pin.framesForwardKinematics(self.model, self.data, zero_q)
        
        # zero_q_pos: 관절이 모두 0일 때의 손가락 끝 위치 (불변)
        self.zero_q_pos = [self.data.oMf[tid].translation.copy() for tid in self.tip_ids]
        
        # [추가] zero_q_rot: 관절이 모두 0일 때의 손가락 끝 방향 (불변, 영점 균형 유지용)
        self.zero_q_rot = [self.data.oMf[tid].rotation.copy() for tid in self.tip_ids]
        
        # curr_pos: 실제 로봇의 현재 위치 (실시간 업데이트됨)
        self.curr_pos = [p.copy() for p in self.zero_q_pos]
        
        # target_pos: 목표 위치. 처음에는 q=0 위치와 동일하게 시작
        self.target_pos = [p.copy() for p in self.zero_q_pos] 

        # 4. Interactive Marker 설정
        self.server = InteractiveMarkerServer(self, "finger_mocap")
        for i in range(3):
            # 마커를 실제 로봇 위치가 아닌 zero_q_pos 에 생성
            self.make_interactive_marker(i, self.zero_q_pos[i])
        self.server.applyChanges()

        # 5. ROS 통신 설정
        self.torque_pub = self.create_publisher(Float64MultiArray, 'hand_joint_torque', 10)
        self.marker_pub = self.create_publisher(Marker, 'finger_triangle_marker', 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)

        # 6. 제어 스레드 (1kHz)
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

    def make_interactive_marker(self, idx, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = f"finger_{idx}"
        int_marker.description = f"Finger {idx+1} Target"
        
        # 마커 초기 위치 지정 (q=0 일 때의 위치)
        int_marker.pose.position.x = position[0]
        int_marker.pose.position.y = position[1]
        int_marker.pose.position.z = position[2]
        int_marker.scale = 0.05 

        # 시각적 구체 마커
        visual_marker = Marker()
        visual_marker.type = Marker.SPHERE
        visual_marker.scale.x = 0.015; visual_marker.scale.y = 0.015; visual_marker.scale.z = 0.015
        visual_marker.color.r = 1.0; visual_marker.color.a = 0.8

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(visual_marker)
        int_marker.controls.append(control)

        # 3축 이동 축 추가
        for axis in ['x', 'y', 'z']:
            move_control = InteractiveMarkerControl()
            move_control.name = f"move_{axis}"
            move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if axis == 'x': move_control.orientation.w = 1.0; move_control.orientation.x = 1.0
            if axis == 'y': move_control.orientation.w = 1.0; move_control.orientation.z = 1.0
            if axis == 'z': move_control.orientation.w = 1.0; move_control.orientation.y = 1.0
            int_marker.controls.append(move_control)

        self.server.insert(int_marker, feedback_callback=self.process_feedback)

    def process_feedback(self, feedback):
        if feedback.event_type == feedback.POSE_UPDATE:
            idx = int(feedback.marker_name.split('_')[1])
            self.target_pos[idx] = np.array([
                feedback.pose.position.x,
                feedback.pose.position.y,
                feedback.pose.position.z
            ])

    def joint_callback(self, msg):
        self.q = np.array(msg.position)
        self.v = np.array(msg.velocity)
        
        # 현재 실제 위치 FK 업데이트
        pin.framesForwardKinematics(self.model, self.data, self.q)
        for i, tid in enumerate(self.tip_ids):
            self.curr_pos[i] = self.data.oMf[tid].translation.copy()

    def control_loop(self):
        rate = self.create_rate(1000) # 1kHz
        while rclpy.ok():
            tau_total = np.zeros(self.model.nv)
            
            # [추가] RPY 슬라이더 값을 통한 Reference Rotation 생성 (한 번만 계산하여 모든 손가락에 공통 적용)
            # RPY 슬라이더가 0이면 단위행렬(Identity)이 되어 초기 방향과 완벽히 균형을 이룸
            R_added = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=False)
            
            for i, tid in enumerate(self.tip_ids):
                # ----------------------------------------------------
                # 1. 위치(Position) 제어
                # ----------------------------------------------------
                error_p = self.target_pos[i] - self.curr_pos[i]
                
                # 자코비안 계산 (LOCAL_WORLD_ALIGNED 기준)
                J = pin.computeFrameJacobian(self.model, self.data, self.q, tid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_v = J[:3, :] # 위치 자코비안
                J_w = J[3:6, :] # 방향 자코비안
                
                v_eef = J_v @ self.v
                force_p = self.K * error_p - self.D * v_eef
                
                # ----------------------------------------------------
                # 2. 방향(Orientation) 제어
                # ----------------------------------------------------
                # a. 초기 방향에 슬라이더 회전을 더하여 Target Rotation 생성
                R_init = R.from_matrix(self.zero_q_rot[i])
                R_target = R_init * R_added
                
                # b. 현재 손가락 끝의 방향
                curr_rot = self.data.oMf[tid].rotation
                R_curr = R.from_matrix(curr_rot)
                
                # c. 방향 오차 계산 (외적 기반의 Rotation Vector 추출)
                # R_err = R_target * R_current^-1
                R_err = R_target * R_curr.inv()
                e_ori = R_err.as_rotvec() 
                
                # d. Orientation Force (Torque) 계산 (댐핑은 생략)
                force_o = self.K_ori * e_ori
                
                # ----------------------------------------------------
                # 3. 최종 토크 합산 (J_v.T * F_p + J_w.T * F_o)
                # ----------------------------------------------------
                tau_total += J_v.T @ force_p + J_w.T @ force_o

            tau_total += self.D_joint * self.D_joint_weight * self.v

            msg = Float64MultiArray()
            msg.data = tau_total.tolist()
            self.torque_pub.publish(msg)
            
            self.publish_triangle_marker()
            rate.sleep()

    def publish_triangle_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP
        marker.id = 99
        marker.scale.x = 0.003
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0
        
        for p in self.curr_pos:
            pt = Point()
            pt.x, pt.y, pt.z = p[0], p[1], p[2]
            marker.points.append(pt)
        marker.points.append(marker.points[0])
        self.marker_pub.publish(marker)

# --- UI 실행 함수 ---
def run_ui(node):
    root = Tk()
    root.title("Spring-Damper Tuning UI")
    root.geometry("380x380")
    
    # 변수 매핑
    k_var = DoubleVar(value=node.K)
    d_var = DoubleVar(value=node.D)
    dj_var = DoubleVar(value=node.D_joint)
    
    # [추가] Orientation 관련 변수 
    k_ori_var = DoubleVar(value=node.K_ori)
    roll_var = DoubleVar(value=0.0)
    pitch_var = DoubleVar(value=0.0)
    yaw_var = DoubleVar(value=0.0)

    def update_params(*args):
        node.K = k_var.get()
        node.D = d_var.get()
        node.D_joint = dj_var.get()
        node.K_ori = k_ori_var.get()
        
        # 사용자는 각도(Degree)로 조절하고, 내부에서는 라디안(Radian)으로 변환하여 사용
        node.target_roll = np.radians(roll_var.get())
        node.target_pitch = np.radians(pitch_var.get())
        node.target_yaw = np.radians(yaw_var.get())

    # Task Spring UI
    Label(root, text="[Position]").grid(row=0, column=0, columnspan=2, pady=(10, 0))
    Label(root, text="Spring (K_p)").grid(row=1, column=0, padx=10)
    Scale(root, from_=0.0, to=50.0, resolution=0.1, orient=HORIZONTAL, variable=k_var, command=update_params).grid(row=1, column=1)
    
    Label(root, text="Damping (D_p)").grid(row=2, column=0, padx=10)
    Scale(root, from_=0.0, to=5.0, resolution=0.01, orient=HORIZONTAL, variable=d_var, command=update_params).grid(row=2, column=1)
    
    Label(root, text="Damping (D_j)").grid(row=3, column=0, padx=10)
    Scale(root, from_=-0.1, to=0.1, resolution=0.001, orient=HORIZONTAL, variable=dj_var, command=update_params).grid(row=3, column=1)

    # Orientation Spring UI
    Label(root, text="[Orientation]").grid(row=4, column=0, columnspan=2, pady=(15, 0))
    Label(root, text="Spring (K_ori)").grid(row=5, column=0, padx=10)
    Scale(root, from_=0.0, to=0.5, resolution=0.01, orient=HORIZONTAL, variable=k_ori_var, command=update_params).grid(row=5, column=1)

    # RPY Slider UI (-180도 ~ 180도)
    Label(root, text="Ref Roll (deg)").grid(row=6, column=0, padx=10)
    Scale(root, from_=-180.0, to=180.0, resolution=1.0, orient=HORIZONTAL, variable=roll_var, command=update_params).grid(row=6, column=1)

    Label(root, text="Ref Pitch (deg)").grid(row=7, column=0, padx=10)
    Scale(root, from_=-180.0, to=180.0, resolution=1.0, orient=HORIZONTAL, variable=pitch_var, command=update_params).grid(row=7, column=1)

    Label(root, text="Ref Yaw (deg)").grid(row=8, column=0, padx=10)
    Scale(root, from_=-180.0, to=180.0, resolution=1.0, orient=HORIZONTAL, variable=yaw_var, command=update_params).grid(row=8, column=1)

    root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveSpringDamperNode()
    
    # ROS2 Spin은 백그라운드 스레드에서 실행
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    # Tkinter UI는 메인 스레드에서 실행 (UI 라이브러리 특성상 메인 스레드 권장)
    run_ui(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
