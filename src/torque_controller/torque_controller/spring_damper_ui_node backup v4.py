#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
import os
import threading
import csv
from datetime import datetime
from tkinter import *

from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

def fric_compensation_function(x, a, b):
    abs_x = np.abs(x)
    condlist = [
        (abs_x >= 0) & (abs_x < a),       # 1번 구간
        (abs_x >= a) & (abs_x < a + b),   # 2번 구간
        (abs_x >= a + b)                  # 3번 구간
    ]
    funclist = [
        lambda z: (b / a) * z,            # y = (b/a) * x
        lambda z: b - (z - a),            # y = b - (x - a) => 기울기 -1
        lambda z: 0                       # y = 0
    ]
    return np.sign(x) * np.piecewise(abs_x, condlist, funclist)

class SynchronizedSpringDamperNode(Node):
    def __init__(self):
        super().__init__('sync_spring_node')

        # 1. Pinocchio 모델 로드 (관성 데이터 포함)
        pkg_name = 'torque_controller'
        urdf_filename = 'hand_0926.urdf'
        urdf_path = os.path.join(get_package_share_directory(pkg_name), 'urdf', urdf_filename)
        self.model = pin.buildModelFromUrdf(urdf_path)
        
        # 중력 가속도 설정 (Z축 아래 방향)
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        self.tip_ids = [self.model.getFrameId(f) for f in ['FL1EEF', 'FL2EEF', 'FL3EEF']]

        # 2. 제어 변수 및 중력 보상 설정
        self.gravity_comp_gain = 0.3
        self.K = 1.0
        self.D = 1.0
        self.D_joint = 0.0
        self.D_joint_weight = np.array([2, 1.5, 1, 1] * 3, np.float64)
        
        # 3. 회전 행렬 최적화 (UI 변경 시에만 업데이트됨)
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.R_added_mat = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=False).as_matrix()
        
        self.F_FRIC_STATIC = 0.005
        self.F_FRIC_BIAS = 0.300
        self.FRIC_V_COMPENSATE = 2.0
        
        self.q = np.zeros(self.model.nq)
        self.v = np.zeros(self.model.nv)

        # 4. 초기 위치 및 목표 지점 계산 설정
        zero_q = np.zeros(self.model.nq)
        pin.framesForwardKinematics(self.model, self.data, zero_q)

        self.TRI_RADIUS = 0.05 
        self.zero_target_bias = [np.array([0, 0, 0.25], np.float64) for _ in range(3)]

        self.zero_target_tri = [np.array([-1 * np.sqrt(3/4), 1 * 0.5, 0], np.float64),
                                np.array([1 * np.sqrt(3/4), 1 * 0.5, 0], np.float64),
                                np.array([0, -1, 0], np.float64)]
        self.curr_pos = [self.data.oMf[tid].translation.copy() for tid in self.tip_ids]
        self.target_pos_actual = [np.zeros(3) for _ in range(3)] # 시각화용 최신 타겟 캐싱

        # 5. 로깅 변수
        self.data_logs = []
        self.start_time = self.get_clock().now()
        self.is_saving = False

        # 6. ROS 퍼블리셔 및 서브스크라이버 설정
        self.torque_pub = self.create_publisher(Float64MultiArray, 'hand_joint_torque', 10)
        self.triangle_marker_pub = self.create_publisher(Marker, 'finger_triangle_marker', 10)
        self.target_marker_pub = self.create_publisher(Marker, 'finger_target_markers', 10)
        
        # [핵심 최적화] 이벤트 동기화: joint_states가 들어올 때마다 제어 계산 및 퍼블리시
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.synchronized_control_callback, 10)

        # [핵심 최적화] 시각화 주기를 30Hz로 분리하여 제어 부하 대폭 감소
        self.create_timer(1.0 / 30.0, self.publish_visualization_markers)

    def synchronized_control_callback(self, msg):
        """joint_states 수신 즉시 실행되는 메인 제어 루프 (약 400~420Hz 예상)"""
        self.q = np.array(msg.position)
        self.v = np.array(msg.velocity)
        
        # A. Pinocchio 모델 연산
        pin.framesForwardKinematics(self.model, self.data, self.q)
        tau_gravity = pin.computeGeneralizedGravity(self.model, self.data, self.q)
        
        tau_task = np.zeros(self.model.nv)
        tau_task_damper = np.zeros(self.model.nv)
        R_mat = self.R_added_mat # UI에서 캐싱된 회전 행렬 사용
        
        current_log = {'timestamp': (self.get_clock().now() - self.start_time).nanoseconds / 1e9}

        for i, tid in enumerate(self.tip_ids):
            # 1. 위치 업데이트
            self.curr_pos[i] = self.data.oMf[tid].translation.copy()
            target_p = R_mat @ self.zero_target_tri[i] * self.TRI_RADIUS + self.zero_target_bias[i]
            self.target_pos_actual[i] = target_p 
            
            # 2. 오차 및 자코비안 계산
            error_p = target_p - self.curr_pos[i]
            # if np.linalg.norm(error_p) > 0.02:
            #     error_p /= np.linalg.norm(error_p)
            J = pin.computeFrameJacobian(self.model, self.data, self.q, tid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_v = J[:3, :]
            
            # 3. Task Space 힘 계산
            force_p = self.K * error_p
            force_pd = self.D * (J_v @ self.v)
            
            tau_task += J_v.T @ force_p
            tau_task_damper += J_v.T @ force_pd

            # 4. 로깅 (속도 향상을 원할 경우 이 부분 주석 처리 권장)
            current_log.update({f'f{i}_des_x': target_p[0], f'f{i}_curr_x': self.curr_pos[i][0]})
            current_log.update({f'f{i}_des_y': target_p[1], f'f{i}_curr_y': self.curr_pos[i][1]})
            current_log.update({f'f{i}_des_z': target_p[2], f'f{i}_curr_z': self.curr_pos[i][2]})
            current_log.update({f'f{i}_distance': np.linalg.norm(error_p)})

        # B. 최종 토크 합산 및 마찰/댐핑 보상
        tau_total = tau_task
        tau_total_scaled = tau_total / self.F_FRIC_STATIC
        cosh_velocity = np.cosh(self.FRIC_V_COMPENSATE * self.v)

        tau_total = tau_total + fric_compensation_function(tau_total, self.F_FRIC_STATIC, self.F_FRIC_BIAS) / cosh_velocity \
                     + (self.gravity_comp_gain * tau_gravity) \
                     - tau_task_damper \
                     + (self.D_joint * self.D_joint_weight * self.v)

        # C. 즉시 퍼블리시
        out_msg = Float64MultiArray()
        out_msg.data = tau_total.tolist()
        self.torque_pub.publish(out_msg)
        
        self.data_logs.append(current_log)

    def publish_visualization_markers(self):
        """30Hz 타이머에 의해 실행되는 가벼운 시각화 함수"""
        now = self.get_clock().now().to_msg()
        
        # 1. 현재 손가락 끝단을 잇는 초록색 삼각형
        tri_marker = Marker()
        tri_marker.header.frame_id = "base_link"
        tri_marker.header.stamp = now
        tri_marker.type = Marker.LINE_STRIP
        tri_marker.id = 99
        tri_marker.scale.x = 0.003
        tri_marker.color.r = 0.0; tri_marker.color.g = 1.0; tri_marker.color.b = 0.0; tri_marker.color.a = 1.0
        for p in self.curr_pos:
            tri_marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
        tri_marker.points.append(tri_marker.points[0]) 
        self.triangle_marker_pub.publish(tri_marker)

        # 2. 목표 위치를 나타내는 빨간색 구체(Sphere List)
        target_marker = Marker()
        target_marker.header.frame_id = "base_link"
        target_marker.header.stamp = now
        target_marker.type = Marker.SPHERE_LIST
        target_marker.id = 100
        target_marker.scale.x = 0.015; target_marker.scale.y = 0.015; target_marker.scale.z = 0.015
        target_marker.color.r = 1.0; target_marker.color.g = 0.0; target_marker.color.b = 0.0; target_marker.color.a = 0.8
        for p in self.target_pos_actual:
            target_marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
        self.target_marker_pub.publish(target_marker)

    def save_to_csv(self):
        if self.is_saving or not self.data_logs: return
        self.is_saving = True
        filename = datetime.now().strftime("hand_log_%Y%m%d_%H%M%S.csv")
        try:
            keys = self.data_logs[0].keys()
            with open(filename, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader(); dict_writer.writerows(self.data_logs)
            self.get_logger().info(f"Saved to {filename}")
        except Exception as e: self.get_logger().error(f"Save failed: {e}")

    def destroy_node(self):
        self.save_to_csv()
        super().destroy_node()

def run_ui(node):
    root = Tk()
    root.title("Control & Gravity Tuning UI")
    root.geometry("400x750")
    
    k_var = DoubleVar(value=node.K); d_var = DoubleVar(value=node.D); dj_var = DoubleVar(value=node.D_joint)
    g_var = DoubleVar(value=node.gravity_comp_gain * 100)
    r_var = DoubleVar(value=0.0); p_var = DoubleVar(value=0.0); y_var = DoubleVar(value=0.0)

    pos_x_var = DoubleVar(value=node.zero_target_bias[0][0])
    pos_y_var = DoubleVar(value=node.zero_target_bias[0][1])
    pos_z_var = DoubleVar(value=node.zero_target_bias[0][2])
    radius_var = DoubleVar(value=node.TRI_RADIUS)

    fric_static_var = DoubleVar(value=node.F_FRIC_STATIC)
    fric_bias_var = DoubleVar(value=node.F_FRIC_BIAS)
    fric_vcomp_var = DoubleVar(value=node.FRIC_V_COMPENSATE)

    def update_params(*args):
        node.K = k_var.get(); node.D = d_var.get(); node.D_joint = dj_var.get()
        node.gravity_comp_gain = g_var.get() / 100.0
        
        # 회전 행렬 갱신 (슬라이더 변경 시에만 작동하여 부하 감소)
        node.target_roll = np.radians(r_var.get())
        node.target_pitch = np.radians(p_var.get())
        node.target_yaw = np.radians(y_var.get())
        node.R_added_mat = R.from_euler('xyz', [node.target_roll, node.target_pitch, node.target_yaw], degrees=False).as_matrix()
        
        for i in range(3):
            node.zero_target_bias[i][0] = pos_x_var.get()
            node.zero_target_bias[i][1] = pos_y_var.get()
            node.zero_target_bias[i][2] = pos_z_var.get()
        node.F_FRIC_STATIC = fric_static_var.get()
        node.F_FRIC_BIAS = fric_bias_var.get()
        node.FRIC_V_COMPENSATE = fric_vcomp_var.get()
        node.TRI_RADIUS = radius_var.get()

    def create_compact_scale(parent, label_text, var, from_val, to_val, res=0.01):
        """세로 크기를 줄인 스케일 위젯을 프레임에 담아 반환"""
        frame = Frame(parent)
        # 레이블 폰트를 작게 설정하여 세로 공간 절약
        Label(frame, text=label_text, font=("Helvetica", 8)).pack(side="top", pady=0)
        
        scale = Scale(frame, from_=from_val, to=to_val, resolution=res,
                        orient=HORIZONTAL, variable=var, command=update_params,
                        width=10,           # 스크롤바 자체의 두께 (기본값 약 15~20)
                        sliderlength=15,    # 슬라이더 손잡이의 길이
                        highlightthickness=0, # 외곽선 제거
                        showvalue=True)     # 현재 값 표시
        scale.pack(fill='x', expand=True, padx=2)
        return frame

    # --- 메인 레이아웃 ---

    def cmd_zero_pos(*args):
        k_var.set(1)
        d_var.set(0.1)
        pos_z_var.set(0.3)
        radius_var.set(0.075)

        fric_static_var.set(0.002)
        fric_bias_var.set(0.1)
        fric_vcomp_var.set(2.0)
        update_params()

    def cmd_unfold(*args):
        pass

    def cmd_grab_small(*args):
        pass

    def cmd_grab_large(*args):
        pass

    Label(root, text="[Shortcut Command]").pack(pady=(1,0))
    Button(root, text="Zero Pose", command=cmd_zero_pos).pack(fill='x', padx=2)
    Button(root, text="Grab Small Object", command=cmd_grab_small).pack(fill='x', padx=2)
    Button(root, text="Grab Large Object", command=cmd_grab_large).pack(fill='x', padx=2)

    # 1. Gravity Compensation
    Label(root, text="[Gravity Compensation]").pack(pady=(5,0))
    Scale(root, from_=0, to=150, resolution=1, orient=HORIZONTAL, variable=g_var, 
        label="Gravity Comp (%)", command=update_params, width=10).pack(fill='x', padx=20)

    # 2. Position Control (가로로 배치하여 공간 절약 가능)
    Label(root, text="[Position Control]").pack(pady=(5,0))
    pos_ctrl_frame = Frame(root)
    pos_ctrl_frame.pack(fill='x', padx=20)
    create_compact_scale(pos_ctrl_frame, "K_p", k_var, 0, 10).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(pos_ctrl_frame, "D_p", d_var, 0, 5).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(pos_ctrl_frame, "D_joint", dj_var, -0.01, 0.01, 0.001).pack(side=LEFT, expand=True, fill='x')

    # 3. Reference RPY (3개 가로 묶음)
    Label(root, text="[Reference RPY (deg)]").pack(pady=(5,0))
    rpy_frame = Frame(root)
    rpy_frame.pack(fill='x', padx=20)
    create_compact_scale(rpy_frame, "Roll", r_var, -45, 45, 1).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(rpy_frame, "Pitch", p_var, -45, 45, 1).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(rpy_frame, "Yaw", y_var, -45, 45, 1).pack(side=LEFT, expand=True, fill='x')

    # 4. Reference XYZ (3개 가로 묶음)
    Label(root, text="[Reference XYZ]").pack(pady=(5,0))
    xyz_frame = Frame(root)
    xyz_frame.pack(fill='x', padx=20)
    create_compact_scale(xyz_frame, "X", pos_x_var, -0.2, 0.2, 0.001).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(xyz_frame, "Y", pos_y_var, -0.2, 0.2, 0.001).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(xyz_frame, "Z", pos_z_var, 0.1, 0.4, 0.001).pack(side=LEFT, expand=True, fill='x')

    # Radius 별도 배치
    create_compact_scale(root, "Radius", radius_var, -0.01, 0.2, 0.001).pack(fill='x', padx=20)

    # 5. Friction Parameter (가로 배치)
    Label(root, text="[Friction Parameter]").pack(pady=(5,0))
    fric_frame = Frame(root)
    fric_frame.pack(fill='x', padx=20)
    create_compact_scale(fric_frame, "Static", fric_static_var, 0.001, 0.05, 0.001).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(fric_frame, "Bias", fric_bias_var, 0, 0.5, 0.001).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(fric_frame, "V_Comp", fric_vcomp_var, 0.1, 10.0, 0.01).pack(side=LEFT, expand=True, fill='x')
    
    root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedSpringDamperNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try: 
        run_ui(node)
    except KeyboardInterrupt: 
        pass
    finally: 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()