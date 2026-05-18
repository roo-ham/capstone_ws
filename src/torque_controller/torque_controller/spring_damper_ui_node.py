#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
import os
import threading
import time
import csv
from datetime import datetime
from tkinter import *
from multiprocessing import shared_memory

from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

def fric_compensation_function(x, a, b):
    abs_x = np.abs(x)
    condlist = [
        (abs_x >= 0) & (abs_x < a),
        (abs_x >= a) & (abs_x < a + b),
        (abs_x >= a + b)
    ]
    funclist = [
        lambda z: (b / a) * z,
        lambda z: b - (z - a),
        lambda z: 0
    ]
    return np.sign(x) * np.piecewise(abs_x, condlist, funclist)

class SynchronizedSpringDamperNode(Node):
    def __init__(self):
        super().__init__('sync_spring_node')

        # [추가] Roll, Pitch를 Shared Memory에서 불러올지 여부 결정 플래그
        self.roll_pitch_shared_memory = False 

        # 1. Pinocchio 모델 로드
        pkg_name = 'torque_controller'
        urdf_filename = 'hand_0926.urdf'
        urdf_path = os.path.join(get_package_share_directory(pkg_name), 'urdf', urdf_filename)
        self.model = pin.buildModelFromUrdf(urdf_path)
        
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        self.tip_ids = [self.model.getFrameId(f) for f in ['FL1EEF', 'FL2EEF', 'FL3EEF']]

        # 2. 제어 변수 및 중력 보상 설정
        self.gravity_comp_gain = 1.0
        self.K = 25.0
        self.K_rot = 0.0
        self.D = 2.0
        self.D_joint = 0.0
        self.D_joint_weight = np.array([2, 1.5, 1, 1] * 3, np.float64)
        
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.R_added_mat = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=False).as_matrix()

        self.target_R = R.from_euler('xyz', [0, 0, 0], degrees=False).as_matrix()
        
        self.F_FRIC_STATIC = 0.05
        self.F_FRIC_BIAS = 0.10
        self.FRIC_V_COMPENSATE = 10.0
        
        self.q = np.zeros(self.model.nq)
        self.v = np.zeros(self.model.nv)

        # 3. 초기 위치 및 목표 지점 계산 설정
        zero_q = np.zeros(self.model.nq)
        pin.framesForwardKinematics(self.model, self.data, zero_q)

        self.TRI_RADIUS = 0.05 
        self.zero_target_bias = [np.array([0, 0, 0.273], np.float64) for _ in range(3)]
        self.zero_target_tri = [np.array([-1 * np.sqrt(3/4), 1 * 0.5, 0], np.float64),
                                np.array([1 * np.sqrt(3/4), 1 * 0.5, 0], np.float64),
                                np.array([0, -1, 0], np.float64)]
        self.curr_pos = [self.data.oMf[tid].translation.copy() for tid in self.tip_ids]
        self.target_pos_actual = None

        # 4. 로깅 변수
        self.data_logs = []
        self.start_time = self.get_clock().now()
        self.is_saving = False

        # 5. Shared Memory 연결 
        self._connect_shared_memory()

        # 6. ROS 퍼블리셔 (시각화용만 남김)
        self.triangle_marker_pub = self.create_publisher(Marker, 'finger_triangle_marker', 10)
        self.target_marker_pub = self.create_publisher(Marker, 'finger_target_markers', 10)
        
        # 7. 제어 루프 스레드 및 시각화 타이머
        self.running = True
        self.control_thread = threading.Thread(target=self.shm_control_loop, daemon=True)
        self.control_thread.start()
        
        self.create_timer(1.0 / 30.0, self.publish_visualization_markers)

    def _connect_shared_memory(self):
        """인터페이스 노드에서 생성한 Shared Memory에 연결"""
        try:
            self.shm_state = shared_memory.SharedMemory(name='dxl_state_shm')
            self.shm_cmd = shared_memory.SharedMemory(name='dxl_cmd_shm')
            self.state_array = np.ndarray((2, 12), dtype=np.float64, buffer=self.shm_state.buf)
            self.cmd_array = np.ndarray((12,), dtype=np.float64, buffer=self.shm_cmd.buf)
        except FileNotFoundError:
            self.get_logger().error("Shared memory buffers not found! Please run dynamixel_shm_interface.py first.")
            raise

        # [추가] Roll, Pitch 값을 읽어오기 위한 Shared Memory 연결 및 예외 처리
        if self.roll_pitch_shared_memory:
            try:
                # 1. 먼저 기존 Shared Memory에 연결 시도
                self.shm_rp = shared_memory.SharedMemory(name='rp_val_shm')
                self.rp_array = np.ndarray((2,), dtype=np.float64, buffer=self.shm_rp.buf)
                self.get_logger().info("Successfully connected to existing Roll/Pitch shared memory.")
                
            except FileNotFoundError:
                # 2. 메모리가 없으면 직접 생성
                self.get_logger().warn("Roll/Pitch shared memory not found. Creating a new one...")
                try:
                    # float64(8 bytes) * 2개 = 16 bytes 할당
                    self.shm_rp = shared_memory.SharedMemory(name='rp_val_shm', create=True, size=16)
                    self.rp_array = np.ndarray((2,), dtype=np.float64, buffer=self.shm_rp.buf)
                    
                    # 쓰레기값 방지를 위해 초기값 0.0으로 세팅
                    self.rp_array[:] = [0.0, 0.0]
                    self.get_logger().info("Successfully created Roll/Pitch shared memory.")
                    
                except Exception as create_e:
                    self.get_logger().error(f"Failed to create Roll/Pitch shared memory: {create_e}. Falling back to UI.")
                    self.roll_pitch_shared_memory = False
                    
            except Exception as e:
                # 권한 문제나 기타 예외 발생 시 UI 백업 모드로 전환
                self.get_logger().error(f"Failed to initialize Roll/Pitch shared memory: {e}. Falling back to UI.")
                self.roll_pitch_shared_memory = False

    def shm_control_loop(self):
        """Shared Memory를 통해 직접 통신하는 초고속 제어 루프 (~500Hz)"""
        loop_rate = 0.002 # 2ms
        
        while self.running and rclpy.ok():
            self.compute_and_send_torque()

    def compute_and_send_torque(self):
        """수학적 연산 및 토크 인가 (ROS 콜백 대체) + Orientation 제어 추가"""
        # A. Shared Memory에서 즉시 State 읽기
        self.q = self.state_array[0].copy()
        self.v = self.state_array[1].copy()
        
        # [추가] 플래그가 켜져있을 경우 Shared Memory에서 Roll, Pitch 동적 업데이트
        if self.roll_pitch_shared_memory:
            # rp_array[0] = roll, rp_array[1] = pitch 라고 가정 (라디안 단위 기준)
            # 만약 shared memory 값이 '도(degree)' 단위라면 np.radians()를 씌워주어야 합니다.
            self.target_roll = self.rp_array[0]
            self.target_pitch = self.rp_array[1]
            
            # 받아온 값을 기반으로 회전 행렬 갱신 (Yaw는 기존 UI 값 유지)
            self.R_added_mat = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=False).as_matrix()

        # B. Pinocchio 연산
        pin.framesForwardKinematics(self.model, self.data, self.q)
        tau_gravity = pin.computeGeneralizedGravity(self.model, self.data, self.q)
        
        tau_task = np.zeros(self.model.nv)
        tau_task_damper = np.zeros(self.model.nv)
        R_mat = self.R_added_mat
        
        current_log = {'timestamp': (self.get_clock().now() - self.start_time).nanoseconds / 1e9}
        if not self.target_pos_actual:
            self.target_pos_actual = [np.zeros(3) for _ in range(3)]

        for i, tid in enumerate(self.tip_ids):
            # --- 1. 현재 위치 및 자세(Rotation) 구하기 ---
            self.curr_pos[i] = self.data.oMf[tid].translation.copy()
            curr_R = self.data.oMf[tid].rotation.copy() # 현재 orientation (3x3)
            
            # --- 2. 목표 위치 및 자세 설정 ---
            target_p = R_mat @ self.zero_target_tri[i] * self.TRI_RADIUS + self.zero_target_bias[i]
            self.target_pos_actual[i] = target_p 
            
            # [예시] 목표 자세를 현재 전역 변환 R_mat 구조에 맞추거나 별도 정의 필요
            # 여기서는 기본적으로 R_mat을 목표 자세 기저로 사용한다고 가정 (환경에 맞게 수정 가능)
            target_R = self.target_R.copy() 
            
            # --- 3. 오차(Error) 계산 ---
            # 위치 오차 (Translation Error)
            error_p = target_p - self.curr_pos[i]
            
            # 자세 오차 (Orientation Error): 3x1 벡터 (통상적인 로봇 공학의 기하학적 오차 계산)
            R_error_matrix = target_R @ curr_R.T
            error_R = 0.5 * np.array([
                R_error_matrix[2, 1] - R_error_matrix[1, 2],
                R_error_matrix[0, 2] - R_error_matrix[2, 0],
                R_error_matrix[1, 0] - R_error_matrix[0, 1]
            ])
            
            error_R[0] = 0.0  # X축 회전 무시
            error_R[1] = 0.0  # Y축 회전 무시
            error_R[2] = 0.0  # Z축 회전 무시
            
            # --- 4. 야코비안 추출 (6xN) ---
            J = pin.computeFrameJacobian(self.model, self.data, self.q, tid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_v = J[:3, :]  # Linear Jacobian (3xN)
            J_w = J[3:, :]  # Rotational/Angular Jacobian (3xN)
            
            # --- 5. 가상 렌치(Wrench: 힘 & 토크) 계산 ---
            force_p = self.K * error_p
            force_pd = self.D * (J_v @ self.v)
            
            torque_R = self.K_rot * error_R
            
            # --- 6. 조인트 토크 사상 (J^T @ Wrench) ---
            tau_task += J_v.T @ force_p + J_w.T @ torque_R
            tau_task_damper += J_v.T @ force_pd

            # 로그 저장
            current_log.update({
                f'f{i}_des_x': target_p[0], f'f{i}_curr_x': self.curr_pos[i][0],
                f'f{i}_distance': np.linalg.norm(error_p),
                f'f{i}_rot_error_norm': np.linalg.norm(error_R)
            })

        # C. 토크 계산 (기존 비선형 마찰 및 중력 보상 로직 유지)
        tau_total = tau_task
        cosh_velocity = np.cosh(self.FRIC_V_COMPENSATE * self.v)

        tau_total = tau_total + fric_compensation_function(tau_total, self.F_FRIC_STATIC, self.F_FRIC_BIAS) / cosh_velocity \
                     + (self.gravity_comp_gain * tau_gravity) \
                     - tau_task_damper \
                     + (self.D_joint * self.D_joint_weight * self.v)

        # D. Shared Memory에 즉시 토크 명령 쓰기
        np.copyto(self.cmd_array, tau_total)
        
        self.data_logs.append(current_log)

    def publish_visualization_markers(self):
        now = self.get_clock().now().to_msg()
        
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

def destroy_node(self):
    self.running = False
    # 제어 스레드가 종료될 때까지 아주 잠시 대기
    time.sleep(0.1) 
    
    # Shared Memory 정리 (unlink는 생성한 쪽에서만 하도록 주의)
    try:
        self.shm_state.close()
        self.shm_cmd.close()
        if hasattr(self, 'shm_rp'):
            self.shm_rp.close()
    except Exception as e:
        print(f"SHM close error: {e}")
        
    super().destroy_node()

def save_to_csv(self):
    """로깅 시스템이 종료된 후에도 호출될 수 있으므로 self.get_logger 대신 print 사용"""
    if self.is_saving or not self.data_logs: return
    self.is_saving = True
    filename = datetime.now().strftime("hand_log_%Y%m%d_%H%M%S.csv")
    print(f"Saving data to {filename}... Please wait.")
    try:
        keys = self.data_logs[0].keys()
        with open(filename, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.data_logs)
        print(f"Successfully saved to {filename}")
    except Exception as e:
        print(f"Save failed: {e}")

def run_ui(node):
    root = Tk()
    root.title("Control & Gravity Tuning UI")
    root.geometry("400x750")
    
    k_var = DoubleVar(value=node.K); d_var = DoubleVar(value=node.D); krot_var = DoubleVar(value=node.K_rot)
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
        node.K = k_var.get(); node.D = d_var.get(); node.K_rot = krot_var.get()
        node.gravity_comp_gain = g_var.get() / 100.0
        
        # [주의] Shared Memory 플래그가 False일 때만 UI의 Roll, Pitch 슬라이더 값이 반영됩니다.
        if not node.roll_pitch_shared_memory:
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
        frame = Frame(parent)
        Label(frame, text=label_text, font=("Helvetica", 8)).pack(side="top", pady=0)
        scale = Scale(frame, from_=from_val, to=to_val, resolution=res,
                        orient=HORIZONTAL, variable=var, command=update_params,
                        width=10, sliderlength=15, highlightthickness=0, showvalue=True)
        scale.pack(fill='x', expand=True, padx=2)
        return frame

    def cmd_zero_pos(*args):
        k_var.set(20)
        d_var.set(2.0)
        pos_z_var.set(0.31)
        radius_var.set(0.075)

        fric_static_var.set(0.02)
        fric_bias_var.set(0.1)
        fric_vcomp_var.set(5.0)
        update_params()

    def cmd_unfold(*args):
        pos_z_var.set(0.3)
        radius_var.set(0.12)
        update_params()

    def cmd_grab_top(*args):
        pos_z_var.set(0.264)
        radius_var.set(0.004)
        update_params()

    def cmd_grab_center(*args):
        pos_z_var.set(0.23)
        radius_var.set(0.08)
        update_params()
        time.sleep(1)
        pos_z_var.set(0.235)
        radius_var.set(0.008)
        update_params()

    def cmd_grab_lower(*args):
        pos_z_var.set(0.21)
        radius_var.set(0.08)
        update_params()
        time.sleep(1)
        pos_z_var.set(0.211)
        radius_var.set(0.016)
        update_params()

    Label(root, text="[Shortcut Command]").pack(pady=(1,0))
    Button(root, text="Zero Pose", command=cmd_zero_pos).pack(fill='x', padx=2)
    Button(root, text="Unfold", command=cmd_unfold).pack(fill='x', padx=2)
    Button(root, text="Grab Top", command=cmd_grab_top).pack(fill='x', padx=2)
    Button(root, text="Grab Center", command=cmd_grab_center).pack(fill='x', padx=2)
    Button(root, text="Grab Bottom", command=cmd_grab_lower).pack(fill='x', padx=2)

    Label(root, text="[Gravity Compensation]").pack(pady=(5,0))
    Scale(root, from_=0, to=150, resolution=1, orient=HORIZONTAL, variable=g_var, 
        label="Gravity Comp (%)", command=update_params, width=10).pack(fill='x', padx=20)

    Label(root, text="[Position Control]").pack(pady=(5,0))
    pos_ctrl_frame = Frame(root)
    pos_ctrl_frame.pack(fill='x', padx=20)
    create_compact_scale(pos_ctrl_frame, "K_p", k_var, 0, 100).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(pos_ctrl_frame, "D_p", d_var, 0, 5).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(pos_ctrl_frame, "K_rot", krot_var, 0, 10, 0.001).pack(side=LEFT, expand=True, fill='x')

    Label(root, text="[Reference RPY (deg)]").pack(pady=(5,0))
    rpy_frame = Frame(root)
    rpy_frame.pack(fill='x', padx=20)
    create_compact_scale(rpy_frame, "Roll", r_var, -45, 45, 1).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(rpy_frame, "Pitch", p_var, -45, 45, 1).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(rpy_frame, "Yaw", y_var, -45, 45, 1).pack(side=LEFT, expand=True, fill='x')

    Label(root, text="[Reference XYZ]").pack(pady=(5,0))
    xyz_frame = Frame(root)
    xyz_frame.pack(fill='x', padx=20)
    create_compact_scale(xyz_frame, "X", pos_x_var, -0.2, 0.2, 0.001).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(xyz_frame, "Y", pos_y_var, -0.2, 0.2, 0.001).pack(side=LEFT, expand=True, fill='x')
    create_compact_scale(xyz_frame, "Z", pos_z_var, 0.1, 0.4, 0.001).pack(side=LEFT, expand=True, fill='x')

    create_compact_scale(root, "Radius", radius_var, -0.01, 0.2, 0.001).pack(fill='x', padx=20)

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