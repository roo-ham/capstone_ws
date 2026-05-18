#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import numpy as np
from multiprocessing import shared_memory
import threading
import tkinter as tk
from tkinter import ttk
import time
import os
import json
import csv
import cv2  # OpenCV 추가

F_TOTAL_MIN = 10
PLATE_MASS = 0.2347  # kg
N_TO_GF = 101.97162

class BallBalancingNode(Node):
    def __init__(self):
        super().__init__('ball_balancing_node')

        self.dt = 0.002
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        self.loop_count = 0
        self.last_freq_time = time.time()
        self.actual_freq = 0.0

        # SHM 변수
        self.shm_ball = None
        self.ball_state_array = None
        self.shm_pose = None
        self.pose_array = None
        self.shm_eef = None
        self.eef_array = None
        self.shm_eef_force = None
        self.eef_force_array = None
        self.shm_eef_dot = None
        self.eef_dot_array = None
        
        self.shm_connected_ball = False
        self.shm_connected_pose = False

        # 플래그 및 모드
        self.is_paused = False
        self.ball_detected = False
        self.prev_ball_detected = False

        # 포스 데이터 변수
        self.curr_f1_raw = 0.0; self.curr_f2_raw = 0.0; self.curr_f3_raw = 0.0
        self.curr_f1_res = 0.0; self.curr_f2_res = 0.0; self.curr_f3_res = 0.0

        # 판 무게 하중 보정 2차 함수 계수 (force_coeffs.json 우선, 없으면 기본값)
        self.force_coeffs = self.load_force_coeffs()
        
        # 공 위치 추정 계수
        self.poly_coeffs_x = np.array([
            -0.018512, -0.002182, -0.000442, -0.001920, -0.000000, 
            -0.000014,  0.000032,  0.000002,  0.000000,  0.000019
        ])
        self.poly_coeffs_y = np.array([
            -0.060019, -0.000129, -0.000021, -0.009140,  0.000002, 
            -0.000008, -0.000069,  0.000005,  0.000010, -0.000009
        ])
        
        # 공 상태 변수
        self.curr_ball_x = 0.0; self.curr_ball_y = 0.0
        self.curr_ball_vx = 0.0; self.curr_ball_vy = 0.0
        self.target_ball_x = 0.0; self.target_ball_y = 0.0
        self.target_ball_vx = 0.0; self.target_ball_vy = 0.0

        self.v_history = np.zeros((5, 2))
        self.v_history_idx = 0

        # Z-axis velocity estimation (게인 스케줄링용)
        self.z_vel_filt = 0.0

        # 기본 게인 초기화
        self.Kp_ball = 1.0
        self.Kd_ball = 0.1

        self.csv_data = [] 
        self.target_data = np.zeros(12)
        self.f_xyz_manual = np.zeros(3, np.float64)

        self.config_file = 'balance_config.json'
        self.load_config()

        # --- Kalman Filter 초기화 ---
        self.kf_dt = self.dt
        self.kf_g = 9.81
        self.kf_x = np.zeros(4)       # [x, y, vx, vy]
        self.kf_P = np.diag([0.01, 0.01, 0.1, 0.1])
        # State transition matrix
        dt = self.kf_dt
        self.kf_F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        # Control matrix: [θ_pitch, -θ_roll] → ball accel
        self.kf_B = np.array([[9.81*dt**2/2, 0],[0, -9.81*dt**2/2],[9.81*dt,0],[0,-9.81*dt]])
        # Measurement matrix
        self.kf_H = np.array([[1,0,0,0],[0,1,0,0]])
        # Process noise (small: ~2% of tilt accel, rolling resistance etc.)
        q = 0.02 * 9.81 * 0.17
        self.kf_Q = np.diag([q**2*dt**4/4, q**2*dt**4/4, q**2*dt**2, q**2*dt**2])

        self.zero_pub = self.create_publisher(Empty, 'set_force_zero', 10)
        self.get_logger().info("Ball Balancing Node Starting...")

        self.declare_parameter('show_gui', True)
        self.show_gui = self.get_parameter('show_gui').value
        self.shutdown_flag = False
        self.pause_control()

    def load_force_coeffs(self):
        defaults = np.array([
            [140.2043,  -4.0341,  -4.5639,  -0.1540,  -0.2591,  -0.1475],
            [140.4777,  -2.8153,  -5.5666,   0.1469,   0.0259,   0.6568],
            [190.4393,  -7.3359,  -5.3972,   0.1569,   0.0036,   0.3956]
        ])
        try:
            with open('force_coeffs.json', 'r') as f:
                j = json.load(f)
            coeffs = np.array([
                [j['sensors'][i][f'c{k}'] for k in range(6)]
                for i in range(3)
            ])
            self.get_logger().info(f"Loaded force_coeffs from JSON: c0=[{coeffs[0][0]:.1f}, {coeffs[1][0]:.1f}, {coeffs[2][0]:.1f}]")
            return coeffs
        except Exception as e:
            self.get_logger().error(f"force_coeffs.json load failed: {e}")
            return defaults

    def kf_predict(self, u_pitch, u_roll):
        """Kalman predict step: tilt command → ball acceleration"""
        u = np.array([u_pitch, u_roll])
        self.kf_x = self.kf_F @ self.kf_x + self.kf_B @ u
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q

    def kf_update(self, z_x, z_y, f_total):
        """Kalman update step: CoP measurement with adaptive R"""
        if f_total < 0:
            return  # dead measurement, skip update
        elif f_total < 10:
            r_val = 1e-4   # 1cm² var
        elif f_total < 30:
            r_val = 1e-6   # 1mm² var
        else:
            r_val = 1e-8   # 0.1mm² var
        R = np.array([[r_val, 0], [0, r_val]])
        z = np.array([z_x, z_y])
        y = z - self.kf_H @ self.kf_x  # innovation
        S = self.kf_H @ self.kf_P @ self.kf_H.T + R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(4) - K @ self.kf_H) @ self.kf_P

    def get_expected_force(self, roll_deg, pitch_deg, sensor_idx):
        c = self.force_coeffs[sensor_idx]
        x = roll_deg; y = pitch_deg
        return c[0] + c[1]*x + c[2]*y + c[3]*(x**2) + c[4]*(y**2) + c[5]*(x*y)

    def load_config(self):
        self.max_tilt_deg = 8.0
        self.MAX_TILT_RAD = np.radians(self.max_tilt_deg)
        self.roll_offset_rad = 0.0
        self.pitch_offset_rad = 0.0
        
        target_defaults = [0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 10.0, 1.0, 0.2, 0.2, 50.0, 5.0]
        self.target_data[:] = target_defaults

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    cfg = json.load(f)
                    self.max_tilt_deg = cfg.get('max_tilt_deg', self.max_tilt_deg)
                    self.MAX_TILT_RAD = np.radians(self.max_tilt_deg)
                    self.roll_offset_rad = cfg.get('roll_offset_rad', self.roll_offset_rad)
                    self.pitch_offset_rad = cfg.get('pitch_offset_rad', self.pitch_offset_rad)
                    
                    self.Kp_ball = cfg.get('Kp_ball', self.Kp_ball)
                    self.Kd_ball = cfg.get('Kd_ball', self.Kd_ball)

                    for i in range(5, 12):
                        idx_str = str(i)
                        if idx_str in cfg.get('target_data', {}):
                            self.target_data[i] = cfg['target_data'][idx_str]
                self.get_logger().info("Configuration loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {e}")

    def save_config(self):
        cfg = {
            'max_tilt_deg': self.max_tilt_deg,
            'roll_offset_rad': self.roll_offset_rad,
            'pitch_offset_rad': self.pitch_offset_rad,
            'Kp_ball': self.Kp_ball,
            'Kd_ball': self.Kd_ball,
            'force_c0_s1': self.force_coeffs[0][0],
            'force_c0_s2': self.force_coeffs[1][0],
            'force_c0_s3': self.force_coeffs[2][0],
            'target_data': {str(i): self.target_data[i] for i in range(5, 12)}
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(cfg, f, indent=4)
            # also save force_coeffs (c0 may have been adjusted via UI sliders)
            fc = {'sensors': [
                {f'c{j}': self.force_coeffs[i][j] for j in range(6)}
                for i in range(3)
            ]}
            with open('force_coeffs.json', 'w') as ff:
                json.dump(fc, ff, indent=4)
            self.get_logger().info("Configuration + force_coeffs saved successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to save config: {e}")

    def set_force_zero(self):
        self.zero_pub.publish(Empty())

    def pause_control(self):
        self.is_paused = True

    def resume_control(self):
        self.is_paused = False

    def connect_shared_memory(self):
        if not self.shm_connected_ball:
            try:
                if self.shm_ball is None:
                    self.shm_ball = shared_memory.SharedMemory(name='ball_state_shm')
                    self.ball_state_array = np.ndarray((4,), dtype=np.float64, buffer=self.shm_ball.buf)
                self.shm_connected_ball = True
            except FileNotFoundError:
                pass

        if not self.shm_connected_pose:
            if os.path.exists('/dev/shm/target_pose_shm') and os.path.exists('/dev/shm/eef_pos_shm') and os.path.exists('/dev/shm/eef_force_shm') and os.path.exists('/dev/shm/eef_dot_shm'):
                self.shm_pose = shared_memory.SharedMemory(name='target_pose_shm', create=False)
                self.pose_array = np.ndarray((12,), dtype=np.float64, buffer=self.shm_pose.buf)
                
                self.shm_eef = shared_memory.SharedMemory(name='eef_pos_shm', create=False)
                self.eef_array = np.ndarray((9,), dtype=np.float64, buffer=self.shm_eef.buf)
                
                self.shm_eef_force = shared_memory.SharedMemory(name='eef_force_shm', create=False)
                self.eef_force_array = np.ndarray((3,), dtype=np.float64, buffer=self.shm_eef_force.buf)

                self.shm_eef_dot = shared_memory.SharedMemory(name='eef_dot_shm', create=False)
                self.eef_dot_array = np.ndarray((9,), dtype=np.float64, buffer=self.shm_eef_dot.buf)

                from multiprocessing.resource_tracker import unregister
                unregister(self.shm_pose._name, 'shared_memory')
                unregister(self.shm_eef._name, 'shared_memory')
                unregister(self.shm_eef_force._name, 'shared_memory')
                unregister(self.shm_eef_dot._name, 'shared_memory')
                
                self.pose_array[:] = self.target_data[:]
                self.shm_connected_pose = True

    def control_loop(self):
        if self.shm_connected_pose and (not os.path.exists('/dev/shm/target_pose_shm') or not os.path.exists('/dev/shm/eef_pos_shm') or not os.path.exists('/dev/shm/eef_force_shm') or not os.path.exists('/dev/shm/eef_dot_shm')):
            os._exit(0)

        self.loop_count += 1
        now = time.time()
        if now - self.last_freq_time >= 1.0:
            self.actual_freq = self.loop_count / (now - self.last_freq_time)
            self.loop_count = 0
            self.last_freq_time = now

        self.connect_shared_memory()
        if not self.shm_connected_ball or not self.shm_connected_pose:
            return

        try:
            # --- 1. 원시 힘 데이터 읽기 ---
            curr_raw = np.array(self.ball_state_array)
            self.curr_f1_raw, self.curr_f2_raw, self.curr_f3_raw, _ = curr_raw

            # --- 2. 손가락 Z속도(global frame) 평균 → 게인 스케줄링용 ---
            if self.eef_dot_array is not None:
                f1_vz = self.eef_dot_array[2]
                f2_vz = self.eef_dot_array[5]
                f3_vz = self.eef_dot_array[8]
                z_vel_raw = (f1_vz + f2_vz + f3_vz) / 3.0
                self.z_vel_filt = 0.85 * self.z_vel_filt + 0.15 * z_vel_raw
            else:
                self.z_vel_filt = 0.0

            # --- 3. 판 무게 보정 (2차 모델) ---
            roll_deg = np.degrees(self.target_data[3])
            pitch_deg = np.degrees(self.target_data[4])

            f1_exp = self.get_expected_force(roll_deg, pitch_deg, 0)
            f2_exp = self.get_expected_force(roll_deg, pitch_deg, 1)
            f3_exp = self.get_expected_force(roll_deg, pitch_deg, 2)

            # --- 4. 정적 보정만 적용한 잔차 (eef_force, 관성 보상 제거) ---
            self.curr_f1_res = self.curr_f1_raw - f1_exp
            self.curr_f2_res = self.curr_f2_raw - f2_exp
            self.curr_f3_res = self.curr_f3_raw - f3_exp

            # --- 5. CoP 계산 (raw measurement) ---
            f_total = self.curr_f1_res + self.curr_f2_res + self.curr_f3_res

            avg_eef_x = (self.eef_array[0] + self.eef_array[3] + self.eef_array[6]) / 3.0
            avg_eef_y = (self.eef_array[1] + self.eef_array[4] + self.eef_array[7]) / 3.0

            if f_total > 0.0:
                cop_x = (self.eef_array[0] * self.curr_f1_res +
                         self.eef_array[3] * self.curr_f2_res +
                         self.eef_array[6] * self.curr_f3_res) / f_total
                cop_y = (self.eef_array[1] * self.curr_f1_res +
                         self.eef_array[4] * self.curr_f2_res +
                         self.eef_array[7] * self.curr_f3_res) / f_total
                bx = cop_x - avg_eef_x
                by = cop_y - avg_eef_y
            else:
                bx = 0.0
                by = 0.0

            # --- 6. Kalman Filter: 공 상태 추정 ---
            # Predict: tilt command → expected ball dynamics
            pitch_cmd = self.target_data[4] - self.pitch_offset_rad
            roll_cmd = -(self.target_data[3] - self.roll_offset_rad)
            self.kf_predict(pitch_cmd, roll_cmd)

            # Update: CoP measurement (adaptive R based on f_total)
            if f_total > 0:
                self.kf_update(bx, by, f_total)

            # Use Kalman state for ball position/velocity
            self.curr_ball_x = self.kf_x[0]
            self.curr_ball_y = self.kf_x[1]
            self.curr_ball_vx = self.kf_x[2]
            self.curr_ball_vy = self.kf_x[3]

            self.ball_detected = (f_total >= F_TOTAL_MIN)
            self.prev_ball_detected = self.ball_detected or self.prev_ball_detected

            # --- 7. PD 제어 + Z속도 기반 적응형 게인 ---
            z_vel_mag = abs(self.z_vel_filt)
            if z_vel_mag > 0.01:
                gain_scale = max(0.15, 1.0 - z_vel_mag / 0.05)
            else:
                gain_scale = 1.0

            Kp_active = self.Kp_ball * gain_scale
            Kd_active = self.Kd_ball * gain_scale

            cmd_tilt = np.zeros(2)
            if not self.is_paused and self.ball_detected:
                err_x = self.target_ball_x - self.curr_ball_x
                err_y = self.target_ball_y - self.curr_ball_y
                cmd_tilt[0] = Kp_active * err_x + Kd_active * (0.0 - self.curr_ball_vx)
                cmd_tilt[1] = Kp_active * err_y + Kd_active * (0.0 - self.curr_ball_vy)

            cmd_tilt = np.clip(cmd_tilt, -self.MAX_TILT_RAD, self.MAX_TILT_RAD)

            self.target_data[0:3] = np.zeros(3)
            self.target_data[3] = -cmd_tilt[1] + self.roll_offset_rad
            self.target_data[4] = cmd_tilt[0] + self.pitch_offset_rad
            self.pose_array[:] = self.target_data[:]

            # --- 8. CSV 로깅 ---
            eef_vel = list(self.eef_dot_array) if self.eef_dot_array is not None else [0.0]*9
            eef_cur = list(self.eef_force_array) if (self.shm_connected_pose and self.eef_force_array is not None) else [0.0]*3
            f_total = self.curr_f1_res + self.curr_f2_res + self.curr_f3_res
            cop_x_raw = (self.eef_array[0]*self.curr_f1_res + self.eef_array[3]*self.curr_f2_res + self.eef_array[6]*self.curr_f3_res) if f_total > 0 else 0.0
            cop_y_raw = (self.eef_array[1]*self.curr_f1_res + self.eef_array[4]*self.curr_f2_res + self.eef_array[7]*self.curr_f3_res) if f_total > 0 else 0.0
            self.csv_data.append([
                now, self.target_data[3], self.target_data[4],
                self.curr_f1_raw, self.curr_f2_raw, self.curr_f3_raw,
                self.curr_f1_res, self.curr_f2_res, self.curr_f3_res,
                f1_exp, f2_exp, f3_exp,
                self.curr_ball_x, self.curr_ball_y, self.curr_ball_vx, self.curr_ball_vy,
                self.z_vel_filt, gain_scale,
                self.target_data[2], self.target_data[5], self.target_data[6],
                cop_x_raw, cop_y_raw, f_total,
                bx, by,  # raw CoP before Kalman
                self.kf_P.trace()
            ] + eef_vel + eef_cur)

        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")

    def destroy_node(self):
        if self.shm_ball: self.shm_ball.close()
        if self.shm_pose: self.shm_pose.close()
        if self.shm_eef: self.shm_eef.close()
        if self.shm_eef_force: self.shm_eef_force.close()
        if self.shm_eef_dot: self.shm_eef_dot.close()
        if self.show_gui: cv2.destroyAllWindows()
        super().destroy_node()

class TuningUI:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Ball Balancing PD Controller (MA + Hysteresis)")
        self.root.geometry("1100x780") 
        self.root.resizable(False, False) 

        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 11, "bold"))
        style.configure("Data.TLabel", font=("Consolas", 10))
        style.configure("DataHeader.TLabel", font=("Consolas", 11, "bold"))
        
        left_col = ttk.Frame(self.root, padding=5, width=530)
        left_col.pack(side='left', fill='both', expand=False, padx=5)
        left_col.pack_propagate(False) 

        right_col = ttk.Frame(self.root, padding=5, width=530)
        right_col.pack(side='right', fill='both', expand=False, padx=5)
        right_col.pack_propagate(False)
        
        # --- [Left] 실시간 모니터링 모듈 ---
        state_frame = ttk.LabelFrame(left_col, text=" Status Monitor ", padding=5)
        state_frame.pack(fill='x', pady=2)
        
        self.lbl_curr_raw = ttk.Label(state_frame, text="Raw Forces", style="DataHeader.TLabel", width=65)
        self.lbl_curr_raw.pack(anchor='w', pady=2)
        self.lbl_curr_res = ttk.Label(state_frame, text="Residues", style="DataHeader.TLabel", foreground="blue", width=65)
        self.lbl_curr_res.pack(anchor='w', pady=2)
        
        ttk.Separator(state_frame, orient='horizontal').pack(fill='x', pady=5)
        
        self.lbl_ball_status = ttk.Label(state_frame, text="Ball: NOT DETECTED", style="DataHeader.TLabel", foreground="red", width=45)
        self.lbl_ball_status.pack(anchor='w', pady=2)
        self.lbl_ball_pos = ttk.Label(state_frame, text="Pos (X, Y):      - ,      -", style="Data.TLabel", width=45)
        self.lbl_ball_pos.pack(anchor='w', pady=2)
        self.lbl_ball_vel = ttk.Label(state_frame, text="Vel (Vx, Vy):     - ,      -", style="Data.TLabel", width=45)
        self.lbl_ball_vel.pack(anchor='w', pady=2)

        self.lbl_actual_freq = ttk.Label(state_frame, text="Actual Freq: 0.0 Hz", style="DataHeader.TLabel", foreground="green", width=25)
        self.lbl_actual_freq.pack(anchor='w', pady=(10, 0))

        self.lbl_eef_force = ttk.Label(state_frame, text="EEF Forces (gf) - F1:   0.0 | F2:   0.0 | F3:   0.0", style="DataHeader.TLabel", foreground="purple", width=65)
        self.lbl_eef_force.pack(anchor='w', pady=2)

        # --- [Left] 독립형 볼 제어 파라미터 (PD) ---
        pd_frame = ttk.LabelFrame(left_col, text=" Ball Controller (PD) ", padding=5)
        pd_frame.pack(fill='x', pady=5)

        self.lbl_kp_ball = ttk.Label(pd_frame, text=f"Kp_ball: {self.node.Kp_ball:.3f}", width=20)
        self.lbl_kp_ball.pack(anchor='w')
        self.sl_kp_ball = ttk.Scale(pd_frame, from_=0.0, to=0.5, orient='horizontal', command=lambda v: self.update_ball_gain("Kp", v))
        self.sl_kp_ball.set(self.node.Kp_ball)
        self.sl_kp_ball.pack(fill='x', expand=True, pady=(0,5))

        self.lbl_kd_ball = ttk.Label(pd_frame, text=f"Kd_ball: {self.node.Kd_ball:.3f}", width=20)
        self.lbl_kd_ball.pack(anchor='w')
        self.sl_kd_ball = ttk.Scale(pd_frame, from_=0.0, to=0.1, orient='horizontal', command=lambda v: self.update_ball_gain("Kd", v))
        self.sl_kd_ball.set(self.node.Kd_ball)
        self.sl_kd_ball.pack(fill='x', expand=True)

        # --- [Left] 제어 각도 한계 제어 ---
        limit_frame = ttk.LabelFrame(left_col, text=" Control Limits ", padding=5)
        limit_frame.pack(fill='x', pady=5)

        self.lbl_max_tilt = ttk.Label(limit_frame, text=f"Max Tilt Angle: {self.node.max_tilt_deg:.1f} deg", width=30)
        self.lbl_max_tilt.pack(anchor='w')
        self.sl_max_tilt = ttk.Scale(limit_frame, from_=1.0, to=15.0, orient='horizontal', command=self.update_max_tilt)
        self.sl_max_tilt.set(self.node.max_tilt_deg)
        self.sl_max_tilt.pack(fill='x', expand=True)

        # --- [Right] Config Management (Save/Load) ---
        config_frame = ttk.LabelFrame(right_col, text=" Config Management ", padding=5)
        config_frame.pack(fill='x', pady=2)
        
        btn_load = ttk.Button(config_frame, text="Load Config", command=self.action_load_config)
        btn_load.pack(side='left', expand=True, padx=2)
        btn_save = ttk.Button(config_frame, text="Save Config", command=self.action_save_config)
        btn_save.pack(side='right', expand=True, padx=2)

        # --- [Right] 로봇 핸드 및 시스템 하드웨어 파라미터 ---
        gain_frame = ttk.LabelFrame(right_col, text=" Hardware Parameters ", padding=5)
        gain_frame.pack(fill='x', pady=2)

        self.gain_labels = {}
        self.hw_sliders = {}
        self.gains_def_list = [
            ("Kp (K_task - Position)", 5, 200.0), 
            ("Kp (K_task - Force)", 11, 5.0), 
            ("Kd (D_task - Velocity)", 6, 10.0), 
            ("Kp_rot (K_ori - Orientation)", 7, 1.0),
            ("Friction Static", 8, 0.2),         
            ("Friction Bias", 9, 0.2),           
            ("Friction Vel Compensate", 10, 50.0) 
        ]
        for name, idx, max_val in self.gains_def_list:
            frame = ttk.Frame(gain_frame)
            frame.pack(fill='x', pady=1)
            lbl = ttk.Label(frame, text=f"{name}: {self.node.target_data[idx]:>7.3f}", style="Data.TLabel", width=45)
            lbl.pack(side='top', anchor='w')
            self.gain_labels[name] = lbl
            slider = ttk.Scale(frame, from_=0.0, to=max_val, orient='horizontal', command=lambda val, i=idx, n=name: self.update_hw_gain(i, val, n))
            slider.set(self.node.target_data[idx])
            slider.pack(fill='x', expand=True)
            self.hw_sliders[name] = slider

        # --- [Right] 수동 매뉴얼 오프셋 제어기 ---
        offset_frame = ttk.LabelFrame(right_col, text=" Manual Roll/Pitch Offsets ", padding=5)
        offset_frame.pack(fill='x', pady=5)

        self.lbl_roll_offset = ttk.Label(offset_frame, text=f"Roll Offset: {np.degrees(self.node.roll_offset_rad):>7.3f}", style="Data.TLabel", width=25)
        self.lbl_roll_offset.pack(anchor='w')
        self.slider_roll_offset = ttk.Scale(offset_frame, from_=-10.0, to=10.0, orient='horizontal', command=lambda val: self.update_offset_value("roll", val))
        self.slider_roll_offset.set(np.degrees(self.node.roll_offset_rad))
        self.slider_roll_offset.pack(fill='x')

        self.lbl_pitch_offset = ttk.Label(offset_frame, text=f"Pitch Offset: {np.degrees(self.node.pitch_offset_rad):>7.3f}", style="Data.TLabel", width=25)
        self.lbl_pitch_offset.pack(anchor='w')
        self.slider_pitch_offset = ttk.Scale(offset_frame, from_=-10.0, to=10.0, orient='horizontal', command=lambda val: self.update_offset_value("pitch", val))
        self.slider_pitch_offset.set(np.degrees(self.node.pitch_offset_rad))
        self.slider_pitch_offset.pack(fill='x')

        # --- [Right] 힘-Roll/Pitch Bias 상수항(c0) 실시간 제어 모듈 ---
        bias_const_frame = ttk.LabelFrame(right_col, text=" Force Bias Constants (c0, Range: 0~300) ", padding=5)
        bias_const_frame.pack(fill='x', pady=5)

        self.lbl_c0_s1 = ttk.Label(bias_const_frame, text=f"Sensor 1 c0: {self.node.force_coeffs[0][0]:>7.3f}", style="Data.TLabel", width=25)
        self.lbl_c0_s1.pack(anchor='w')
        self.slider_c0_s1 = ttk.Scale(bias_const_frame, from_=0.0, to=300.0, orient='horizontal', command=lambda val: self.update_c0_value(0, val))
        self.slider_c0_s1.set(self.node.force_coeffs[0][0])
        self.slider_c0_s1.pack(fill='x', pady=(0, 5))

        self.lbl_c0_s2 = ttk.Label(bias_const_frame, text=f"Sensor 2 c0: {self.node.force_coeffs[1][0]:>7.3f}", style="Data.TLabel", width=25)
        self.lbl_c0_s2.pack(anchor='w')
        self.slider_c0_s2 = ttk.Scale(bias_const_frame, from_=0.0, to=300.0, orient='horizontal', command=lambda val: self.update_c0_value(1, val))
        self.slider_c0_s2.set(self.node.force_coeffs[1][0])
        self.slider_c0_s2.pack(fill='x', pady=(0, 5))

        self.lbl_c0_s3 = ttk.Label(bias_const_frame, text=f"Sensor 3 c0: {self.node.force_coeffs[2][0]:>7.3f}", style="Data.TLabel", width=25)
        self.lbl_c0_s3.pack(anchor='w')
        self.slider_c0_s3 = ttk.Scale(bias_const_frame, from_=0.0, to=300.0, orient='horizontal', command=lambda val: self.update_c0_value(2, val))
        self.slider_c0_s3.set(self.node.force_coeffs[2][0])
        self.slider_c0_s3.pack(fill='x')

        # --- [Right] 노드 상태 및 기능 버튼 ---
        ctrl_btn_frame = ttk.Frame(right_col)
        ctrl_btn_frame.pack(fill='x', pady=5)
        
        btn_pause = ttk.Button(ctrl_btn_frame, text="Pause (Manual Tilt Only)", command=self.node.pause_control)
        btn_pause.pack(side='left', expand=True, padx=2)
        btn_resume = ttk.Button(ctrl_btn_frame, text="Resume Control (PD ON)", command=self.node.resume_control)
        btn_resume.pack(side='right', expand=True, padx=2)
        
        btn_zero = ttk.Button(right_col, text="Set Force Zero (Tactile Offset)", command=self.node.set_force_zero)
        btn_zero.pack(fill='x', pady=2)

        # [추가] OpenCV 실시간 모니터링 윈도우 생성 초기화
        if self.node.show_gui:
            cv2.namedWindow("Ball & EEF Position Map", cv2.WINDOW_AUTOSIZE)

        self.update_ui_loop()

    def action_load_config(self):
        self.node.load_config()
        self.sl_kp_ball.set(self.node.Kp_ball)
        self.sl_kd_ball.set(self.node.Kd_ball)
        self.sl_max_tilt.set(self.node.max_tilt_deg)
        self.slider_roll_offset.set(np.degrees(self.node.roll_offset_rad))
        self.slider_pitch_offset.set(np.degrees(self.node.pitch_offset_rad))
        
        self.slider_c0_s1.set(self.node.force_coeffs[0][0])
        self.slider_c0_s2.set(self.node.force_coeffs[1][0])
        self.slider_c0_s3.set(self.node.force_coeffs[2][0])
        
        for name, idx, _ in self.gains_def_list:
            self.hw_sliders[name].set(self.node.target_data[idx])
            
        print("[System] Config successfully loaded and synced to UI.")

    def action_save_config(self):
        self.node.save_config()
        print("[System] Config successfully saved.")

    def update_ball_gain(self, name, value):
        val = float(value)
        if name == "Kp":
            self.node.Kp_ball = val
            self.lbl_kp_ball.config(text=f"Kp_ball: {val:.3f}")
        elif name == "Kd":
            self.node.Kd_ball = val
            self.lbl_kd_ball.config(text=f"Kd_ball: {val:.3f}")

    def update_max_tilt(self, value):
        val = float(value)
        self.node.max_tilt_deg = val
        self.node.MAX_TILT_RAD = np.radians(val)
        self.lbl_max_tilt.config(text=f"Max Tilt Angle: {val:.1f} deg")

    def update_hw_gain(self, idx, value, name):
        val = float(value)
        self.node.target_data[idx] = val
        if self.node.shm_connected_pose and self.node.pose_array is not None:
            self.node.pose_array[idx] = val
        self.gain_labels[name].config(text=f"{name}: {val:>7.3f}")

    def update_offset_value(self, name, value):
        val_deg = float(value)
        val_rad = np.radians(val_deg)
        if name == "roll":
            self.node.roll_offset_rad = val_rad
            self.lbl_roll_offset.config(text=f"Roll Offset: {val_deg:>7.3f}")
        elif name == "pitch":
            self.node.pitch_offset_rad = val_rad
            self.lbl_pitch_offset.config(text=f"Pitch Offset: {val_deg:>7.3f}")

    def update_c0_value(self, idx, value):
        val = float(value)
        self.node.force_coeffs[idx][0] = val
        if idx == 0:
            self.lbl_c0_s1.config(text=f"Sensor 1 c0: {val:>7.3f}")
        elif idx == 1:
            self.lbl_c0_s2.config(text=f"Sensor 2 c0: {val:>7.3f}")
        elif idx == 2:
            self.lbl_c0_s3.config(text=f"Sensor 3 c0: {val:>7.3f}")

    def update_ui_loop(self):
        if self.node.shutdown_flag:
            self.root.quit()
            return

        # 1. 기존 Tkinter 텍스트 필드 실시간 업데이트 로직
        if self.node.shm_connected_ball:
            self.lbl_curr_raw.config(text=f"Raw 1: {self.node.curr_f1_raw:>6.1f} | Raw 2: {self.node.curr_f2_raw:>6.1f} | Raw 3: {self.node.curr_f3_raw:>6.1f}")
            self.lbl_curr_res.config(text=f"Res 1: {self.node.curr_f1_res:>6.1f} | Res 2: {self.node.curr_f2_res:>6.1f} | Res 3: {self.node.curr_f3_res:>6.1f}")
            
            if self.node.ball_detected:
                self.lbl_ball_status.config(text="Ball: DETECTED", foreground="blue")
                self.lbl_ball_pos.config(text=f"Pos (X, Y): {self.node.curr_ball_x:>6.3f}, {self.node.curr_ball_y:>6.3f}")
                self.lbl_ball_vel.config(text=f"Vel (Vx, Vy): {self.node.curr_ball_vx:>6.3f}, {self.node.curr_ball_vy:>6.3f}")
            else:
                self.lbl_ball_status.config(text=f"Ball: NOT DETECTED (f_total < {F_TOTAL_MIN}gf)", foreground="red")
                self.lbl_ball_pos.config(text="Pos (X, Y):      - ,      -")
                self.lbl_ball_vel.config(text="Vel (Vx, Vy):     - ,      -")
        
        self.lbl_actual_freq.config(text=f"Actual Freq: {self.node.actual_freq:>5.1f} Hz | Z_vel: {self.node.z_vel_filt:>6.3f} m/s | Gain: {self.node.Kp_ball * (max(0.15, 1.0 - abs(self.node.z_vel_filt)/0.05) if abs(self.node.z_vel_filt) > 0.01 else 1.0):.3f}")

        if self.node.shm_connected_pose and self.node.eef_force_array is not None:
            self.lbl_eef_force.config(text=f"EEF Forces (gf) - F1: {self.node.eef_force_array[0]:>6.1f} | F2: {self.node.eef_force_array[1]:>6.1f} | F3: {self.node.eef_force_array[2]:>6.1f}")

        # 2. [신설] OpenCV 500x500 실시간 위치 시각화 맵 드로잉 모듈
        if self.node.show_gui:
            try:
                canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
                cv2.line(canvas, (250, 0), (250, 500), (220, 220, 220), 1)
                cv2.line(canvas, (0, 250), (500, 250), (220, 220, 220), 1)
                cv2.putText(canvas, "+X (mm)", (430, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(canvas, "+Y (mm)", (255, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

                if self.node.shm_connected_pose and self.node.eef_array is not None:
                    for i in range(3):
                        eef_x_m = self.node.eef_array[3 * i]
                        eef_y_m = self.node.eef_array[3 * i + 1]
                        pixel_ex = int(250 + (eef_x_m * 1000.0))
                        pixel_ey = int(250 - (eef_y_m * 1000.0))
                        pixel_ex = np.clip(pixel_ex, 0, 499)
                        pixel_ey = np.clip(pixel_ey, 0, 499)
                        cv2.rectangle(canvas, (pixel_ex - 5, pixel_ey - 5), (pixel_ex + 5, pixel_ey + 5), (0, 180, 0), -1)
                        cv2.putText(canvas, f"E{i+1}", (pixel_ex + 7, pixel_ey + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 130, 0), 1, cv2.LINE_AA)

                ball_x_m = self.node.curr_ball_x
                ball_y_m = self.node.curr_ball_y
                pixel_bx = int(250 + (ball_x_m * 1000.0))
                pixel_by = int(250 - (ball_y_m * 1000.0))
                pixel_bx = np.clip(pixel_bx, 0, 499)
                pixel_by = np.clip(pixel_by, 0, 499)

                if self.node.ball_detected:
                    cv2.circle(canvas, (pixel_bx, pixel_by), 7, (0, 0, 255), -1)
                    cv2.circle(canvas, (pixel_bx, pixel_by), 8, (0, 0, 130), 1, cv2.LINE_AA)
                    cv2.putText(canvas, "Ball", (pixel_bx + 10, pixel_by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (pixel_bx, pixel_by), 6, (180, 180, 180), -1)
                    cv2.putText(canvas, "Lost", (pixel_bx + 10, pixel_by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

                cv2.imshow("Ball & EEF Position Map", canvas)
                cv2.waitKey(1)
            except KeyboardInterrupt:
                self.node.shutdown_flag = True
                self.root.quit()
                return
            except Exception:
                pass

        # 100ms 주기로 Tkinter UI 및 OpenCV 타일 맵 리프레시 틱 가동
        self.root.after(100, self.update_ui_loop)

    def run(self):
        self.root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = BallBalancingNode()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    ros_thread.start()

    try:
        ui = TuningUI(node)
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        if len(node.csv_data) > 0:
            filename = "ball_balancing_log.csv"
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'roll_rad', 'pitch_rad',
                        'f1_raw', 'f2_raw', 'f3_raw',
                        'f1_res', 'f2_res', 'f3_res',
                        'f1_exp', 'f2_exp', 'f3_exp',
                        'ball_x', 'ball_y', 'ball_vx', 'ball_vy',
                        'z_vel_filt', 'gain_scale',
                        'xyz_des_z', 'K_task', 'D_task',
                        'cop_x_raw', 'cop_y_raw', 'f_total',
                        'cop_x_meas', 'cop_y_meas', 'kf_P_trace',
                        'eef_v1_x', 'eef_v1_y', 'eef_v1_z',
                        'eef_v2_x', 'eef_v2_y', 'eef_v2_z',
                        'eef_v3_x', 'eef_v3_y', 'eef_v3_z',
                        'eef_cur_1', 'eef_cur_2', 'eef_cur_3'
                    ])
                    writer.writerows(node.csv_data)
                print(f"\n[CSV 저장] '{filename}'에 데이터를 성공적으로 저장했습니다.")
            except Exception as e:
                print(f"\n[CSV 저장 실패] 에러 발생: {e}")
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()