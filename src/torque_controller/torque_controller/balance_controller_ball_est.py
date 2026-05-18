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
        
        self.shm_connected_ball = False
        self.shm_connected_pose = False
        self.last_connect_log_time = 0.0

        # 플래그 및 모드
        self.is_paused = False
        self.scenario_active = False
        self.scenario_thread = None
        self.scenario_status_str = "IDLE"

        # 포스 데이터 저장용 변수 신설
        self.curr_f1_raw = 0.0; self.curr_f2_raw = 0.0; self.curr_f3_raw = 0.0
        self.curr_f1_res = 0.0; self.curr_f2_res = 0.0; self.curr_f3_res = 0.0

        # [업데이트 완료] 판 무게 하중 보정 2차 함수 계수
        self.force_coeffs = np.array([
            [202.7785,   4.2069,   2.4746, -0.3023, -0.6640, -1.1094],  # Sensor 1
            [157.1844,   2.1786,  -8.0355, -0.5935, -0.2524,  0.2542],  # Sensor 2
            [124.3782, -12.7876,   0.4890, -0.8615,  0.1039,  0.0125]   # Sensor 3
        ])

        # 호환성을 위한 더미 변수
        self.curr_x = 0.0; self.curr_y = 0.0
        self.curr_vx = 0.0; self.curr_vy = 0.0
        self.target_vx = 0.0; self.target_vy = 0.0

        self.last_obs_time = 0.0 
        
        # 데이터 로깅 관련 변수 신설
        self.csv_data = []       
        self.is_logging_active = False
        self.target_ball_x = 0.0
        self.target_ball_y = 0.0

        self.target_data = np.zeros(12)
        self.f_xyz_manual = np.zeros(3, np.float64)

        self.config_file = 'balance_config.json'
        self.load_config()

        self.zero_pub = self.create_publisher(Empty, 'set_force_zero', 10)
        self.vel_err_sum = np.zeros(2)
        self.MAX_TARGET_VEL = 0.1

        self.get_logger().info("Ball Balancing Node (Force Residue Logger Mode) Starting...")

        self.pause_control()

    def get_expected_force(self, roll_deg, pitch_deg, sensor_idx):
        c = self.force_coeffs[sensor_idx]
        x = roll_deg
        y = pitch_deg
        return c[0] + c[1]*x + c[2]*y + c[3]*(x**2) + c[4]*(y**2) + c[5]*(x*y)

    def load_config(self):
        self.use_prediction = False
        self.Kp_vel = 0.41
        self.Ki_vel = 0.42
        self.MAX_VI = 0.89
        self.max_tilt_deg = 8.0
        self.MAX_TILT_RAD = np.radians(self.max_tilt_deg)
        self.roll_offset_rad = 0.0
        self.pitch_offset_rad = 0.0

        target_defaults = [0.0, 0.0, 0.0, 0.0, 0.0,
        160.0, 0.35, 0.11, 0.006, 0.04, 1.4, 0.0]
        self.target_data[:] = target_defaults

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    cfg = json.load(f)
                    self.max_tilt_deg = cfg.get('max_tilt_deg', self.max_tilt_deg)
                    self.MAX_TILT_RAD = np.radians(self.max_tilt_deg)
                    self.roll_offset_rad = cfg.get('roll_offset_rad', self.roll_offset_rad)
                    self.pitch_offset_rad = cfg.get('pitch_offset_rad', self.pitch_offset_rad)

                    for i in range(5, 12):
                        idx_str = str(i)
                        if idx_str in cfg.get('target_data', {}):
                            self.target_data[i] = cfg['target_data'][idx_str]
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {e}")

    def save_config(self):
        cfg = {
            'max_tilt_deg': self.max_tilt_deg,
            'roll_offset_rad': self.roll_offset_rad,
            'pitch_offset_rad': self.pitch_offset_rad,
            'target_data': {str(i): self.target_data[i] for i in range(5, 12)}
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            pass

    def set_force_zero(self):
        msg = Empty()
        self.zero_pub.publish(msg)

    def reset_integrals(self):
        self.vel_err_sum = np.zeros(2)

    def pause_control(self):
        self.stop_scenario()
        self.is_paused = True

    def resume_control(self):
        self.reset_integrals()
        self.is_paused = False

    def start_position_scenario(self):
        self.get_logger().info("Automated Scenario is disabled to safely log Force Data.")
        pass 

    def stop_scenario(self):
        self.scenario_active = False
        self.scenario_status_str = "IDLE"

    def check_shm_alive(self):
        if self.shm_connected_pose:
            # eef_force_shm 도 살아있는지 함께 체크
            if not os.path.exists('/dev/shm/target_pose_shm') or not os.path.exists('/dev/shm/eef_pos_shm') or not os.path.exists('/dev/shm/eef_force_shm'):
                os._exit(0)

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
            if os.path.exists('/dev/shm/target_pose_shm') and os.path.exists('/dev/shm/eef_pos_shm') and os.path.exists('/dev/shm/eef_force_shm'):
                self.shm_pose = shared_memory.SharedMemory(name='target_pose_shm', create=False)
                self.pose_array = np.ndarray((12,), dtype=np.float64, buffer=self.shm_pose.buf)
                
                self.shm_eef = shared_memory.SharedMemory(name='eef_pos_shm', create=False)
                self.eef_array = np.ndarray((6,), dtype=np.float64, buffer=self.shm_eef.buf)
                
                self.shm_eef_force = shared_memory.SharedMemory(name='eef_force_shm', create=False)
                self.eef_force_array = np.ndarray((3,), dtype=np.float64, buffer=self.shm_eef_force.buf)

                from multiprocessing.resource_tracker import unregister
                unregister(self.shm_pose._name, 'shared_memory')
                unregister(self.shm_eef._name, 'shared_memory')
                unregister(self.shm_eef_force._name, 'shared_memory')
                
                self.pose_array[:] = self.target_data[:]
                self.shm_connected_pose = True

    def control_loop(self):
        self.check_shm_alive()

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
            # 1. SHM에서 f1, f2, f3 Raw Data 취득
            curr_raw = np.array(self.ball_state_array)
            self.curr_f1_raw, self.curr_f2_raw, self.curr_f3_raw, obs_time = curr_raw 

            # 2. 현재 Target 각도를 Degree로 변환
            roll_deg = np.degrees(self.target_data[3])
            pitch_deg = np.degrees(self.target_data[4])

            # 3. 2차 함수 기반 예측 힘 계산 (Expected Force)
            f1_exp = self.get_expected_force(roll_deg, pitch_deg, 0)
            f2_exp = self.get_expected_force(roll_deg, pitch_deg, 1)
            f3_exp = self.get_expected_force(roll_deg, pitch_deg, 2)

            # [수정] 실시간 EEF 힘 데이터(3자유도) 추출
            eef_force_data = list(self.eef_force_array) if (self.shm_connected_pose and self.eef_force_array is not None) else [0.0]*3

            # 4. 잔차(Residue) 도출: 센서 측정값 - 판 무게 예측 - 손가락의 인위적 상향 토크(eef_force)
            # 물리 법칙에 따라 eef_force가 양수(상승 압축)일 때 측정값이 뻥튀기 되므로, 이를 '빼서(-)' 상쇄시킵니다.
            self.curr_f1_res = self.curr_f1_raw - f1_exp - eef_force_data[0]
            self.curr_f2_res = self.curr_f2_raw - f2_exp - eef_force_data[1]
            self.curr_f3_res = self.curr_f3_raw - f3_exp - eef_force_data[2]

            xyz_input = np.array([0.0, 0.0, 0.0])

            if self.is_paused:
                self.target_data[0:3] = self.f_xyz_manual
                self.target_data[3] = 0.0 + self.roll_offset_rad   
                self.target_data[4] = 0.0 + self.pitch_offset_rad  
                self.pose_array[:] = self.target_data[:]
            else:
                cmd_tilt = np.zeros(2) # PID 미적용 모드 (매뉴얼 오프셋 제어)
                self.target_data[0:3] = xyz_input
                self.target_data[3] = -cmd_tilt[1] + self.roll_offset_rad 
                self.target_data[4] = cmd_tilt[0] + self.pitch_offset_rad 
                self.pose_array[:] = self.target_data[:]

            # 5. CSV 데이터 로깅 (활성화 시에만 저장, eef_force 추가)
            if self.is_logging_active:
                self.csv_data.append([
                    now,
                    self.target_ball_x, self.target_ball_y,   # Target 공 위치 기록
                    self.target_data[3], self.target_data[4], # rad
                    roll_deg, pitch_deg,                      # deg
                    self.curr_f1_raw, self.curr_f2_raw, self.curr_f3_raw,
                    self.curr_f1_res, self.curr_f2_res, self.curr_f3_res
                ] + eef_force_data)

        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")

    def destroy_node(self):
        self.stop_scenario()
        if self.shm_ball: self.shm_ball.close()
        if self.shm_pose: self.shm_pose.close()
        if self.shm_eef: self.shm_eef.close()
        if self.shm_eef_force: self.shm_eef_force.close() 
        super().destroy_node()


class TuningUI:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Force Residue Logger (EEF Comp)")
        self.root.geometry("1100x600") 

        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 11, "bold"))
        style.configure("Data.TLabel", font=("Courier", 10))

        left_col = ttk.Frame(self.root, padding=5)
        left_col.pack(side='left', fill='both', expand=True, padx=5)
        right_col = ttk.Frame(self.root, padding=5)
        right_col.pack(side='right', fill='both', expand=True, padx=5)
        
        # --- 1. 상태 표기 (Raw, Residue, EEF Force) ---
        state_frame = ttk.LabelFrame(left_col, text=" Current Force & Residue Monitor ", padding=5)
        state_frame.pack(fill='x', padx=5, pady=2)
        
        self.lbl_curr_raw = ttk.Label(state_frame, text="Raw 1: 0.00 | Raw 2: 0.00 | Raw 3: 0.00", style="Header.TLabel")
        self.lbl_curr_raw.pack(anchor='center', pady=2)
        
        self.lbl_curr_res = ttk.Label(state_frame, text="Res 1: 0.00 | Res 2: 0.00 | Res 3: 0.00", style="Header.TLabel", foreground="blue")
        self.lbl_curr_res.pack(anchor='center', pady=2)

        # 신규 추가: EEF 힘을 보여주는 라벨
        self.lbl_eef_force = ttk.Label(state_frame, text="EEF Forces (gf) - F1: 0.0 | F2: 0.0 | F3: 0.0", style="Header.TLabel", foreground="purple")
        self.lbl_eef_force.pack(anchor='center', pady=2)
        
        ttk.Separator(state_frame, orient='horizontal').pack(fill='x', pady=5)

        self.lbl_actual_freq = ttk.Label(state_frame, text="Actual Freq: 0.0 Hz", style="Header.TLabel", foreground="green")
        self.lbl_actual_freq.pack(anchor='center', pady=2)
        
        self.lbl_sample_count = ttk.Label(state_frame, text="Recorded Samples: 0", style="Header.TLabel", foreground="purple")
        self.lbl_sample_count.pack(anchor='center', pady=2)

        # --- 2. Target Position 5x5 Grid ---
        grid_frame = ttk.LabelFrame(left_col, text=" Target Position Record (Toggle to Log) ", padding=5)
        grid_frame.pack(fill='both', expand=True, padx=5, pady=10)

        self.grid_buttons = {}
        self.active_button_id = None
        self.default_btn_bg = "#e0e0e0"

        # 좌표 설정 (-0.15 ~ 0.15, 0.075 간격)
        coords = [-0.15, -0.075, 0.0, 0.075, 0.15]
        
        # 화면의 Top이 +y가 되도록 뒤집어서 매핑
        for r, y in enumerate(reversed(coords)):
            for c, x in enumerate(coords):
                btn = tk.Button(grid_frame, text=f"X:{x}\nY:{y}", width=6, height=2, bg=self.default_btn_bg,
                                command=lambda x_val=x, y_val=y, r_idx=r, c_idx=c: self.on_grid_click(x_val, y_val, r_idx, c_idx))
                btn.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
                self.grid_buttons[(r, c)] = btn
                
        # 그리드 열/행 비율 맞추기
        for i in range(5):
            grid_frame.grid_columnconfigure(i, weight=1)
            grid_frame.grid_rowconfigure(i, weight=1)

        # --- 3. 오프셋 설정 ---
        offset_frame = ttk.LabelFrame(right_col, text=" Roll / Pitch / Force Offsets ", padding=5)
        offset_frame.pack(fill='x', padx=5, pady=2)

        self.lbl_roll_offset = ttk.Label(offset_frame, text=f"Roll Offset: {np.degrees(self.node.roll_offset_rad):.3f}")
        self.lbl_roll_offset.pack(side='top', anchor='w')
        self.slider_roll_offset = ttk.Scale(offset_frame, from_=-10.0, to=10.0, orient='horizontal', command=lambda val: self.update_offset_value("roll", val))
        self.slider_roll_offset.set(np.degrees(self.node.roll_offset_rad))
        self.slider_roll_offset.pack(fill='x', expand=True, pady=(0, 2))

        self.lbl_pitch_offset = ttk.Label(offset_frame, text=f"Pitch Offset: {np.degrees(self.node.pitch_offset_rad):.3f}")
        self.lbl_pitch_offset.pack(side='top', anchor='w')
        self.slider_pitch_offset = ttk.Scale(offset_frame, from_=-10.0, to=10.0, orient='horizontal', command=lambda val: self.update_offset_value("pitch", val))
        self.slider_pitch_offset.set(np.degrees(self.node.pitch_offset_rad))
        self.slider_pitch_offset.pack(fill='x', expand=True)

        # --- 4. 하단 제어 버튼 ---
        ctrl_btn_frame = ttk.Frame(right_col, padding=5)
        ctrl_btn_frame.pack(fill='x', pady=2)
        
        btn_pause = ttk.Button(ctrl_btn_frame, text="Pause (Roll/Pitch=0)", command=self.node.pause_control)
        btn_pause.pack(side='left', expand=True, padx=5)
        btn_resume = ttk.Button(ctrl_btn_frame, text="Resume Control", command=self.node.resume_control)
        btn_resume.pack(side='right', expand=True, padx=5)
        
        btn_zero = ttk.Button(right_col, text="Set Force Zero (Tactile Offset)", command=self.node.set_force_zero)
        btn_zero.pack(fill='x', padx=5, pady=2)

        self.update_ui_loop()

    def on_grid_click(self, x, y, r, c):
        btn_id = (r, c)
        
        if self.active_button_id == btn_id:
            self.grid_buttons[btn_id].config(bg=self.default_btn_bg)
            self.active_button_id = None
            self.node.is_logging_active = False
            self.node.get_logger().info("Data logging STOPPED.")
        else:
            if self.active_button_id:
                self.grid_buttons[self.active_button_id].config(bg=self.default_btn_bg)
            
            self.grid_buttons[btn_id].config(bg="lightgreen")
            self.active_button_id = btn_id
            
            self.node.target_ball_x = x
            self.node.target_ball_y = y
            self.node.is_logging_active = True
            
            self.node.get_logger().info(f"Data logging STARTED at Target(X:{x}, Y:{y})")

    def update_offset_value(self, name, value):
        val_deg = float(value)
        val_rad = np.radians(val_deg)
        if name == "roll":
            self.node.roll_offset_rad = val_rad
            self.lbl_roll_offset.config(text=f"Roll Offset: {val_deg:.3f}")
        elif name == "pitch":
            self.node.pitch_offset_rad = val_rad
            self.lbl_pitch_offset.config(text=f"Pitch Offset: {val_deg:.3f}")

    def update_ui_loop(self):
        if self.node.shm_connected_ball:
            self.lbl_curr_raw.config(text=f"Raw 1: {self.node.curr_f1_raw:>6.1f} | Raw 2: {self.node.curr_f2_raw:>6.1f} | Raw 3: {self.node.curr_f3_raw:>6.1f}")
            self.lbl_curr_res.config(text=f"Res 1: {self.node.curr_f1_res:>6.1f} | Res 2: {self.node.curr_f2_res:>6.1f} | Res 3: {self.node.curr_f3_res:>6.1f}")
        else:
            self.lbl_curr_raw.config(text="Waiting for Ball State SHM...")
            self.lbl_curr_res.config(text="Waiting for Ball State SHM...")
            
        if self.node.shm_connected_pose and self.node.eef_force_array is not None:
            self.lbl_eef_force.config(text=f"EEF Forces (gf) - F1: {self.node.eef_force_array[0]:>6.1f} | F2: {self.node.eef_force_array[1]:>6.1f} | F3: {self.node.eef_force_array[2]:>6.1f}")

        self.lbl_actual_freq.config(text=f"Actual Freq: {self.node.actual_freq:.1f} Hz")
        self.lbl_sample_count.config(text=f"Recorded Samples: {len(self.node.csv_data)}")
        
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
            filename = "force_residue_log.csv"
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # [업데이트] EEF force 컬럼 3개 추가
                    writer.writerow([
                        'timestamp', 'target_x', 'target_y', 'roll_rad', 'pitch_rad', 'roll_deg', 'pitch_deg', 
                        'f1_raw', 'f2_raw', 'f3_raw', 
                        'f1_res', 'f2_res', 'f3_res',
                        'eef_force_1', 'eef_force_2', 'eef_force_3'
                    ])
                    writer.writerows(node.csv_data)
                print(f"\n[CSV 저장] '{filename}'에 {len(node.csv_data)}개 데이터를 성공적으로 저장했습니다.")
            except Exception as e:
                print(f"\n[CSV 저장 실패] 에러 발생: {e}")
        else:
            print("\n[CSV 저장] 기록된 데이터가 없습니다.")
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()