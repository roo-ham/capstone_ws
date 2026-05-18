#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import numpy as np
from multiprocessing import shared_memory
import threading
import tkinter as tk
from tkinter import ttk
import time

class BallBalancingNode(Node):
    def __init__(self):
        super().__init__('ball_balancing_node')

        # 1. 제어 주기 (100Hz)
        self.dt = 0.01
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        # 2. Shared Memory 변수 초기화
        self.shm_pose = None
        self.pose_array = None
        self.shm_connected = False
        self.last_connect_log_time = 0.0  # 로그 스팸 방지용

        # 3. 제어 상태 플래그
        self.is_paused = False

        # 4. 상태 변수 및 8차원 제어 데이터 배열 
        # [z_des0, z_des1, z_des2, roll_des, pitch_des, K_task, D_task, K_ori]
        self.curr_pos = np.zeros(2)
        self.curr_vel = np.zeros(2)
        self.target_data = np.zeros(8)
        self.target_data[:] = [0, 0, 0, 0, 0, 500.0, 5.0, 1.0]

        # 안전 제한 설정 (하드웨어 보호용 상한선)
        self.MAX_FINGER_Z = 0.50          # Z축 이동 제한 (0 ~ 0.5m)
        self.MAX_FINGER_RPY = np.radians(45.0)  # Roll/Pitch 회전 제한 (라디안)

        self.get_logger().info("8-Param Robot Position & Gain Control Node Starting...")

    def pause_control(self):
        """제어 일시 중지 (공유 메모리 출력을 모두 0으로 만들어 토크 인가 차단)"""
        self.is_paused = True
        self.get_logger().info("Control PAUSED. Targets and Gains set to zero.")

    def resume_control(self):
        """제어 재개"""
        self.is_paused = False
        self.get_logger().info("Control RESUMED.")

    def connect_shared_memory(self):
        """C++ 노드가 생성한 공유 메모리를 대기하고 연결하는 메서드"""
        try:
            # create=False를 명시하여 무조건 C++ 마스터가 생성한 블록을 찾도록 유도
            # if self.shm_ball is None:
            #     self.shm_ball = shared_memory.SharedMemory(name='ball_state_shm', create=False)
            #     self.ball_state_array = np.ndarray((4,), dtype=np.float64, buffer=self.shm_ball.buf)
            
            if self.shm_pose is None:
                self.shm_pose = shared_memory.SharedMemory(name='target_pose_shm', create=False)
                self.pose_array = np.ndarray((8,), dtype=np.float64, buffer=self.shm_pose.buf)

                from multiprocessing.resource_tracker import unregister
                unregister(self.shm_pose._name, 'shared_memory')
                
                # [중요] 최초 연결 시, 현재 UI 슬라이더에 세팅되어 있는 값을 공유 메모리에 즉시 동기화
                self.pose_array[:] = self.target_data[:]

            if not self.shm_connected:
                self.get_logger().info("Successfully attached to C++ Master Shared Memory!")
            self.shm_connected = True

        except FileNotFoundError:
            # 아직 C++ 노드가 공유 메모리를 생성하지 않은 경우
            self.disconnect_shared_memory()
            
            # 터미널 로그 스팸을 막기 위해 3초에 한 번만 대기 메시지 출력
            curr_time = time.time()
            if curr_time - self.last_connect_log_time > 3.0:
                self.get_logger().warn("Waiting for C++ Master to create Shared Memory ('target_pose_shm')...")
                self.last_connect_log_time = curr_time

    def disconnect_shared_memory(self):
        """공유 메모리 연결 해제 및 변수 초기화 (재연결 청소용)"""
        self.shm_connected = False
            
        if self.shm_pose is not None:
            try: self.shm_pose.close()
            except: pass
            self.shm_pose = None
            self.pose_array = None

    def control_loop(self):
        # 연결되지 않은 경우 연결을 계속 시도하며 대기
        if not self.shm_connected:
            self.connect_shared_memory()
            return

        try:
            # 1. 센서/상태 업데이트 (공 상태 모니터링)
            self.curr_pos = 0
            self.curr_vel = 0

            # Pause 상태일 경우 모든 파라미터 및 게인을 0으로 인가하여 안전 확보
            if self.is_paused:
                self.pose_array[:] = 0.0
                return

            # 2. 파라미터별 안전 범위 제한 (Safety Clip)
            cmd_z = np.clip(self.target_data[0:3], 0.0, self.MAX_FINGER_Z)
            cmd_rp = np.clip(self.target_data[3:5], -self.MAX_FINGER_RPY, self.MAX_FINGER_RPY)
            cmd_gains = np.clip(self.target_data[5:8], 0.0, 1000.0)

            # 3. 공유 메모리에 최종 인가
            self.pose_array[0:3] = cmd_z
            self.pose_array[3:5] = cmd_rp
            self.pose_array[5:8] = cmd_gains

        except Exception as e:
            # C++ 노드가 도중에 꺼지거나 메모리가 깨진 경우 자원을 청소하고 대기 상태로 복귀
            self.get_logger().error(f"Shared Memory Connection Lost or Error: {e}")
            self.disconnect_shared_memory()

    def destroy_node(self):
        self.disconnect_shared_memory()
        super().destroy_node()


class TuningUI:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Robot Spring-Damper Controller")
        self.root.geometry("480x920") 

        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 11, "bold"))

        # --- 1. 현재 공의 위치 출력 ---
        state_frame = ttk.LabelFrame(self.root, text=" Current Ball State ", padding=10)
        state_frame.pack(fill='x', padx=10, pady=5)
        
        self.lbl_curr_pos = ttk.Label(state_frame, text="Current X: 0.000  |  Current Y: 0.000", style="Header.TLabel")
        self.lbl_curr_pos.pack(anchor='center')

        # --- 2. 손가락별 Target Z 위치 제어 슬라이더 (범위: 0.0 ~ 0.5m) ---
        z_frame = ttk.LabelFrame(self.root, text=" Target Position Setup (z_des - Meter) ", padding=10)
        z_frame.pack(fill='x', padx=10, pady=5)

        self.z_labels = {}
        self.z_sliders = {}
        axes_z = [("FL1 Z", 0), ("FL2 Z", 1), ("FL3 Z", 2)]

        for name, idx in axes_z:
            frame = ttk.Frame(z_frame, padding=2)
            frame.pack(fill='x', pady=2)

            lbl = ttk.Label(frame, text=f"{name}: {self.node.target_data[idx]:.3f} m")
            lbl.pack(side='top', anchor='w')
            self.z_labels[name] = lbl

            slider = ttk.Scale(
                frame, from_=0.0, to=0.5, orient='horizontal',
                command=lambda val, i=idx, n=name: self.update_z_value(i, val, n)
            )
            slider.set(self.node.target_data[idx])
            slider.pack(fill='x', expand=True)
            self.z_sliders[name] = slider

        # --- 3. Roll, Pitch 목표 각도 제어 슬라이더 (범위: +- 45 deg) ---
        ori_frame = ttk.LabelFrame(self.root, text=" Target Orientation Setup (RP - Degree) ", padding=10)
        ori_frame.pack(fill='x', padx=10, pady=5)

        self.ori_labels = {}
        self.ori_sliders = {}
        axes_rp = [("Roll", 3), ("Pitch", 4)]

        for axis, idx in axes_rp:
            frame = ttk.Frame(ori_frame, padding=2)
            frame.pack(fill='x', pady=2)

            init_deg = np.degrees(self.node.target_data[idx])
            lbl = ttk.Label(frame, text=f"Target {axis}: {init_deg:.1f} deg")
            lbl.pack(side='top', anchor='w')
            self.ori_labels[axis] = lbl

            slider = ttk.Scale(
                frame, from_=-45.0, to=45.0, orient='horizontal',
                command=lambda val, i=idx, ax=axis: self.update_rp_value(i, val, ax)
            )
            slider.set(init_deg)
            slider.pack(fill='x', expand=True)
            self.ori_sliders[axis] = slider

        # --- 4. 가상 스프링-댐퍼 게인(Gain) 튜닝 슬라이더 ---
        gain_frame = ttk.LabelFrame(self.root, text=" Virtual Spring-Damper Gains Setup ", padding=10)
        gain_frame.pack(fill='x', padx=10, pady=5)

        self.gain_labels = {}
        self.gain_sliders = {}
        
        gains_def = [("K_task (Position Spring)", 5, 1000.0), 
                     ("D_task (Velocity Damper)", 6, 10.0), 
                     ("K_ori (Orientation Spring)", 7, 2.0)]

        for name, idx, max_val in gains_def:
            frame = ttk.Frame(gain_frame, padding=2)
            frame.pack(fill='x', pady=2)

            lbl = ttk.Label(frame, text=f"{name}: {self.node.target_data[idx]:.1f}")
            lbl.pack(side='top', anchor='w')
            self.gain_labels[name] = lbl

            slider = ttk.Scale(
                frame, from_=0.0, to=max_val, orient='horizontal',
                command=lambda val, i=idx, n=name: self.update_gain_value(i, val, n)
            )
            slider.set(self.node.target_data[idx])
            slider.pack(fill='x', expand=True)
            self.gain_sliders[name] = slider

        # --- 5. 제어 상태 변경 버튼 (Pause / Resume) ---
        ctrl_btn_frame = ttk.Frame(self.root, padding=10)
        ctrl_btn_frame.pack(fill='x', pady=5)

        btn_pause = ttk.Button(ctrl_btn_frame, text="Pause (Zero Output)", command=self.node.pause_control)
        btn_pause.pack(side='left', expand=True, padx=5)

        btn_resume = ttk.Button(ctrl_btn_frame, text="Resume Control", command=self.node.resume_control)
        btn_resume.pack(side='right', expand=True, padx=5)

        # --- 6. 모든 파라미터 리셋 버튼 (0.0 초기화) ---
        btn_center = ttk.Button(self.root, text="Reset All Desire to 0.0", command=self.reset_all_targets)
        btn_center.pack(pady=5)

        self.update_ui_loop()

    def update_z_value(self, idx, value, name):
        val = float(value)
        self.node.target_data[idx] = val
        if self.node.shm_connected and self.node.pose_array is not None:
            self.node.pose_array[idx] = val
        self.z_labels[name].config(text=f"{name}: {val:.3f} m")

    def update_rp_value(self, idx, value, axis_name):
        val_deg = float(value)
        rad = np.radians(val_deg)
        self.node.target_data[idx] = rad
        if self.node.shm_connected and self.node.pose_array is not None:
            self.node.pose_array[idx] = rad
        self.ori_labels[axis_name].config(text=f"Target {axis_name}: {val_deg:.1f} deg")

    def update_gain_value(self, idx, value, name):
        val = float(value)
        self.node.target_data[idx] = val
        if self.node.shm_connected and self.node.pose_array is not None:
            self.node.pose_array[idx] = val
        self.gain_labels[name].config(text=f"{name}: {val:.1f}")

    def reset_all_targets(self):
        self.node.target_data[:5] = 0.0
        if self.node.shm_connected and self.node.pose_array is not None:
            self.node.pose_array[:5] = 0.0
        
        for name in ["FL1 Z", "FL2 Z", "FL3 Z"]:
            self.z_sliders[name].set(0.0)
            self.z_labels[name].config(text=f"{name}: 0.000 m")
            
        for axis in ["Roll", "Pitch"]:
            self.ori_sliders[axis].set(0.0)
            self.ori_labels[axis].config(text=f"Target {axis}: 0.0 deg")
            
        self.node.get_logger().info("All 8 parameters and gains reset to zero.")

    def update_ui_loop(self):
            
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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()