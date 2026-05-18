#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

MAX_AMP = 0.1
# MAX_AMP = 1.614

class SineCurrentController(Node):
    def __init__(self):
        super().__init__('joint_current_tester')
        
        # Publisher & Subscriber
        self.torque_pub = self.create_publisher(Float64MultiArray, 'hand_joint_torque', 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
        # 제어 파라미터 
        self.target_joint_idx = 0
        self.amplitude = 0.0
        self.frequency = 0.0
        self.is_on = False
        
        # 모니터링 데이터 버퍼
        self.current_velocity = np.zeros(12)
        
        # 상태 기록용 (시간 제한 없이 누적하기 위해 일반 list 사용)
        self.history_vel = []
        self.history_cur = []
        
        self.start_time = time.perf_counter()
        
        # 제어 루프 (1000Hz)
        self.timer = self.create_timer(0.001, self.control_loop)
        self.get_logger().info("Joint Current Tester Node Started!")

    def joint_state_callback(self, msg):
        # 12개의 joint 데이터를 받아온다고 가정
        if len(msg.velocity) >= 12:
            self.current_velocity = np.array(msg.velocity[:12])

    def control_loop(self):
        # 12개의 Joint 전류 배열 초기화 (기본 0)
        torque_cmd = np.zeros(12)
        applied_current = 0.0
        
        if self.is_on:
            # 선택된 Joint에 사인파 전류 계산 (기존 로직 유지, amplitude 값을 고정으로 넣고 있음)
            # ※ 원래 코드에서 사인파 주파수(frequency) 적용 로직이 생략되어 있었습니다. 
            # 진폭(amplitude)만 그대로 유지했습니다.
            applied_current = self.amplitude * np.sin(2 * 3.14 * self.frequency * time.time())
            torque_cmd[self.target_joint_idx] = applied_current
            
        # 퍼블리시
        msg = Float64MultiArray()
        msg.data = torque_cmd.tolist()
        self.torque_pub.publish(msg)
        
        # O(1)의 속도로 빠르게 append 하므로 제어 스레드 성능에 영향 없음
        self.history_vel.append(self.current_velocity[self.target_joint_idx])
        self.history_cur.append(applied_current)

class App:
    def __init__(self, root, ros_node):
        self.root = root
        self.node = ros_node
        self.root.title("Joint Current Tester")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # --- UI 구성 ---
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 1. Joint 선택 Combobox
        ttk.Label(control_frame, text="Select Joint:").grid(row=0, column=0, sticky=tk.W)
        self.joint_var = tk.StringVar()
        self.joint_cb = ttk.Combobox(control_frame, textvariable=self.joint_var, values=[f"Joint {i}" for i in range(12)], state="readonly", width=10)
        self.joint_cb.current(0)
        self.joint_cb.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.joint_cb.bind("<<ComboboxSelected>>", self.on_joint_change)
        
        # 2. Amplitude Slider
        ttk.Label(control_frame, text=f"Amplitude (Max {MAX_AMP}):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.amp_var = tk.DoubleVar(value=0.0)
        self.amp_slider = tk.Scale(control_frame, variable=self.amp_var, from_=-MAX_AMP, to=MAX_AMP, resolution=0.001, orient=tk.HORIZONTAL, length=200, command=self.update_params)
        self.amp_slider.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # 3. Frequency Slider
        ttk.Label(control_frame, text="Frequency (Max 100Hz):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.freq_var = tk.DoubleVar(value=1.0)
        self.freq_slider = tk.Scale(control_frame, variable=self.freq_var, from_=0.0, to=0.1, resolution=0.001, orient=tk.HORIZONTAL, length=200, command=self.update_params)
        self.freq_slider.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # 4. ON/OFF Button
        self.is_on = False
        self.toggle_btn = tk.Button(control_frame, text="Current OFF", bg="red", fg="white", width=15, command=self.toggle_current)
        self.toggle_btn.grid(row=0, column=2, rowspan=3, padx=20)
        
        # --- Matplotlib 구성 ---
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # 성능 최적화를 위해 Line 객체를 미리 생성 (점도표 형식, marker='o', 선 없음)
        # alpha=0.5를 주어 점이 겹칠 때 밀도 파악이 쉽도록 했습니다.
        self.scatter_plot, = self.ax.plot([], [], marker='o', linestyle='None', color='blue', markersize=3, alpha=0.5)
        
        self.ax.set_title(f"Joint 0: Current vs Velocity")
        self.ax.set_xlabel("Current (A)")
        self.ax.set_ylabel("Velocity (rad/s)")
        
        # x축은 MAX_AMP에 맞춰 고정, y축은 데이터에 따라 자동 조절
        self.ax.set_xlim(-MAX_AMP * 1.2, MAX_AMP * 1.2)
        
        # GUI 업데이트 타이머 시작 (50ms 마다 갱신 -> 20fps)
        self.root.after(50, self.update_plot)

    def on_joint_change(self, event):
        idx_str = self.joint_var.get()
        self.node.target_joint_idx = int(idx_str.split(" ")[1])
        # Joint 변경 시 히스토리 초기화 및 타이틀 업데이트
        self.node.history_vel.clear()
        self.node.history_cur.clear()
        self.ax.set_title(f"Joint {self.node.target_joint_idx}: Current vs Velocity")
        
    def update_params(self, event=None):
        self.node.amplitude = self.amp_var.get()
        self.node.frequency = self.freq_var.get()
        
    def toggle_current(self):
        self.is_on = not self.is_on
        self.node.is_on = self.is_on
        if self.is_on:
            self.toggle_btn.config(text="Current ON", bg="green")
        else:
            self.toggle_btn.config(text="Current OFF", bg="red")

    def update_plot(self):
        # 얕은 복사를 통해 제어 스레드에서 append 중일 때 생기는 충돌을 최소화
        c_data = list(self.node.history_cur)
        v_data = list(self.node.history_vel)
        
        if len(c_data) > 0:
            # ax.clear()를 쓰지 않고 데이터만 갱신하여 렌더링 부하 최소화
            self.scatter_plot.set_data(c_data, v_data)
            
            # y축 자동 스케일링 (속도 범위에 맞춰 y축을 조절)
            self.ax.relim()
            self.ax.autoscale_view(scalex=False, scaley=True) 
            
            # draw() 대신 draw_idle()을 사용하여 메인 스레드 블로킹 방지
            self.canvas.draw_idle()
            
        self.root.after(50, self.update_plot)

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

def main(args=None):
    rclpy.init(args=args)
    ros_node = SineCurrentController()
    
    # ROS 루프를 백그라운드 스레드에서 실행
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()
    
    # Tkinter 메인 루프 실행
    root = tk.Tk()
    app = App(root, ros_node)
    root.mainloop()
    
    # 종료 처리
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()