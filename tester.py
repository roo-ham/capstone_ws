import threading
import time
import tkinter as tk
from tkinter import messagebox
from dynamixel_sdk import *

# --- 설정값 ---
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

# 레지스터 주소 (XM/XH 시리즈)
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

class DxlGuiTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamixel SDK GUI Tester")
        self.root.geometry("400x450")

        # SDK 초기화
        self.port_handler = PortHandler(DEVICENAME)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.is_connected = False
        self.monitoring = False

        self.setup_ui()

    def setup_ui(self):
        # ID 및 연결 설정
        frame_conn = tk.LabelFrame(self.root, text="Connection", padx=10, pady=5)
        frame_conn.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_conn, text="ID:").grid(row=0, column=0)
        self.ent_id = tk.Entry(frame_conn, width=5)
        self.ent_id.insert(0, "1")
        self.ent_id.grid(row=0, column=1)

        self.btn_connect = tk.Button(frame_conn, text="Connect Port", command=self.connect_port)
        self.btn_connect.grid(row=0, column=2, padx=10)

        # 제어 모드 및 토크
        frame_ctrl = tk.LabelFrame(self.root, text="Control", padx=10, pady=5)
        frame_ctrl.pack(fill="x", padx=10, pady=5)

        self.btn_torque_on = tk.Button(frame_ctrl, text="Torque ON", fg="blue", command=lambda: self.set_torque(True))
        self.btn_torque_on.grid(row=0, column=0, padx=5)
        
        self.btn_torque_off = tk.Button(frame_ctrl, text="Torque OFF", fg="red", command=lambda: self.set_torque(False))
        self.btn_torque_off.grid(row=0, column=1, padx=5)

        tk.Label(frame_ctrl, text="Mode:").grid(row=1, column=0, pady=5)
        self.btn_mode_pos = tk.Button(frame_ctrl, text="Position Mode", command=lambda: self.set_mode(3))
        self.btn_mode_pos.grid(row=1, column=1)
        self.btn_mode_curr = tk.Button(frame_ctrl, text="Current Mode", command=lambda: self.set_mode(0))
        self.btn_mode_curr.grid(row=1, column=2)

        # 위치 이동
        frame_move = tk.LabelFrame(self.root, text="Movement", padx=10, pady=5)
        frame_move.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_move, text="Goal Pos (0-4095):").grid(row=0, column=0)
        self.ent_pos = tk.Entry(frame_move, width=10)
        self.ent_pos.insert(0, "2048")
        self.ent_pos.grid(row=0, column=1)

        self.btn_go = tk.Button(frame_move, text="GO", width=10, command=self.move_to_pos)
        self.btn_go.grid(row=0, column=2, padx=5)

        # 모니터링 정보
        frame_mon = tk.LabelFrame(self.root, text="Monitoring", padx=10, pady=5)
        frame_mon.pack(fill="both", expand=True, padx=10, pady=5)

        self.lbl_pos = tk.Label(frame_mon, text="Present Position: -", font=("Arial", 10, "bold"))
        self.lbl_pos.pack(anchor="w")
        self.lbl_vel = tk.Label(frame_mon, text="Present Velocity: -", font=("Arial", 10, "bold"))
        self.lbl_vel.pack(anchor="w")

    # --- 기능 로직 ---
    def connect_port(self):
        if self.port_handler.openPort() and self.port_handler.setBaudRate(BAUDRATE):
            self.is_connected = True
            self.btn_connect.config(text="Connected", state="disabled", bg="lightgreen")
            # 모니터링 쓰레드 시작
            self.monitoring = True
            threading.Thread(target=self.update_status, daemon=True).start()
        else:
            messagebox.showerror("Error", "포트를 열 수 없습니다. 권한이나 연결을 확인하세요.")

    def set_torque(self, enable):
        dxl_id = int(self.ent_id.get())
        val = 1 if enable else 0
        self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, val)

    def set_mode(self, mode):
        dxl_id = int(self.ent_id.get())
        self.set_torque(False) # 모드 변경 시 토크 OFF 필수
        time.sleep(0.1)
        res, err = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_OPERATING_MODE, mode)
        if res == COMM_SUCCESS:
            mode_str = "Position" if mode == 3 else "Current"
            messagebox.showinfo("Success", f"Mode changed to {mode_str}")

    def move_to_pos(self):
        dxl_id = int(self.ent_id.get())
        pos = int(self.ent_pos.get())
        self.packet_handler.write4ByteTxRx(self.port_handler, dxl_id, ADDR_GOAL_POSITION, pos)

    def update_status(self):
        """실시간 데이터 갱신 (별도 쓰레드)"""
        while self.monitoring:
            if self.is_connected:
                dxl_id = int(self.ent_id.get())
                # 위치 읽기
                p_pos, _, _ = self.packet_handler.read4ByteTxRx(self.port_handler, dxl_id, ADDR_PRESENT_POSITION)
                # 속도 읽기
                p_vel, _, _ = self.packet_handler.read4ByteTxRx(self.port_handler, dxl_id, ADDR_PRESENT_VELOCITY)
                # p_vel_max, _, _ = self.packet_handler.read4ByteTxRx(self.port_handler, dxl_id, ADDR_MAX_VELOCITY)
                
                # 속도 부호 처리 (32-bit signed)
                if p_vel is not None and p_vel > 0x7FFFFFFF:
                    p_vel -= 4294967296

                # UI 업데이트
                self.lbl_pos.config(text=f"Present Position: {p_pos}")
                self.lbl_vel.config(text=f"Present Velocity: {p_vel} (raw)")
                # self.lbl_vel.config(text=f"Present Velocity: {p_vel_max} (raw)")
            
            time.sleep(0.1) # 10Hz

if __name__ == "__main__":
    root = tk.Tk()
    app = DxlGuiTester(root)
    root.mainloop()