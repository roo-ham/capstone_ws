#!/usr/bin/python3

import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import rclpy
from rclpy.node import Node
from dynamixel_sdk import *

# --- Dynamixel Constants & Addresses ---
ADDR_OPERATING_MODE         = 11
ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_CURRENT        = 126 
ADDR_RETURN_DELAY_TIME      = 9

LEN_GOAL_POSITION           = 4
LEN_PRESENT_CURRENT         = 2
PROTOCOL_VERSION            = 2.0
BAUDRATE                    = 4000000
DEVICENAME                  = '/dev/ttyUSB0'

TORQUE_ENABLE               = 1
TORQUE_DISABLE              = 0
OP_MODE_POSITION            = 3

# --- [수정] C++ 코드 기반 오프셋 설정 ---
# ID 1~12 순서대로 매핑
CPP_OFFSETS = {
    1: 337.5, 2: 180.0, 3: 180.0, 4: 180.0,
    5: 112.5, 6: 180.0, 7: 180.0, 8: 180.0,
    9: 247.5, 10: 180.0, 11: 180.0, 12: 180.0
}

def deg_to_raw(deg, dxl_id):
    """C++ 오프셋을 적용하여 각도를 Dynamixel Raw Value(0~4095)로 변환"""
    # UI상 0도 = 모터의 Offset 위치
    # XH430 모델 기준 1 unit = 0.088 deg (360/4096)
    target_deg = deg + CPP_OFFSETS[dxl_id]
    # 0~360 범위를 0~4095로 변환 (음수 각도 고려)
    raw_val = int(target_deg / 0.08789)
    return max(0, min(4095, raw_val))

class DXLController(Node):
    def __init__(self):
        super().__init__('dxl_ik_controller_offset')
        self.target_ids = list(range(1, 13))
        self.ik_ids = [3, 4, 7, 8, 11, 12] # IK 제어 대상
        self.fixed_ids = [1, 2, 5, 6, 9, 10] # 고정 오프셋 대상
        
        self.fixed_cmd_deg = np.array([-60, 60, 60, 60, 0, 60], np.float64) # 6차원 offset array (degrees)
        self.la, self.lb = 0.06, 0.07
        self.ee_pos = np.zeros((3, 2)) # [k][x, y]
        
        self.current_data = {idx: deque([0]*1000, maxlen=1000) for idx in self.fixed_ids}
        self.lock = threading.Lock()
        self.running = True

        # SDK & Port 초기화
        self.port_handler = PortHandler(DEVICENAME)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        self.group_sync_read_curr = GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)

        if not self.port_handler.openPort() or not self.port_handler.setBaudRate(BAUDRATE):
            self.get_logger().error("Port Open/Baudrate Failed!")
            exit()

        for dxl_id in self.target_ids:
            # 통신 최적화: Return Delay Time 0으로 설정 (C++ 코드 반영)
            self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_RETURN_DELAY_TIME, 0)
            # 모드 설정 및 토크 온
            self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_OPERATING_MODE, OP_MODE_POSITION)
            self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_id in self.fixed_ids:
                self.group_sync_read_curr.addParam(dxl_id)
        
        self.init_ik_positions()

    def init_ik_positions(self):
        t1, t2 = np.radians(15), np.radians(45)
        x = self.la * np.cos(t1) + self.lb * np.cos(t1 + t2)
        y = self.la * np.sin(t1) + self.lb * np.sin(t1 + t2)
        for k in range(3): self.ee_pos[k] = [x, y]

    def solve_ik(self, x, y, la, lb):
        dist_sq = x**2 + y**2
        dist = np.sqrt(dist_sq)
        if dist >= (la + lb):
            q1 = np.arctan2(y, x)
            return np.degrees(q1), 0
        if dist > (la + lb) or dist < abs(la - lb): return None
        cos_q2 = (dist_sq - la**2 - lb**2) / (2 * la * lb)
        q2 = np.arccos(np.clip(cos_q2, -1.0, 1.0))
        q1 = np.arctan2(y, x) - np.arctan2(lb * np.sin(q2), la + lb * np.cos(q2))
        return np.degrees(q1), np.degrees(q2)

    def control_loop(self):
        """400Hz 제어 루프"""
        while rclpy.ok() and self.running:
            with self.lock:
                la, lb, ee_pos, fixed_deg = self.la, self.lb, np.copy(self.ee_pos), np.copy(self.fixed_cmd_deg)

            self.group_sync_write.clearParam()
            for k in range(3):
                base = 4 * k
                # 1. Fixed Joints (4k+1, 4k+2) + UI Offset
                for i, dxl_id in enumerate([base+1, base+2]):
                    raw = deg_to_raw(fixed_deg[k*2+i], dxl_id)
                    self.group_sync_write.addParam(dxl_id, [DXL_LOBYTE(DXL_LOWORD(raw)), DXL_HIBYTE(DXL_LOWORD(raw)), 
                                                            DXL_LOBYTE(DXL_HIWORD(raw)), DXL_HIBYTE(DXL_HIWORD(raw))])
                # 2. IK Joints (4k+3, 4k+4)
                res = self.solve_ik(ee_pos[k][0], ee_pos[k][1], la, lb)
                if res:
                    q1, q2 = res
                    # A-B 링크 연결 구조: B의 각도는 A에 종속됨 (q1 + q2)
                    for i, (dxl_id, angle) in enumerate(zip([base+3, base+4], [q1, q2])):
                        raw = deg_to_raw(angle, dxl_id)
                        self.group_sync_write.addParam(dxl_id, [DXL_LOBYTE(DXL_LOWORD(raw)), DXL_HIBYTE(DXL_LOWORD(raw)), 
                                                                DXL_LOBYTE(DXL_HIWORD(raw)), DXL_HIBYTE(DXL_HIWORD(raw))])
            self.group_sync_write.txPacket()

            # 전류 데이터 읽기
            if self.group_sync_read_curr.txRxPacket() == COMM_SUCCESS:
                for dxl_id in self.fixed_ids:
                    val = self.group_sync_read_curr.getData(dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
                    if val > 32767: val -= 65536
                    self.current_data[dxl_id].append(val)

    def stop(self):
        self.running = False
        self.get_logger().info("Disabling Torque...")
        for dxl_id in self.target_ids:
            self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.port_handler.closePort()

# --- UI with Scrollbar & PanedWindow ---
class ControlUI:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("DXL IK Control with C++ Offsets")
        self.root.geometry("1100x850")

        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill="both", expand=True)

        # Left (Controls)
        self.l_container = ttk.Frame(self.paned)
        self.paned.add(self.l_container, weight=1)
        self.canvas_s = tk.Canvas(self.l_container, width=450)
        self.sb = ttk.Scrollbar(self.l_container, orient="vertical", command=self.canvas_s.yview)
        self.s_frame = ttk.Frame(self.canvas_s)
        self.s_frame.bind("<Configure>", lambda e: self.canvas_s.configure(scrollregion=self.canvas_s.bbox("all")))
        self.canvas_s.create_window((0,0), window=self.s_frame, anchor="nw")
        self.canvas_s.configure(yscrollcommand=self.sb.set)
        self.canvas_s.pack(side="left", fill="both", expand=True)
        self.sb.pack(side="right", fill="y")

        # Right (Graphs)
        self.r_container = ttk.Frame(self.paned)
        self.paned.add(self.r_container, weight=2)

        self.setup_widgets()
        self.setup_graph()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_loop()

    def setup_widgets(self):
        # 1. Link Length
        f1 = ttk.LabelFrame(self.s_frame, text="Link Length (0.01~0.10m)")
        f1.pack(fill="x", padx=10, pady=5)
        self.la_v = tk.DoubleVar(value=self.node.la); self.lb_v = tk.DoubleVar(value=self.node.lb)
        for t, v in [("Link A", self.la_v), ("Link B", self.lb_v)]:
            r = ttk.Frame(f1); r.pack(fill="x")
            ttk.Label(r, text=t, width=10).pack(side="left")
            tk.Scale(r, from_=0.01, to=0.1, resolution=0.001, orient="horizontal", variable=v, command=self.sync).pack(side="right", expand=True, fill="x")

        # 2. Fixed Offsets (C++ 기반 0점 기준 추가 조절)
        f2 = ttk.LabelFrame(self.s_frame, text="Joint Offset (6-dim array, deg)")
        f2.pack(fill="x", padx=10, pady=5)
        self.off_v = []
        ids = [1, 2, 5, 6, 9, 10]
        for i in range(6):
            v = tk.DoubleVar(value=self.node.fixed_cmd_deg[i])
            r = ttk.Frame(f2); r.pack(fill="x")
            ttk.Label(r, text=f"ID {ids[i]}", width=10).pack(side="left")
            tk.Scale(r, from_=-90, to=90, orient="horizontal", variable=v, command=self.sync).pack(side="right", expand=True, fill="x")
            self.off_v.append(v)

        # 3. EEF Control
        self.ee_v = []
        for k in range(3):
            f3 = ttk.LabelFrame(self.s_frame, text=f"Finger {k+1} Workspace (ID {4*k+3},{4*k+4})")
            f3.pack(fill="x", padx=10, pady=5)
            xv = tk.DoubleVar(value=self.node.ee_pos[k][0]); yv = tk.DoubleVar(value=self.node.ee_pos[k][1])
            for t, v in [("X", xv), ("Y", yv)]:
                r = ttk.Frame(f3); r.pack(fill="x")
                ttk.Label(r, text=t, width=5).pack(side="left")
                tk.Scale(r, from_=-0.15, to=0.15, resolution=0.001, orient="horizontal", variable=v, command=self.sync).pack(side="right", expand=True, fill="x")
            self.ee_v.append((xv, yv))

    def setup_graph(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Moving Joints Present Current (mA)")
        self.ax.set_ylim(-50, 50)
        self.lines = {idx: self.ax.plot([], [], label=f"ID {idx}")[0] for idx in [2, 6, 10]}
        self.ax.legend(loc="upper right", ncol=3, fontsize='8')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.r_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def sync(self, _=None):
        with self.node.lock:
            self.node.la, self.node.lb = self.la_v.get(), self.lb_v.get()
            for i in range(6): self.node.fixed_cmd_deg[i] = self.off_v[i].get()
            for k in range(3):
                self.node.ee_pos[k][0], self.node.ee_pos[k][1] = self.ee_v[k][0].get(), self.ee_v[k][1].get()

    def update_loop(self):
        for idx in [2, 6, 10]:
            d = list(self.node.current_data[idx])
            self.lines[idx].set_data(range(len(d)), d)
        self.ax.set_xlim(0, 1000)
        self.canvas.draw()
        self.root.after(33, self.update_loop)

    def on_closing(self):
        self.node.stop()
        self.root.destroy()

def main():
    rclpy.init()
    node = DXLController()
    thr = threading.Thread(target=node.control_loop, daemon=True)
    thr.start()
    ui = ControlUI(node)
    try:
        ui.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        rclpy.shutdown()

if __name__ == "__main__":
    main()