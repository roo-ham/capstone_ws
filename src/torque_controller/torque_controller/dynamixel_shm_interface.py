#!/usr/bin/python3
import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from multiprocessing import shared_memory
from dynamixel_sdk import *

# --- Control Table (XH/XM Series) ---
ADDR_RETURN_DELAY_TIME = 9
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

# Data Length
LEN_TORQUE_ENABLE = 1
LEN_GOAL_CURRENT = 2
LEN_GOAL_POSITION = 4
LEN_POS_VEL_READ = 8
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_POSITION = 4

# Values
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OP_MODE_CURRENT = 0
OP_MODE_POSITION = 3

# Communication
PROTOCOL_VERSION = 2.0
BAUDRATE = 4000000
DEVICENAME = "/dev/ttyUSB0"

# Physics Constants
KT_CONSTANT = 1.0
CURRENT_UNIT_A = 0.00415123456
VELOCITY_UNIT_RPM = 0.229
RPM_TO_RAD_SEC = 0.104719755
DEG_TO_DXL = 1.0 / 0.088


class DynamixelSHMInterface:
    def __init__(self, enable_base_pos_ctrl: bool = True):
        self.node = rclpy.create_node('dynamixel_shm_interface')
        self.pub_joint_state = self.node.create_publisher(JointState, 'joint_states', 10)

        self.enable_base_pos_ctrl = enable_base_pos_ctrl
        self.base_ids = [1, 5, 9] if self.enable_base_pos_ctrl else []
        self.target_ids = list(range(1, 13))
        
        self.offsets = [
            337.5 if enable_base_pos_ctrl else 180, 180.0, 180.0, 180.0,
            112.5, 180.0, 180.0, 180.0,
            247.5, 180.0, 180.0, 180.0
        ]
        self.joint_names = [
            "F11", "F12", "F13", "F14",
            "F21", "F22", "F23", "F24",
            "F31", "F32", "F33", "F34"
        ]
        self.torque_enabled = False

        self.shm_state = None
        self.shm_cmd = None

        # 생성자 도중 에러가 나더라도 메모리가 누수되지 않도록 try-except로 감쌉니다.
        try:
            self._init_shared_memory()

            self.portHandler = PortHandler(DEVICENAME)
            self.packetHandler = PacketHandler(PROTOCOL_VERSION)
            self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_VELOCITY, LEN_POS_VEL_READ)
            self.groupSyncWriteCurrent = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)
            self.groupSyncWritePosition = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
            self.groupSyncWriteTorque = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_TORQUE_ENABLE, LEN_TORQUE_ENABLE)

            if not self._init_dynamixel():
                raise RuntimeError("Failed to initialize Dynamixel. Please check the Port and Power.")
                
            self.node.get_logger().info(f"Dynamixel SHM Interface Started. Base Pos Ctrl Enabled: {self.enable_base_pos_ctrl}")
            
        except Exception as e:
            self.shutdown() # 예외 발생 시 메모리를 안전하게 정리하고 종료
            raise e

    def _init_shared_memory(self):
        state_bytes = 2 * 12 * 8
        try:
            self.shm_state = shared_memory.SharedMemory(name='dxl_state_shm', create=True, size=state_bytes)
        except FileExistsError:
            self.shm_state = shared_memory.SharedMemory(name='dxl_state_shm', create=False)
        self.state_array = np.ndarray((2, 12), dtype=np.float64, buffer=self.shm_state.buf)
        self.state_array.fill(0.0)

        cmd_bytes = 12 * 8
        try:
            self.shm_cmd = shared_memory.SharedMemory(name='dxl_cmd_shm', create=True, size=cmd_bytes)
        except FileExistsError:
            self.shm_cmd = shared_memory.SharedMemory(name='dxl_cmd_shm', create=False)
        self.cmd_array = np.ndarray((12,), dtype=np.float64, buffer=self.shm_cmd.buf)
        self.cmd_array.fill(0.0)

    def _init_dynamixel(self):
        if not self.portHandler.openPort(): return False
        if not self.portHandler.setBaudRate(BAUDRATE): return False

        for id in self.target_ids:
            # [수정된 부분] python ping()은 3개의 값을 리턴합니다 (model_num, result, error)
            dxl_model_num, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, id)
            if dxl_comm_result != COMM_SUCCESS: 
                continue

            self.groupSyncRead.addParam(id)
            self.packetHandler.write1ByteTxRx(self.portHandler, id, ADDR_RETURN_DELAY_TIME, 0)
            self.packetHandler.write1ByteTxRx(self.portHandler, id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

            mode = OP_MODE_POSITION if id in self.base_ids else OP_MODE_CURRENT
            self.packetHandler.write1ByteTxRx(self.portHandler, id, ADDR_OPERATING_MODE, mode)

        return True

    def set_torque_all(self, enable: bool):
        self.groupSyncWriteTorque.clearParam()
        data = [TORQUE_ENABLE] if enable else [TORQUE_DISABLE]

        for id in self.target_ids:
            self.groupSyncWriteTorque.addParam(id, data)
        
        if self.groupSyncWriteTorque.txPacket() == COMM_SUCCESS:
            self.torque_enabled = enable

        if enable and self.enable_base_pos_ctrl:
            self.groupSyncWritePosition.clearParam()
            targets = {1: -60.0, 5: 60.0, 9: 0.0}
            for id, target_deg in targets.items():
                pos_raw = int((target_deg + self.offsets[id-1]) * DEG_TO_DXL)
                pos_raw_unsigned = pos_raw & 0xFFFFFFFF
                param = [
                    pos_raw_unsigned & 0xFF,
                    (pos_raw_unsigned >> 8) & 0xFF,
                    (pos_raw_unsigned >> 16) & 0xFF,
                    (pos_raw_unsigned >> 24) & 0xFF
                ]
                self.groupSyncWritePosition.addParam(id, param)
            self.groupSyncWritePosition.txPacket()

    def run_loop(self):
        self.set_torque_all(True)
        
        try:
            while rclpy.ok():
                if self.groupSyncRead.txRxPacket() == COMM_SUCCESS:
                    msg = JointState()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    
                    for i, id in enumerate(self.target_ids):
                        if self.groupSyncRead.isAvailable(id, ADDR_PRESENT_VELOCITY, LEN_POS_VEL_READ):
                            vel_raw = self.groupSyncRead.getData(id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                            pos_raw = self.groupSyncRead.getData(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)

                            vel_raw = np.int32(np.uint32(vel_raw))
                            pos_raw = np.int32(np.uint32(pos_raw))

                            vel_rad_s = float(vel_raw * VELOCITY_UNIT_RPM * RPM_TO_RAD_SEC)
                            final_deg = float(pos_raw * 0.088) - self.offsets[id - 1]
                            pos_rad = float(final_deg * (math.pi / 180.0))

                            self.state_array[0, i] = pos_rad
                            self.state_array[1, i] = vel_rad_s

                            msg.name.append(self.joint_names[i])
                            msg.position.append(pos_rad)
                            msg.velocity.append(vel_rad_s)

                    self.pub_joint_state.publish(msg)

                self.groupSyncWriteCurrent.clearParam()
                
                for i, id in enumerate(self.target_ids):
                    if id in self.base_ids: continue

                    torque_nm = self.cmd_array[i]
                    current_a = torque_nm / KT_CONSTANT
                    goal_current_val = int(current_a / CURRENT_UNIT_A)
                    goal_current_val = max(-600, min(600, goal_current_val))

                    val_unsigned = goal_current_val & 0xFFFF
                    param = [val_unsigned & 0xFF, (val_unsigned >> 8) & 0xFF]
                    self.groupSyncWriteCurrent.addParam(id, param)

                self.groupSyncWriteCurrent.txPacket()

                rclpy.spin_once(self.node, timeout_sec=0)

        except KeyboardInterrupt:
            self.node.get_logger().info("Keyboard Interrupt detected. Stopping interface...")
        finally:
            self.shutdown()

    def shutdown(self):
        """프로그램 종료 또는 에러 발생 시 리소스(포트, 메모리)를 안전하게 해제합니다."""
        try:
            self.set_torque_all(False)
            self.node.get_logger().info("self.set_torque_all(False)")
        except: pass

        try:
            self.portHandler.closePort()
        except: pass
        
        # [수정된 부분] 안전한 메모리 해제
        try:
            if self.shm_state is not None:
                self.shm_state.close()
                self.shm_state.unlink()
        except: pass
        
        try:
            if self.shm_cmd is not None:
                self.shm_cmd.close()
                self.shm_cmd.unlink()
        except: pass
        
        try:
            self.node.destroy_node()
        except: pass


def main(args=None):
    rclpy.init(args=args)
    dxl_interface = None
    try:
        dxl_interface = DynamixelSHMInterface(enable_base_pos_ctrl=True)
        dxl_interface.run_loop()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if dxl_interface is not None:
            dxl_interface.shutdown()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()