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
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

class InteractiveSpringDamperNode(Node):
    def __init__(self):
        super().__init__('interactive_spring_node')

        # 1. Pinocchio 모델 로드 (관성 데이터 포함) [cite: 1, 5, 8, 10, 13]
        pkg_name = 'torque_controller'
        urdf_filename = 'hand_0926.urdf'
        urdf_path = os.path.join(get_package_share_directory(pkg_name), 'urdf', urdf_filename)
        self.model = pin.buildModelFromUrdf(urdf_path)
        
        # 중력 가속도 설정 (Z축 아래 방향)
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        self.tip_ids = [self.model.getFrameId(f) for f in ['FL1EEF', 'FL2EEF', 'FL3EEF']]

        # 2. 제어 변수 및 중력 보상 설정
        self.K = 1.
        self.D = 2.0
        self.D_joint = 0.0
        self.D_joint_weight = np.array([2, 1.5, 1, 1] * 3, np.float64)
        
        # 중력 보상 강도 (0.0 ~ 1.0)
        self.gravity_comp_gain = 0.3
        
        self.K_ori_R = 0.0
        self.K_ori_P = 0.0
        self.K_ori_Y = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.F_FRIC_STATIC = 0.001
        self.F_FRIC_BIAS = 0.005
        
        self.q = np.zeros(self.model.nq)
        self.v = np.zeros(self.model.nv)

        # 초기 위치 계산
        zero_q = np.zeros(self.model.nq)
        pin.framesForwardKinematics(self.model, self.data, zero_q)
        self.TRI_RADIUS = 0.10 # Radius from Zero Point to Finger
        self.zero_target_bias = [np.array([0, 0, 0.3], np.float64)] * 3
        self.zero_target_tri = [np.array([-1 * np.sqrt(3/4), 1 * 0.5, 0], np.float64),
                           np.array([1 * np.sqrt(3/4), 1 * 0.5, 0], np.float64),
                           np.array([0, -1, 0], np.float64)]
        self.initial_pos = [self.data.oMf[tid].translation.copy() for tid in self.tip_ids]
        # self.zero_q_rot = [self.data.oMf[tid].rotation.copy() for tid in self.tip_ids]
        self.curr_pos = [p.copy() for p in self.initial_pos]
        self.target_pos = [p.copy() for p in self.initial_pos]
        # self.get_logger().info(str(self.target_pos))

        # 로깅 변수
        self.data_logs = []
        self.start_time = self.get_clock().now()
        self.is_saving = False

        # ROS 및 서버 설정
        self.server = InteractiveMarkerServer(self, "finger_mocap")
        for i in range(3):
            self.make_interactive_marker(i, self.target_pos[i])
        self.server.applyChanges()

        self.torque_pub = self.create_publisher(Float64MultiArray, 'hand_joint_torque', 10)
        self.marker_pub = self.create_publisher(Marker, 'finger_triangle_marker', 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)

        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

    def make_interactive_marker(self, idx, position):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = f"finger_{idx}"
        int_marker.pose.position.x = position[0]
        int_marker.pose.position.y = position[1]
        int_marker.pose.position.z = position[2]
        int_marker.scale = 0.05 
        visual_marker = Marker()
        visual_marker.type = Marker.SPHERE
        visual_marker.scale.x = 0.015; visual_marker.scale.y = 0.015; visual_marker.scale.z = 0.015
        visual_marker.color.r = 1.0; visual_marker.color.a = 0.8
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(visual_marker)
        int_marker.controls.append(control)
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
            # self.target_pos[idx] = np.array([feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z])

    def joint_callback(self, msg):
        self.q = np.array(msg.position)
        self.v = np.array(msg.velocity)
        pin.framesForwardKinematics(self.model, self.data, self.q)
        for i, tid in enumerate(self.tip_ids):
            self.curr_pos[i] = self.data.oMf[tid].translation.copy()

    def control_loop(self):
        rate = self.create_rate(1000)
        while rclpy.ok():
            # A. 중력 보상 토크 계산 (URDF 관성 데이터 기반)
            # pin.computeGeneralizedGravity는 g(q)를 반환함
            tau_gravity = pin.computeGeneralizedGravity(self.model, self.data, self.q)
            
            tau_task = np.zeros(self.model.nv)
            tau_task_damper = np.zeros(self.model.nv)
            R_added = R.from_euler('xyz', [self.target_roll, self.target_pitch, self.target_yaw], degrees=False)
            # self.get_logger().info(f"{R_added.as_matrix()} {self.target_pos}")
            
            current_log = {'timestamp': (self.get_clock().now() - self.start_time).nanoseconds / 1e9}

            for i, tid in enumerate(self.tip_ids):
                # 1. 방향 제어
                # R_init = R.from_matrix(self.zero_q_rot[i])
                # R_target = R_added * R_init

                # 2. 위치 제어
                error_p = R_added.as_matrix() @ self.zero_target_tri[i] * self.TRI_RADIUS + self.zero_target_bias[i] - self.curr_pos[i]
                J = pin.computeFrameJacobian(self.model, self.data, self.q, tid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_v = J[:3, :]; J_w = J[3:6, :] 
                v_eef = J_v @ self.v
                force_p = self.K * error_p
                force_pd = self.D * v_eef
                # R_curr = R.from_matrix(self.data.oMf[tid].rotation)
                # R_err = R_target * R_curr.inv()
                # force_o = np.array([self.K_ori_R, self.K_ori_P, self.K_ori_Y]) * R_err.as_rotvec() 
                
                tau_task += J_v.T @ force_p
                tau_task_damper += J_v.T @ force_pd
                # tau_task += J_v.T @ force_p + J_w.T @ force_o

                # 로깅 데이터 (EEF)
                current_log.update({f'f{i}_des_x': self.target_pos[i][0], f'f{i}_curr_x': self.curr_pos[i][0]})
                current_log.update({f'f{i}_des_y': self.target_pos[i][1], f'f{i}_curr_y': self.curr_pos[i][1]})
                current_log.update({f'f{i}_des_z': self.target_pos[i][2], f'f{i}_curr_z': self.curr_pos[i][2]})
                current_log.update({f'f{i}_distance': np.linalg.norm(error_p)})

            # B. 최종 토크 합산: Task 토크 + (중력 보상 게인 * 중력 토크) + 관절 댐핑
            tau_total = tau_task + (self.gravity_comp_gain * tau_gravity)
            # self.get_logger().info(f"{tau_total}")
            # feed forward 마찰보상
            tanh_friction = lambda x: x + self.F_FRIC_BIAS * np.tanh(x / self.F_FRIC_STATIC)
            tau_total = tanh_friction(tau_total) - tau_task_damper + (self.D_joint * self.D_joint_weight * self.v)

            # 토크 명령 전송
            msg = Float64MultiArray()
            msg.data = tau_total.tolist()
            self.torque_pub.publish(msg)
            
            self.data_logs.append(current_log)
            self.publish_triangle_marker()
            rate.sleep()

    def publish_triangle_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"; marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP; marker.id = 99; marker.scale.x = 0.003
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0
        for p in self.curr_pos:
            pt = Point(); pt.x, pt.y, pt.z = p[0], p[1], p[2]
            marker.points.append(pt)
        marker.points.append(marker.points[0])
        self.marker_pub.publish(marker)

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
    root.geometry("400x650")
    
    # 변수 매핑
    k_var = DoubleVar(value=node.K); d_var = DoubleVar(value=node.D); dj_var = DoubleVar(value=node.D_joint)
    g_var = DoubleVar(value=node.gravity_comp_gain * 100) # 0-100% scale
    ko_r = DoubleVar(value=node.K_ori_R); ko_p = DoubleVar(value=node.K_ori_P); ko_y = DoubleVar(value=node.K_ori_Y)
    r_var = DoubleVar(value=0.0); p_var = DoubleVar(value=0.0); y_var = DoubleVar(value=45.0)
    pos_z_var = DoubleVar(value=0.3)
    fric_static_var = DoubleVar(value=node.F_FRIC_STATIC)
    fric_bias_var = DoubleVar(value=node.F_FRIC_BIAS)
    radius_var = DoubleVar(value=node.TRI_RADIUS)

    def update_params(*args):
        node.K = k_var.get(); node.D = d_var.get(); node.D_joint = dj_var.get()
        node.gravity_comp_gain = g_var.get() / 100.0
        # node.K_ori_R = ko_r.get(); node.K_ori_P = ko_p.get(); node.K_ori_Y = ko_y.get()
        node.target_roll = np.radians(r_var.get()); node.target_pitch = np.radians(p_var.get()); node.target_yaw = np.radians(y_var.get())
        for i in range(3):
            node.zero_target_bias[i][2] = pos_z_var.get()
        node.F_FRIC_STATIC = fric_static_var.get()
        node.F_FRIC_BIAS = fric_bias_var.get()
        node.TRI_RADIUS = radius_var.get()

    # UI 레이아웃
    Label(root, text="[Gravity Compensation]", fg="blue", font=("Helvetica", 10, "bold")).pack(pady=(1,0))
    Scale(root, from_=0, to=150, resolution=1, orient=HORIZONTAL, variable=g_var, label="Gravity Comp (%)", command=update_params).pack(fill='x', padx=20)

    Label(root, text="[Position Control]").pack(pady=(1,0))
    Scale(root, from_=0, to=200, resolution=1, orient=HORIZONTAL, variable=k_var, label="K_p", command=update_params).pack(fill='x', padx=20)
    Scale(root, from_=0, to=5, resolution=0.01, orient=HORIZONTAL, variable=d_var, label="D_p", command=update_params).pack(fill='x', padx=20)
    Scale(root, from_=-0.1, to=0.1, resolution=0.001, orient=HORIZONTAL, variable=dj_var, label="D_joint", command=update_params).pack(fill='x', padx=20)
    
    Label(root, text="[Reference RPY (deg)]").pack(pady=(1,0))
    Scale(root, from_=-45, to=45, orient=HORIZONTAL, variable=r_var, label="Roll", command=update_params).pack(fill='x', padx=20)
    Scale(root, from_=-45, to=45, orient=HORIZONTAL, variable=p_var, label="Pitch", command=update_params).pack(fill='x', padx=20)
    Scale(root, from_=-45, to=45, orient=HORIZONTAL, variable=y_var, label="Yaw", command=update_params).pack(fill='x', padx=20)
    Scale(root, from_=-1, to=1, resolution=0.01, orient=HORIZONTAL, variable=pos_z_var, label="Z", command=update_params).pack(fill='x', padx=20)
    Scale(root, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, variable=radius_var, label="Radius", command=update_params).pack(fill='x', padx=20)

    Label(root, text="[Friction Parameter]").pack(pady=(1,0))
    Scale(root, from_=0, to=0.1, resolution=0.001, orient=HORIZONTAL, variable=fric_static_var, label="F_FRIC_STATIC", command=update_params).pack(fill='x', padx=20)
    Scale(root, from_=0, to=0.2, resolution=0.001, orient=HORIZONTAL, variable=fric_bias_var, label="F_FRIC_BIAS", command=update_params).pack(fill='x', padx=20)

    root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveSpringDamperNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True); spin_thread.start()
    try: run_ui(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()