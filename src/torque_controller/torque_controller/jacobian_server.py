#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from robot_interfaces.srv import ComputeTorque

class JacobianTransposeServer(Node):

    def __init__(self):
        super().__init__('jacobian_transpose_server')

        # 1. Load URDF
        pkg_name = 'torque_controller'
        urdf_filename = 'hand_0926.urdf' # Ensure this matches your file

        try:
            pkg_share = get_package_share_directory(pkg_name)
            urdf_path = os.path.join(pkg_share, 'urdf', urdf_filename)
        except Exception:
            self.get_logger().error(f"Could not find package '{pkg_name}'")
            return

        # 2. Build Model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # 3. Define the 3 Tip Frames (Order matters! Must match input force order)
        self.tip_frames = ['FL1EEF', 'FL2EEF', 'FL3EEF']
        self.tip_ids = []

        # Validate frames and cache their IDs
        for frame_name in self.tip_frames:
            if not self.model.existFrame(frame_name):
                self.get_logger().error(f"Frame {frame_name} not found in URDF!")
            else:
                self.tip_ids.append(self.model.getFrameId(frame_name))

        self.get_logger().info(f"Server Ready. Controlling tips: {self.tip_frames}")
        
        # 4. Create Service
        self.srv = self.create_service(
            ComputeTorque,
            'compute_torque_from_force',
            self.compute_callback
        )

    def compute_callback(self, request, response):
        """
        Expects:
        - joint_positions: 12 floats
        - task_force_1: 6 floats
        - task_force_2: 6 floats
        - task_force_3: 6 floats
        """
        try:
            q = np.array(request.joint_positions)
            
            # Organize inputs into a list for easy iteration
            # We convert them to numpy arrays immediately
            forces_list = [
                np.array(request.task_force_1),
                np.array(request.task_force_2),
                np.array(request.task_force_3)
            ]

            # --- Validation ---
            if len(q) != self.model.nq:
                self.get_logger().warn(f"Joint size mismatch. Got {len(q)}, expected {self.model.nq}")
                response.success = False
                return response
            
            # Check if each force vector has exactly 6 elements
            for i, f in enumerate(forces_list):
                if len(f) != 6:
                    self.get_logger().warn(f"Force {i+1} has wrong size: {len(f)} (Expected 6)")
                    response.success = False
                    return response

            # --- Calculation ---
            # 1. Update Kinematics
            pin.framesForwardKinematics(self.model, self.data, q)
            
            # Initialize total torque accumulator
            tau_total = np.zeros(self.model.nq)

            # 2. Iterate through fingers
            for i, frame_id in enumerate(self.tip_ids):
                f_current = forces_list[i]

                # Optimization: If force is zero, skip computation
                if np.linalg.norm(f_current) < 1e-6:
                    continue

                # Compute Jacobian for this specific tip
                J = pin.computeFrameJacobian(
                    self.model, 
                    self.data, 
                    q, 
                    frame_id, 
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )

                # Tau = J^T * F
                # Note: J.T is (12x6), f_current is (6,) -> Result is (12,)
                tau_i = J.T @ f_current
                
                # Add to total
                tau_total += tau_i
                
                # Optional: Debug log to see individual finger contributions
                # self.get_logger().info(f"Finger {i+1} Torque: {np.round(tau_i, 2)}")

            # --- Export ---
            response.joint_torques = tau_total.tolist()
            response.success = True
            
            # self.get_logger().info(f"Total Torque: {np.round(tau_total, 2)}")

        except Exception as e:
            self.get_logger().error(f"Calculation failed: {e}")
            response.success = False

        return response

def main(args=None):
    rclpy.init(args=args)
    node = JacobianTransposeServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()