import numpy as np


class BallKalmanFilter:
    """Kalman filter for ball state estimation.

    State: [x, y, vx, vy] (ball position and velocity on plate)
    Control: [pitch_cmd, roll_cmd] → tilt-induced acceleration
    Measurement: [bx, by] (CoP position)
    """

    def __init__(self, dt, g=9.81):
        self.dt = dt
        self.g = g

        self.x = np.zeros(4)
        self.P = np.diag([0.01, 0.01, 0.1, 0.1])

        # Constant-velocity kinematic model
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Control: tilt → acceleration (roll axis inverted)
        self.B = np.array([
            [g * dt**2 / 2, 0],
            [0, -g * dt**2 / 2],
            [g * dt, 0],
            [0, -g * dt]
        ])

        # Measurement: position only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise
        q = 0.02 * g * 0.17
        self.Q = np.diag([
            q**2 * dt**4 / 4,
            q**2 * dt**4 / 4,
            q**2 * dt**2,
            q**2 * dt**2
        ])

    def predict(self, u_pitch, u_roll):
        u = np.array([u_pitch, u_roll])
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_x, z_y, f_total):
        if f_total < 0:
            return
        if f_total < 10:
            r_val = 1e-4
        elif f_total < 30:
            r_val = 1e-6
        else:
            r_val = 1e-8
        R = np.diag([r_val, r_val])
        z = np.array([z_x, z_y])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self):
        return self.x[0:2]

    @property
    def velocity(self):
        return self.x[2:4]

    @property
    def P_trace(self):
        return self.P.trace()
