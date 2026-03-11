import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class EKF:

    def __init__(self, Q, R, dt):

        self.dt = dt

        # μ and Σ
        self.mu = np.zeros((3, 1))
        self.Sigma = 0.5 * np.eye(3)

        # Noise matrices
        self.R = Q      # Motion noise (R_t in slides)
        self.Q = R      # Measurement noise (Q_t in slides)

    # g(u, μ)
    def motion_model(self, mu, u):

        v, omega = u
        x, y, theta = mu.flatten()
        dt = self.dt

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        return np.array([[x_new], [y_new], [theta_new]])

    # Jacobian G_t
    def motion_jacobian(self, mu, u):

        v, _ = u
        _, _, theta = mu.flatten()
        dt = self.dt

        G = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0, 1]
        ])

        return G

    # h(μ)
    def measurement_model(self, mu, landmark):

        lx, ly = landmark
        x, y, theta = mu.flatten()

        dx = lx - x
        dy = ly - y

        r = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx) - theta

        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))

        return np.array([[r], [bearing]])

    # Jacobian H_t
    def measurement_jacobian(self, mu, landmark):

        lx, ly = landmark
        x, y, theta = mu.flatten()

        dx = lx - x
        dy = ly - y

        q = dx**2 + dy**2
        r = np.sqrt(q)

        H = np.array([
            [-dx/r, -dy/r, 0],
            [ dy/q, -dx/q, -1]
        ])

        return H
    def step(self, u, z, landmark):

        mu_bar = self.motion_model(self.mu, u)
        G = self.motion_jacobian(self.mu, u)
        Sigma_bar = G @ self.Sigma @ G.T + self.R

        H = self.measurement_jacobian(mu_bar, landmark)
        z_pred = self.measurement_model(mu_bar, landmark)

        S = H @ Sigma_bar @ H.T + self.Q
        K = Sigma_bar @ H.T @ np.linalg.inv(S)

        innovation = z.reshape(2,1) - z_pred
        innovation[1] = np.arctan2(np.sin(innovation[1]),
                                   np.cos(innovation[1]))

        self.mu = mu_bar + K @ innovation
        self.Sigma = (np.eye(3) - K @ H) @ Sigma_bar

class Robot:

    def __init__(self, x=0, y=0, theta=0,
                 sigma_v=0.1, sigma_omega=0.05):

        self.x = x
        self.y = y
        self.theta = theta

        self.sigma_v = sigma_v
        self.sigma_omega = sigma_omega

        self.history_x = [x]
        self.history_y = [y]

    def move(self, v, omega, dt):

        v_noisy = v + np.random.normal(0, self.sigma_v)
        omega_noisy = omega + np.random.normal(0, self.sigma_omega)

        self.x += v_noisy * np.cos(self.theta) * dt
        self.y += v_noisy * np.sin(self.theta) * dt
        self.theta += omega_noisy * dt
        self.theta = np.arctan2(np.sin(self.theta),
                                np.cos(self.theta))

        self.history_x.append(self.x)
        self.history_y.append(self.y)

    def get_pose(self):
        return self.x, self.y, self.theta

class Simulator:

    def __init__(self, robot, landmarks, v, omega,
                 dt=0.05, sim_time=20):

        self.robot = robot
        self.landmarks = landmarks
        self.v = v
        self.omega = omega
        self.dt = dt
        self.frames = int(sim_time / dt)

        motion_noise = np.diag([0.05, 0.05, 0.01])
        meas_noise = np.diag([0.1, 0.05])

        self.ekf = EKF(motion_noise, meas_noise, dt)

        # Plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        lm = np.array(landmarks)
        self.ax.scatter(lm[:,0], lm[:,1],
                        marker='*', s=200)

        self.true_line, = self.ax.plot([], [])
        self.ekf_line, = self.ax.plot([], [], '--')

        self.ekf_x = []
        self.ekf_y = []

    def update(self, frame):

        # True motion
        self.robot.move(self.v, self.omega, self.dt)
        x, y, theta = self.robot.get_pose()

        # Simulated measurement from first landmark
        lx, ly = self.landmarks[0]
        dx = lx - x
        dy = ly - y

        r = np.sqrt(dx**2 + dy**2) + np.random.normal(0,0.1)
        bearing = np.arctan2(dy,dx) - theta + np.random.normal(0,0.05)

        z = np.array([r, bearing])

        # EKF full step
        self.ekf.step((self.v, self.omega), z, (lx, ly))

        est_x, est_y, _ = self.ekf.mu.flatten()
        self.ekf_x.append(est_x)
        self.ekf_y.append(est_y)

        self.true_line.set_data(self.robot.history_x,
                                self.robot.history_y)
        self.ekf_line.set_data(self.ekf_x,
                               self.ekf_y)

        return self.true_line, self.ekf_line

    def run(self):

        self.ani = FuncAnimation(self.fig,
                                 self.update,
                                 frames=self.frames,
                                 interval=self.dt*1000,
                                 blit=True)

        plt.show()

if __name__ == "__main__":

    robot = Robot()

    landmarks = [
        (4,4),
        (-4,3),
        (-3,-4),
        (5,-5)
    ]

    sim = Simulator(robot,
                    landmarks,
                    v=2.0,
                    omega=0.4)

    sim.run()