import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def step_unicycle(q, v, w, dt):
    x, y, th = q
    x_new = x + v * np.cos(th) * dt
    y_new = y + v * np.sin(th) * dt
    th_new = wrap_angle(th + w * dt)
    return np.array([x_new, y_new, th_new])

def compute_bounds(trajs, landmarks=None, pad=1.0):
    xs = np.concatenate([tr[:, 0] for tr in trajs])
    ys = np.concatenate([tr[:, 1] for tr in trajs])

    if landmarks is not None:
        xs = np.concatenate([xs, landmarks[:, 0]])
        ys = np.concatenate([ys, landmarks[:, 1]])

    xmin, xmax = xs.min() - pad, xs.max() + pad
    ymin, ymax = ys.min() - pad, ys.max() + pad
    return xmin, xmax, ymin, ymax

def measure_range_bearing(q, landmark, noisy=False, sigma_r=0.0, sigma_b=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    x, y, th = q
    lx, ly = landmark
    dx, dy = lx - x, ly - y
    r_true = np.hypot(dx, dy)
    b_true = wrap_angle(np.arctan2(dy, dx) - th)

    if not noisy:
        return np.array([r_true, b_true])

    r = r_true + rng.normal(0.0, sigma_r)
    b = wrap_angle(b_true + rng.normal(0.0, sigma_b))
    return np.array([r, b])

def rb_to_xy(q, r, b):
    x, y, th = q
    ang = th + b
    return x + r * np.cos(ang), y + r * np.sin(ang)

class EKF:
    def __init__(self, Q_motion, R_meas, dt, mu0=None, Sigma0=None):
        """
        Q_motion: 3x3 (motion/process covariance added in prediction)
        R_meas:   2x2 (measurement covariance for [range, bearing])
        """
        self.dt = dt
        self.mu = np.zeros((3, 1)) if mu0 is None else mu0.reshape(3, 1).copy()
        self.Sigma = 0.5 * np.eye(3) if Sigma0 is None else Sigma0.copy()

        # Naming like your EKF example:
        self.R = Q_motion   # motion noise
        self.Q = R_meas     # measurement noise

    def motion_model(self, mu, u):
        v, omega = u
        x, y, theta = mu.flatten()
        dt = self.dt

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = wrap_angle(theta + omega * dt)
        return np.array([[x_new], [y_new], [theta_new]])

    def motion_jacobian(self, mu, u):
        v, _ = u
        _, _, theta = mu.flatten()
        dt = self.dt
        G = np.array([
            [1.0, 0.0, -v * np.sin(theta) * dt],
            [0.0, 1.0,  v * np.cos(theta) * dt],
            [0.0, 0.0, 1.0]
        ])
        return G

    def measurement_model(self, mu, landmark):
        lx, ly = landmark
        x, y, theta = mu.flatten()

        dx = lx - x
        dy = ly - y
        r = np.sqrt(dx**2 + dy**2)
        bearing = wrap_angle(np.arctan2(dy, dx) - theta)

        return np.array([[r], [bearing]])

    def measurement_jacobian(self, mu, landmark):
        lx, ly = landmark
        x, y, theta = mu.flatten()

        dx = lx - x
        dy = ly - y

        q = dx**2 + dy**2
        r = np.sqrt(q)

        # safety (avoid divide-by-zero if robot hits landmark)
        eps = 1e-9
        r = max(r, eps)
        q = max(q, eps)

        H = np.array([
            [-dx / r, -dy / r, 0.0],
            [ dy / q, -dx / q, -1.0]
        ])
        return H

    def step(self, u, z, landmark):
        # Predict
        mu_bar = self.motion_model(self.mu, u)
        G = self.motion_jacobian(self.mu, u)
        Sigma_bar = G @ self.Sigma @ G.T + self.R

        # Update
        H = self.measurement_jacobian(mu_bar, landmark)
        z_pred = self.measurement_model(mu_bar, landmark)

        S = H @ Sigma_bar @ H.T + self.Q
        K = Sigma_bar @ H.T @ np.linalg.inv(S)

        innovation = z.reshape(2, 1) - z_pred
        innovation[1] = wrap_angle(innovation[1])

        self.mu = mu_bar + K @ innovation
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])
        self.Sigma = (np.eye(3) - K @ H) @ Sigma_bar

        return self.mu, self.Sigma, mu_bar, Sigma_bar, K

# Simulation: noisy motion + noisy meas + EKF
def simulate_noisy_motion_and_ekf(
    q0, N, dt, v_cmd, w_cmd,
    sigma_v, sigma_w,
    sigma_r, sigma_b,
    landmarks,
    used_landmark_idx=0,
    seed_motion=7,
    seed_meas=123
):
    rng_m = np.random.default_rng(seed_motion)
    rng_z = np.random.default_rng(seed_meas)

    # "True" (noisy) motion trajectory
    q_true = np.zeros((N, 3))
    q = q0.copy()
    for k in range(N):
        v = v_cmd + rng_m.normal(0.0, sigma_v)
        w = w_cmd + rng_m.normal(0.0, sigma_w)
        q = step_unicycle(q, v, w, dt)
        q_true[k] = q

    # Noisy measurements z_k = [r, b] to one landmark
    lm = tuple(landmarks[used_landmark_idx])
    z_hist = np.zeros((N, 2))
    meas_xy = np.zeros((N, 2))  # for visualization (measured point in world)
    for k in range(N):
        z = measure_range_bearing(q_true[k], lm, noisy=True, sigma_r=sigma_r, sigma_b=sigma_b, rng=rng_z)
        z_hist[k] = z
        mx, my = rb_to_xy(q_true[k], z[0], z[1])
        meas_xy[k] = [mx, my]

    # EKF init (similar vibe: start at origin with some covariance)
    Q_motion = np.diag([0.05**2, 0.05**2, 0.01**2])  # tune like your EKF example
    R_meas = np.diag([sigma_r**2, sigma_b**2])

    # reasonable init: use q0, but you can set to zeros if you want
    mu0 = np.array([q0[0], q0[1], q0[2]])
    Sigma0 = np.diag([0.5**2, 0.5**2, np.deg2rad(20)**2])

    ekf = EKF(Q_motion, R_meas, dt, mu0=mu0, Sigma0=Sigma0)

    mu_hist = np.zeros((N, 3))
    for k in range(N):
        ekf.step((v_cmd, w_cmd), z_hist[k], lm)
        mu_hist[k] = ekf.mu.flatten()

    return q_true, z_hist, mu_hist, meas_xy, lm

def animate_noisy_vs_ekf(
    q_true, mu_hist, meas_xy, landmarks, used_landmark,
    title, out_gif, fps=25
):
    N = q_true.shape[0]
    trajs_for_bounds = [q_true, mu_hist]
    xmin, xmax, ymin, ymax = compute_bounds(trajs_for_bounds, landmarks=landmarks, pad=1.0)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    # Landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker="s", color="black", label="landmarks")
    ax.scatter([used_landmark[0]], [used_landmark[1]], marker="*", s=180, color="black", label="used landmark")

    # Lines
    (true_ln,) = ax.plot([], [], linewidth=2.2, color="black", label="true (noisy motion)")
    (ekf_ln,)  = ax.plot([], [], linewidth=2.2, linestyle="--", color="purple", label="EKF estimate")

    # Current position dots
    (true_pt,) = ax.plot([], [], marker="o", markersize=6, linestyle="None", color="black")
    (ekf_pt,)  = ax.plot([], [], marker="o", markersize=6, linestyle="None", color="purple")

    # Optional: measurement ray to measured point
    (meas_ray,) = ax.plot([], [], linewidth=1.0, color="red", alpha=0.9, label="measurement ray")
    (meas_pt,)  = ax.plot([], [], marker="x", markersize=7, linestyle="None", color="red", alpha=0.9)

    ax.legend(loc="upper right")

    def init():
        true_ln.set_data([], [])
        ekf_ln.set_data([], [])
        true_pt.set_data([], [])
        ekf_pt.set_data([], [])
        meas_ray.set_data([], [])
        meas_pt.set_data([], [])
        return [true_ln, ekf_ln, true_pt, ekf_pt, meas_ray, meas_pt]

    def update(k):
        # paths
        true_ln.set_data(q_true[:k+1, 0], q_true[:k+1, 1])
        ekf_ln.set_data(mu_hist[:k+1, 0], mu_hist[:k+1, 1])

        # current points
        tx, ty = q_true[k, 0], q_true[k, 1]
        ex, ey = mu_hist[k, 0], mu_hist[k, 1]
        true_pt.set_data([tx], [ty])
        ekf_pt.set_data([ex], [ey])

        # measurement viz (from true pose to measured point)
        mx, my = meas_xy[k]
        meas_ray.set_data([tx, mx], [ty, my])
        meas_pt.set_data([mx], [my])

        return [true_ln, ekf_ln, true_pt, ekf_pt, meas_ray, meas_pt]

    anim = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=int(1000 / fps))
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {out_gif}")

if __name__ == "__main__":
    dt = 0.05
    T = 40.0
    N = int(T / dt)

    R_circle = 3.0
    v_cmd = 0.6
    w_cmd = v_cmd / R_circle

    # Motion noise (inputs)
    sigma_v = 0.25
    sigma_w = 0.20

    # Measurement noise (range-bearing)
    sigma_r = 0.20
    sigma_b = np.deg2rad(2.0)

    q0 = np.array([0.0, 0.0, 0.0])

    landmarks = np.array([
        [4.0, 4.0],
        [6.0, 1.0],
        [1.0, 6.5],
        [-2.0, 5.0],
        [-4.5, 1.5],
        [-3.5, -2.5],
        [2.5, -3.5],
        [6.0, -2.0],
    ])

    q_true, z_hist, mu_hist, meas_xy, used_lm = simulate_noisy_motion_and_ekf(
        q0=q0, N=N, dt=dt,
        v_cmd=v_cmd, w_cmd=w_cmd,
        sigma_v=sigma_v, sigma_w=sigma_w,
        sigma_r=sigma_r, sigma_b=sigma_b,
        landmarks=landmarks,
        used_landmark_idx=0,
        seed_motion=7,
        seed_meas=123
    )

    animate_noisy_vs_ekf(
        q_true=q_true,
        mu_hist=mu_hist,
        meas_xy=meas_xy,
        landmarks=landmarks,
        used_landmark=used_lm,
        title="Simulation 5: Noisy motion + noisy measurement + EKF correction",
        out_gif="sim5_noisy_motion_ekf.gif",
        fps=25
    )