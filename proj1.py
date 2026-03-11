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

def motion_and_measurement_step(q, u, landmark, R, Q, rng, dt):
    """
    One simulation step with both process noise R and measurement noise Q.

    q_{k+1} = f(q_k, u_k) + e_k,   e_k ~ N(0, R)
    z_k     = h(q_{k+1}) + v_k,    v_k ~ N(0, Q)

    state q = [x, y, theta]
    measurement z = [range, bearing]
    """
    v_cmd, w_cmd = u
    x, y, th = q

    # Nominal motion
    x_mean = x + v_cmd * np.cos(th) * dt
    y_mean = y + v_cmd * np.sin(th) * dt
    th_mean = wrap_angle(th + w_cmd * dt)

    q_mean = np.array([x_mean, y_mean, th_mean])

    # Process noise from covariance R
    e = rng.multivariate_normal(mean=np.zeros(3), cov=R)

    q_new = q_mean + e
    q_new[2] = wrap_angle(q_new[2])

    # Measurement from new true state
    lx, ly = landmark
    dx = lx - q_new[0]
    dy = ly - q_new[1]

    r_true = np.hypot(dx, dy)
    b_true = wrap_angle(np.arctan2(dy, dx) - q_new[2])

    z_mean = np.array([r_true, b_true])

    # Measurement noise from covariance Q
    v_meas = rng.multivariate_normal(mean=np.zeros(2), cov=Q)

    z = z_mean + v_meas
    z[1] = wrap_angle(z[1])

    return q_new, z

def simulate_ground_truth_noiseless(q0, N, dt, v_cmd, w_cmd):
    """
    Ideal ground-truth trajectory with no process noise and no measurements.
    """
    q_hist = np.zeros((N, 3))
    q = q0.copy()

    for k in range(N):
        q = step_unicycle(q, v_cmd, w_cmd, dt)
        q_hist[k] = q

    return q_hist

def is_landmark_in_fov(q, landmark, fov_deg=180.0, max_range=np.inf):
    x, y, th = q
    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    r = np.hypot(dx, dy)
    b = wrap_angle(np.arctan2(dy, dx) - th)

    half_fov = np.deg2rad(fov_deg / 2.0)

    in_fov = (-half_fov <= b <= half_fov)
    in_range = (r <= max_range)

    return in_fov and in_range, r, b

def visible_landmarks(q, landmarks, fov_deg=180.0, max_range=np.inf):
    visible = []

    for i, lm in enumerate(landmarks):
        ok, r, b = is_landmark_in_fov(q, lm, fov_deg=fov_deg, max_range=max_range)
        if ok:
            visible.append((i, lm, r, b))

    return visible

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

# Simulation: noisy motion with R + noisy measurement with Q + EKF
def simulate_noisy_motion_and_ekf(
    q0, N, dt, v_cmd, w_cmd,
    R_motion, Q_meas,
    landmarks,
    used_landmark_idx=0,
    seed=7
):
    rng = np.random.default_rng(seed)

    lm = tuple(landmarks[used_landmark_idx])

    # True noisy motion + noisy measurements
    q_true = np.zeros((N, 3))
    z_hist = np.zeros((N, 2))
    meas_xy = np.zeros((N, 2))

    q = q0.copy()
    u = (v_cmd, w_cmd)

    for k in range(N):
        q, z = motion_and_measurement_step(q, u, lm, R_motion, Q_meas, rng, dt)
        q_true[k] = q
        z_hist[k] = z

        mx, my = rb_to_xy(q, z[0], z[1])
        meas_xy[k] = [mx, my]

    # EKF init
    mu0 = np.array([q0[0], q0[1], q0[2]])
    Sigma0 = np.diag([0.5**2, 0.5**2, np.deg2rad(20)**2])

    ekf = EKF(R_motion, Q_meas, dt, mu0=mu0, Sigma0=Sigma0)

    mu_hist = np.zeros((N, 3))
    Sigma_hist = np.zeros((N, 3, 3))

    for k in range(N):
        ekf.step((v_cmd, w_cmd), z_hist[k], lm)
        mu_hist[k] = ekf.mu.flatten()
        Sigma_hist[k] = ekf.Sigma

    return q_true, z_hist, mu_hist, Sigma_hist, meas_xy, lm

def compute_fov_measurements_for_animation(
    q_true, landmarks, fov_deg=180.0, max_range=np.inf
):
    """
    For each true robot pose, compute which landmarks are visible.
    This is ONLY for visualization. It does NOT affect the EKF.
    
    Returns:
        visible_rays[k] = list of tuples (landmark_index, robot_x, robot_y, meas_x, meas_y)
    """
    visible_rays = []

    for q in q_true:
        x, y, th = q
        rays_k = []

        for i, lm in enumerate(landmarks):
            ok, r, b = is_landmark_in_fov(q, lm, fov_deg=fov_deg, max_range=max_range)
            if ok:
                mx, my = rb_to_xy(q, r, b)
                rays_k.append((i, x, y, mx, my))

        visible_rays.append(rays_k)

    return visible_rays

def animate_noisy_vs_ekf(
    q_true, mu_hist, q_gt_noiseless, meas_xy, landmarks, used_landmark,
    title, out_gif, fps=25, fov_deg=180.0, max_range=5.0
):
    N = q_true.shape[0]
    trajs_for_bounds = [q_true, mu_hist]
    xmin, xmax, ymin, ymax = compute_bounds(trajs_for_bounds, landmarks=landmarks, pad=1.0)

    # Visible landmarks/rays ONLY for animation
    visible_rays = compute_fov_measurements_for_animation(
        q_true, landmarks, fov_deg=fov_deg, max_range=max_range
    )

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
    ax.scatter([used_landmark[0]], [used_landmark[1]], marker="*", s=180, color="black", label="EKF landmark")

    # Lines
    (true_ln,) = ax.plot([], [], linewidth=2.2, color="black", label="true (noisy motion)")
    (ekf_ln,)  = ax.plot([], [], linewidth=2.2, linestyle="--", color="purple", label="EKF estimate")

    # Current position dots
    (true_pt,) = ax.plot([], [], marker="o", markersize=6, linestyle="None", color="black")
    (ekf_pt,)  = ax.plot([], [], marker="o", markersize=6, linestyle="None", color="purple")

    # FOV boundary lines
    (fov_left_ln,) = ax.plot([], [], linewidth=1.2, linestyle="-.", color="green", label="FOV")
    (fov_right_ln,) = ax.plot([], [], linewidth=1.2, linestyle="-.", color="green")

    # Up to 2 measurement rays for visible landmarks
    (meas_ray_1,) = ax.plot([], [], linewidth=1.0, color="red", alpha=0.9, label="visible ray")
    (meas_pt_1,) = ax.plot([], [], marker="x", markersize=7, linestyle="None", color="red", alpha=0.9)

    (meas_ray_2,) = ax.plot([], [], linewidth=1.0, color="orange", alpha=0.9, label="2nd visible ray")
    (meas_pt_2,) = ax.plot([], [], marker="x", markersize=7, linestyle="None", color="orange", alpha=0.9)

    half_fov = np.deg2rad(fov_deg / 2.0)

    (gt_ln,) = ax.plot([], [], linewidth=2.0, color="lightgray",
                   linestyle="--", label="ground truth (noiseless)")
    
    ax.legend(loc="upper right")

    def init():
        true_ln.set_data([], [])
        ekf_ln.set_data([], [])
        true_pt.set_data([], [])
        ekf_pt.set_data([], [])

        fov_left_ln.set_data([], [])
        fov_right_ln.set_data([], [])

        meas_ray_1.set_data([], [])
        meas_pt_1.set_data([], [])

        meas_ray_2.set_data([], [])
        meas_pt_2.set_data([], [])

        gt_ln.set_data([], [])

        return [
            true_ln, ekf_ln, true_pt, ekf_pt,
            fov_left_ln, fov_right_ln,
            meas_ray_1, meas_pt_1,
            meas_ray_2, meas_pt_2,
            gt_ln
        ]

    def update(k):
        gt_ln.set_data(q_gt_noiseless[:k+1,0], q_gt_noiseless[:k+1,1])
        true_ln.set_data(q_true[:k+1, 0], q_true[:k+1, 1])
        ekf_ln.set_data(mu_hist[:k+1, 0], mu_hist[:k+1, 1])

        tx, ty, tth = q_true[k]
        ex, ey = mu_hist[k, 0], mu_hist[k, 1]

        true_pt.set_data([tx], [ty])
        ekf_pt.set_data([ex], [ey])

        # FOV lines
        left_ang = tth + half_fov
        right_ang = tth - half_fov

        x_left = tx + max_range * np.cos(left_ang)
        y_left = ty + max_range * np.sin(left_ang)

        x_right = tx + max_range * np.cos(right_ang)
        y_right = ty + max_range * np.sin(right_ang)

        fov_left_ln.set_data([tx, x_left], [ty, y_left])
        fov_right_ln.set_data([tx, x_right], [ty, y_right])

        # Visible rays (ONLY visualization)
        rays = visible_rays[k]

        # Reset both rays
        meas_ray_1.set_data([], [])
        meas_pt_1.set_data([], [])
        meas_ray_2.set_data([], [])
        meas_pt_2.set_data([], [])

        if len(rays) >= 1:
            _, sx, sy, mx, my = rays[0]
            meas_ray_1.set_data([sx, mx], [sy, my])
            meas_pt_1.set_data([mx], [my])

        if len(rays) >= 2:
            _, sx, sy, mx, my = rays[1]
            meas_ray_2.set_data([sx, mx], [sy, my])
            meas_pt_2.set_data([mx], [my])

        return [
            true_ln, ekf_ln, true_pt, ekf_pt,
            fov_left_ln, fov_right_ln,
            meas_ray_1, meas_pt_1,
            meas_ray_2, meas_pt_2,
            gt_ln
        ]

    anim = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=int(1000 / fps))
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {out_gif}")

def animate_ground_truth_only(
    q_gt, title, out_gif, fps=25
):
    N = q_gt.shape[0]
    xmin, xmax, ymin, ymax = compute_bounds([q_gt], landmarks=None, pad=1.0)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    (traj_ln,) = ax.plot([], [], linewidth=2.2, color="black", label="ground truth (noiseless)")
    (traj_pt,) = ax.plot([], [], marker="o", markersize=6, linestyle="None", color="black")

    ax.legend(loc="upper right")

    def init():
        traj_ln.set_data([], [])
        traj_pt.set_data([], [])
        return [traj_ln, traj_pt]

    def update(k):
        traj_ln.set_data(q_gt[:k+1, 0], q_gt[:k+1, 1])
        traj_pt.set_data([q_gt[k, 0]], [q_gt[k, 1]])
        return [traj_ln, traj_pt]

    anim = FuncAnimation(
        fig, update, frames=N, init_func=init,
        blit=True, interval=int(1000 / fps)
    )
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {out_gif}")


# def plot_mu_and_sigma(time, mu_hist, Sigma_hist):
#     """
#     Plot 1: ||mu|| vs time
#     Plot 2: ||Sigma|| vs time
#     """

#     mu_norm = np.linalg.norm(mu_hist, axis=1)

#     sigma_norm = np.array([
#         np.linalg.norm(Sigma_hist[k], ord='fro')
#         for k in range(len(Sigma_hist))
#     ])

#     # ---- mu plot ----
#     plt.figure(figsize=(8,5))
#     plt.plot(time, mu_norm, color="purple", linewidth=2)
#     plt.xlabel("time [s]")
#     plt.ylabel("||mu||")
#     plt.title("Magnitude of EKF state estimate ||μ(t)||")
#     plt.grid(True)

#     # ---- Sigma plot ----
#     plt.figure(figsize=(8,5))
#     plt.plot(time, sigma_norm, color="darkred", linewidth=2)
#     plt.xlabel("time [s]")
#     plt.ylabel("||Sigma||")
#     plt.title("Magnitude of covariance ||Σ(t)||")
#     plt.grid(True)

#     plt.show()

def animate_mu_sigma(time, mu_hist, Sigma_hist, out_gif="mu_sigma_evolution.gif", fps=25):
    """
    Real-time animation of:
    1) ||mu|| vs time
    2) ||Sigma|| vs time
    """

    mu_norm = np.linalg.norm(mu_hist, axis=1)

    sigma_norm = np.array([
        np.linalg.norm(Sigma_hist[k], ord='fro')
        for k in range(len(Sigma_hist))
    ])

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].set_ylabel("||μ||")
    ax[0].set_title("State estimate magnitude")
    ax[0].grid(True)

    ax[1].set_ylabel("||Σ||")
    ax[1].set_xlabel("time [s]")
    ax[1].set_title("Covariance magnitude")
    ax[1].grid(True)

    ax[0].set_xlim(time[0], time[-1])
    ax[1].set_xlim(time[0], time[-1])

    ax[0].set_ylim(0, np.max(mu_norm)*1.2)
    ax[1].set_ylim(0, np.max(sigma_norm)*1.2)

    (mu_line,) = ax[0].plot([], [], color="purple", linewidth=2)
    (sigma_line,) = ax[1].plot([], [], color="darkred", linewidth=2)

    def init():
        mu_line.set_data([], [])
        sigma_line.set_data([], [])
        return mu_line, sigma_line

    def update(k):
        mu_line.set_data(time[:k+1], mu_norm[:k+1])
        sigma_line.set_data(time[:k+1], sigma_norm[:k+1])
        return mu_line, sigma_line

    anim = FuncAnimation(
        fig,
        update,
        frames=len(time),
        init_func=init,
        blit=True,
        interval=int(1000/fps)
    )

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

    # Process noise covariance R for state [x, y, theta]
    R_motion = np.diag([
        0.005**2,
        0.005**2,
        np.deg2rad(0.2)**2
    ])

    # Measurement noise covariance Q for [range, bearing]
    Q_meas = np.diag([
        0.02**2,
        np.deg2rad(0.5)**2
    ])

    q0 = np.array([0.0, 0.0, 0.0])

    landmarks = np.array([
        [2.0, 3.0],
        [-4.5, 1.5]
    ])

    q_true, z_hist, mu_hist, Sigma_hist, meas_xy, used_lm = simulate_noisy_motion_and_ekf(
        q0=q0, N=N, dt=dt,
        v_cmd=v_cmd, w_cmd=w_cmd,
        R_motion=R_motion,
        Q_meas=Q_meas,
        landmarks=landmarks,
        used_landmark_idx=0,
        seed=7
    )
# Second gif: noiseless ground-truth only
    q_gt_noiseless = simulate_ground_truth_noiseless(
        q0=q0,
        N=N,
        dt=dt,
        v_cmd=v_cmd,
        w_cmd=w_cmd
    )

    animate_noisy_vs_ekf(
        q_true=q_true,
        q_gt_noiseless=q_gt_noiseless,
        mu_hist=mu_hist,
        meas_xy=meas_xy,
        landmarks=landmarks,
        used_landmark=used_lm,
        title="Simulation 5: EKF + FOV visualization",
        out_gif="sim5_noisy_motion_ekf_fov.gif",
        fps=25,
        fov_deg=90.0,
        max_range=5.0
    )

    

    animate_ground_truth_only(
        q_gt=q_gt_noiseless,
        title="Ground truth only (noiseless motion)",
        out_gif="ground_truth_noiseless.gif",
        fps=25
    )

    time = np.arange(N) * dt

    animate_mu_sigma(
        time=time,
        mu_hist=mu_hist,
        Sigma_hist=Sigma_hist,
        out_gif="mu_sigma_evolution.gif",
        fps=25
    )

    #plot_mu_and_sigma(time, mu_hist, Sigma_hist)