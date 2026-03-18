import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def step_unicycle(q, v, w, dt):
    x, y, th = q
    return np.array([x + v * np.cos(th) * dt,
                     y + v * np.sin(th) * dt,
                     wrap_angle(th + w * dt)])

def compute_bounds(trajs, landmarks=None, est_landmarks_hist=None, pad=1.0):
    xs = np.concatenate([tr[:, 0] for tr in trajs])
    ys = np.concatenate([tr[:, 1] for tr in trajs])

    if landmarks is not None:
        xs = np.concatenate([xs, landmarks[:, 0]])
        ys = np.concatenate([ys, landmarks[:, 1]])

    if est_landmarks_hist is not None:
        xs = np.concatenate([xs, est_landmarks_hist[:, :, 0].ravel()])
        ys = np.concatenate([ys, est_landmarks_hist[:, :, 1].ravel()])

    return xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad

def rb_to_xy(q, r, b):
    x, y, th = q
    ang = th + b
    return x + r * np.cos(ang), y + r * np.sin(ang)

def simulate_ground_truth_noiseless(q0, N, dt, v_cmd, w_cmd):
    q_hist = np.zeros((N, 3))
    q = q0.copy()
    for k in range(N):
        q = step_unicycle(q, v_cmd, w_cmd, dt)
        q_hist[k] = q
    return q_hist

def is_landmark_in_fov(q, landmark, fov_deg=180.0, max_range=np.inf):
    x, y, th = q
    lx, ly = landmark
    dx, dy = lx - x, ly - y
    r = np.hypot(dx, dy)
    b = wrap_angle(np.arctan2(dy, dx) - th)
    half_fov = np.deg2rad(fov_deg / 2.0)
    return (-half_fov <= b <= half_fov) and (r <= max_range), r, b

class EKFSLAM:
    def __init__(self, Q_motion, R_meas, dt, n_landmarks=2, mu0=None, Sigma0=None):
        self.dt = dt
        self.n_landmarks = n_landmarks
        self.n = 3 + 2 * n_landmarks
        self.mu = np.zeros((self.n, 1)) if mu0 is None else mu0.reshape(self.n, 1).copy()

        if Sigma0 is None:
            self.Sigma = np.eye(self.n)
            self.Sigma[:3, :3] *= 0.5
            self.Sigma[3:, 3:] *= 1e6
        else:
            self.Sigma = Sigma0.copy()

        self.R = Q_motion
        self.Q = R_meas
        self.observed = [False] * n_landmarks

    def landmark_base_index(self, j):
        return 3 + 2 * j

    def get_landmark_from_state(self, mu, j):
        b = self.landmark_base_index(j)
        return mu[b, 0], mu[b + 1, 0]

    def motion_model(self, mu, u):
        v, omega = u
        x, y, th = mu[0, 0], mu[1, 0], mu[2, 0]
        mu_new = mu.copy()
        mu_new[0, 0] = x + v * np.cos(th) * self.dt
        mu_new[1, 0] = y + v * np.sin(th) * self.dt
        mu_new[2, 0] = wrap_angle(th + omega * self.dt)
        return mu_new

    def motion_jacobian(self, mu, u):
        v, _ = u
        th = mu[2, 0]
        G = np.eye(self.n)
        G[0, 2] = -v * np.sin(th) * self.dt
        G[1, 2] =  v * np.cos(th) * self.dt
        return G

    def measurement_model_j(self, mu, j):
        x, y, th = mu[0, 0], mu[1, 0], mu[2, 0]
        lx, ly = self.get_landmark_from_state(mu, j)
        dx, dy = lx - x, ly - y
        return np.array([[np.hypot(dx, dy)], [wrap_angle(np.arctan2(dy, dx) - th)]])

    def measurement_jacobian_j(self, mu, j):
        x, y = mu[0, 0], mu[1, 0]
        lx, ly = self.get_landmark_from_state(mu, j)
        dx, dy = lx - x, ly - y
        q = max(dx**2 + dy**2, 1e-9)
        r = max(np.sqrt(q), 1e-9)

        H = np.zeros((2, self.n))
        H[0, 0], H[0, 1] = -dx / r, -dy / r
        H[1, 0], H[1, 1], H[1, 2] = dy / q, -dx / q, -1.0

        b = self.landmark_base_index(j)
        H[0, b], H[0, b + 1] = dx / r, dy / r
        H[1, b], H[1, b + 1] = -dy / q, dx / q
        return H

    def initialize_landmark(self, j, z):
        r, b_meas = z
        x, y, th = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]
        b = self.landmark_base_index(j)
        self.mu[b, 0] = x + r * np.cos(th + b_meas)
        self.mu[b + 1, 0] = y + r * np.sin(th + b_meas)
        self.observed[j] = True

    def predict(self, u):
        self.mu = self.motion_model(self.mu, u)
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])

        G = self.motion_jacobian(self.mu, u)
        R_full = np.zeros((self.n, self.n))
        R_full[:3, :3] = self.R
        self.Sigma = G @ self.Sigma @ G.T + R_full
        return self.mu, self.Sigma, G

    def correct_one_landmark(self, j, z):
        z = np.asarray(z).reshape(2)

        if not self.observed[j]:
            self.initialize_landmark(j, z)

        H = self.measurement_jacobian_j(self.mu, j)
        z_pred = self.measurement_model_j(self.mu, j)

        S = H @ self.Sigma @ H.T + self.Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        innovation = z.reshape(2, 1) - z_pred
        innovation[1, 0] = wrap_angle(innovation[1, 0])

        self.mu = self.mu + K @ innovation
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])

        I = np.eye(self.n)
        self.Sigma = (I - K @ H) @ self.Sigma
        return self.mu, self.Sigma, H, K, innovation

    def step(self, u, measurements):
        self.predict(u)
        debug_info = []
        for j, z in measurements:
            mu, Sigma, H, K, innovation = self.correct_one_landmark(j, z)
            debug_info.append({
                "landmark_id": j,
                "H": H,
                "K": K,
                "innovation": innovation.copy(),
                "mu": mu.copy(),
                "Sigma": Sigma.copy(),
            })
        return self.mu, self.Sigma, debug_info

def simulate_slam_step(q, u, landmarks, R_motion, Q_meas, rng, dt, fov_deg=180.0, max_range=5.0):
    v_cmd, w_cmd = u
    q_mean = step_unicycle(q, v_cmd, w_cmd, dt)
    q_new = q_mean + rng.multivariate_normal(np.zeros(3), R_motion)
    q_new[2] = wrap_angle(q_new[2])

    measurements = []
    for j, lm in enumerate(landmarks):
        ok, r_true, b_true = is_landmark_in_fov(q_new, lm, fov_deg=fov_deg, max_range=max_range)
        if ok:
            z = np.array([r_true, b_true]) + rng.multivariate_normal(np.zeros(2), Q_meas)
            z[1] = wrap_angle(z[1])
            measurements.append((j, z))

    return q_new, measurements

def simulate_noisy_motion_and_ekf_slam(
    q0, N, dt, v_cmd, w_cmd, R_motion, Q_meas, landmarks,
    fov_deg=180.0, max_range=5.0, seed=7
):
    rng = np.random.default_rng(seed)
    u = (v_cmd, w_cmd)
    q = q0.copy()

    n_landmarks = len(landmarks)
    n_state = 3 + 2 * n_landmarks

    q_true = np.zeros((N, 3))
    mu_hist = np.zeros((N, n_state))
    Sigma_hist = np.zeros((N, n_state, n_state))
    meas_hist = []
    observed_hist = np.zeros((N, n_landmarks), dtype=bool)

    mu0 = np.zeros(n_state)
    mu0[:3] = q0

    Sigma0 = np.zeros((n_state, n_state))
    Sigma0[0, 0] = 0.5**2
    Sigma0[1, 1] = 0.5**2
    Sigma0[2, 2] = np.deg2rad(20)**2
    Sigma0[3:, 3:] = 1e6 * np.eye(n_state - 3)

    ekf = EKFSLAM(Q_motion=R_motion, R_meas=Q_meas, dt=dt,
                  n_landmarks=n_landmarks, mu0=mu0, Sigma0=Sigma0)

    for k in range(N):
        q, measurements = simulate_slam_step(
            q, u, landmarks, R_motion, Q_meas, rng, dt,
            fov_deg=fov_deg, max_range=max_range
        )
        q_true[k] = q
        meas_hist.append(measurements)

        ekf.step(u, measurements)

        mu_hist[k] = ekf.mu.flatten()
        Sigma_hist[k] = ekf.Sigma
        observed_hist[k] = np.array(ekf.observed, dtype=bool)

    return q_true, mu_hist, Sigma_hist, meas_hist, observed_hist

def compute_fov_measurements_for_animation(q_true, landmarks, fov_deg=180.0, max_range=np.inf):
    visible_rays = []
    for q in q_true:
        rays_k = []
        x, y, _ = q
        for i, lm in enumerate(landmarks):
            ok, r, b = is_landmark_in_fov(q, lm, fov_deg=fov_deg, max_range=max_range)
            if ok:
                mx, my = rb_to_xy(q, r, b)
                rays_k.append((i, x, y, mx, my))
        visible_rays.append(rays_k)
    return visible_rays

def extract_estimated_landmarks(mu_hist, n_landmarks):
    est = np.zeros((mu_hist.shape[0], n_landmarks, 2))
    for j in range(n_landmarks):
        b = 3 + 2 * j
        est[:, j, 0] = mu_hist[:, b]
        est[:, j, 1] = mu_hist[:, b + 1]
    return est

def animate_ekf_slam(
    q_true, mu_hist, q_gt_noiseless, landmarks, observed_hist,
    title, out_gif, fps=25, fov_deg=180.0, max_range=5.0
):
    N = q_true.shape[0]
    n_landmarks = landmarks.shape[0]
    est_landmarks_hist = extract_estimated_landmarks(mu_hist, n_landmarks)

    xmin, xmax, ymin, ymax = compute_bounds(
        [q_true, mu_hist[:, :3]],
        landmarks=landmarks,
        est_landmarks_hist=est_landmarks_hist,
        pad=1.0
    )

    visible_rays = compute_fov_measurements_for_animation(
        q_true, landmarks, fov_deg=fov_deg, max_range=max_range
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker="s", color="black", s=70, label="true landmarks")

    (true_ln,) = ax.plot([], [], linewidth=2.2, color="black", label="true (noisy motion)")
    (ekf_ln,) = ax.plot([], [], linewidth=2.2, linestyle="--", color="purple", label="EKF-SLAM estimate")
    (gt_ln,) = ax.plot([], [], linewidth=2.0, color="lightgray", linestyle="--", label="ground truth (noiseless)")

    (true_pt,) = ax.plot([], [], marker="o", markersize=6, linestyle="None", color="black")
    (ekf_pt,) = ax.plot([], [], marker="o", markersize=6, linestyle="None", color="purple")

    (fov_left_ln,) = ax.plot([], [], linewidth=1.2, linestyle="-.", color="green", label="FOV")
    (fov_right_ln,) = ax.plot([], [], linewidth=1.2, linestyle="-.", color="green")

    (meas_ray_1,) = ax.plot([], [], linewidth=1.0, color="red", alpha=0.9, label="visible ray")
    (meas_pt_1,) = ax.plot([], [], marker="x", markersize=7, linestyle="None", color="red", alpha=0.9)

    (meas_ray_2,) = ax.plot([], [], linewidth=1.0, color="orange", alpha=0.9, label="2nd visible ray")
    (meas_pt_2,) = ax.plot([], [], marker="x", markersize=7, linestyle="None", color="orange", alpha=0.9)

    est_lm_pts, est_lm_texts = [], []
    est_colors = ["blue", "magenta"]

    for j in range(n_landmarks):
        (pt,) = ax.plot([], [], marker="D", markersize=7, linestyle="None",
                        color=est_colors[j % len(est_colors)], label=f"estimated LM{j+1}")
        txt = ax.text(0.0, 0.0, "", fontsize=9, color=est_colors[j % len(est_colors)])
        est_lm_pts.append(pt)
        est_lm_texts.append(txt)

    half_fov = np.deg2rad(fov_deg / 2.0)
    ax.legend(loc="upper right")

    def init():
        for obj in [true_ln, ekf_ln, gt_ln, true_pt, ekf_pt,
                    fov_left_ln, fov_right_ln,
                    meas_ray_1, meas_pt_1, meas_ray_2, meas_pt_2]:
            obj.set_data([], [])

        for pt, txt in zip(est_lm_pts, est_lm_texts):
            pt.set_data([], [])
            txt.set_position((0.0, 0.0))
            txt.set_text("")

        return [true_ln, ekf_ln, gt_ln, true_pt, ekf_pt,
                fov_left_ln, fov_right_ln, meas_ray_1, meas_pt_1,
                meas_ray_2, meas_pt_2, *est_lm_pts, *est_lm_texts]

    def update(k):
        gt_ln.set_data(q_gt_noiseless[:k + 1, 0], q_gt_noiseless[:k + 1, 1])
        true_ln.set_data(q_true[:k + 1, 0], q_true[:k + 1, 1])
        ekf_ln.set_data(mu_hist[:k + 1, 0], mu_hist[:k + 1, 1])

        tx, ty, tth = q_true[k]
        ex, ey = mu_hist[k, 0], mu_hist[k, 1]
        true_pt.set_data([tx], [ty])
        ekf_pt.set_data([ex], [ey])

        left_ang, right_ang = tth + half_fov, tth - half_fov
        fov_left_ln.set_data([tx, tx + max_range * np.cos(left_ang)],
                             [ty, ty + max_range * np.sin(left_ang)])
        fov_right_ln.set_data([tx, tx + max_range * np.cos(right_ang)],
                              [ty, ty + max_range * np.sin(right_ang)])

        rays = visible_rays[k]
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

        for j in range(n_landmarks):
            if observed_hist[k, j]:
                lx, ly = mu_hist[k, 3 + 2 * j], mu_hist[k, 3 + 2 * j + 1]
                est_lm_pts[j].set_data([lx], [ly])
                est_lm_texts[j].set_position((lx + 0.1, ly + 0.1))
                est_lm_texts[j].set_text(f"LM{j+1}")
            else:
                est_lm_pts[j].set_data([], [])
                est_lm_texts[j].set_text("")

        return [true_ln, ekf_ln, gt_ln, true_pt, ekf_pt,
                fov_left_ln, fov_right_ln, meas_ray_1, meas_pt_1,
                meas_ray_2, meas_pt_2, *est_lm_pts, *est_lm_texts]

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

    R_motion = np.diag([
        0.003**2,
        0.003**2,
        np.deg2rad(0.1)**2
    ])

    Q_meas = np.diag([
        0.01**2,
        np.deg2rad(0.25)**2
    ])

    q0 = np.array([0.0, 0.0, 0.0])

    landmarks = np.array([
        [2.0, 3.0],
        [-4.5, 1.5]
    ])

    fov_deg = 180.0
    max_range = 7.0

    q_true, mu_hist, Sigma_hist, meas_hist, observed_hist = simulate_noisy_motion_and_ekf_slam(
        q0=q0, N=N, dt=dt, v_cmd=v_cmd, w_cmd=w_cmd,
        R_motion=R_motion, Q_meas=Q_meas, landmarks=landmarks,
        fov_deg=fov_deg, max_range=max_range, seed=7
    )

    q_gt_noiseless = simulate_ground_truth_noiseless(
        q0=q0, N=N, dt=dt, v_cmd=v_cmd, w_cmd=w_cmd
    )

    animate_ekf_slam(
        q_true=q_true,
        mu_hist=mu_hist,
        q_gt_noiseless=q_gt_noiseless,
        landmarks=landmarks,
        observed_hist=observed_hist,
        title="EKF-SLAM with 2 landmarks and FOV",
        out_gif="ekf_slam_2_landmarks.gif",
        fps=25,
        fov_deg=fov_deg,
        max_range=max_range
    )