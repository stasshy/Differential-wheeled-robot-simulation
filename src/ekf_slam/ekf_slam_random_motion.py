import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

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

def velocity_motion_model(q, u, dt):
    x, y, th = q
    v, w = u
    if abs(w) < 1e-9:
        return np.array([x + v * np.cos(th) * dt,
                         y + v * np.sin(th) * dt,
                         wrap_angle(th)])
    return np.array([
        x - (v / w) * np.sin(th) + (v / w) * np.sin(th + w * dt),
        y + (v / w) * np.cos(th) - (v / w) * np.cos(th + w * dt),
        wrap_angle(th + w * dt)
    ])

def motion_jacobian_robot(q, u, dt):
    _, _, th = q
    v, w = u
    Gx = np.eye(3)
    if abs(w) < 1e-9:
        Gx[0, 2], Gx[1, 2] = -v * np.sin(th) * dt, v * np.cos(th) * dt
    else:
        Gx[0, 2] = -(v / w) * np.cos(th) + (v / w) * np.cos(th + w * dt)
        Gx[1, 2] = -(v / w) * np.sin(th) + (v / w) * np.sin(th + w * dt)
    return Gx

def is_landmark_in_fov(q, landmark, fov_deg=180.0, max_range=np.inf):
    x, y, th = q
    lx, ly = landmark
    dx, dy = lx - x, ly - y
    r = np.hypot(dx, dy)
    b = wrap_angle(np.arctan2(dy, dx) - th)
    half_fov = np.deg2rad(fov_deg / 2.0)
    return (-half_fov <= b <= half_fov) and (r <= max_range), r, b

def measure_range_bearing(q, landmark, Q_meas, rng):
    x, y, th = q
    lx, ly = landmark
    dx, dy = lx - x, ly - y
    z = np.array([np.hypot(dx, dy), wrap_angle(np.arctan2(dy, dx) - th)])
    z += rng.multivariate_normal(np.zeros(2), Q_meas)
    z[1] = wrap_angle(z[1])
    return z

def control_law(k, dt):
    t = k * dt
    if t < 8.0:
        return np.array([0.8, 0.25])
    if t < 14.0:
        return np.array([0.7, -0.18])
    if t < 22.0:
        return np.array([0.9, 0.10])
    if t < 30.0:
        return np.array([0.6, -0.28])
    return np.array([0.75, 0.05])

class EKFSLAM:
    def __init__(self, R_motion_filter, Q_meas_filter, dt, n_landmarks=2, mu0=None, Sigma0=None):
        self.dt = dt
        self.n_landmarks = n_landmarks
        self.n = 3 + 2 * n_landmarks
        self.mu = np.zeros((self.n, 1)) if mu0 is None else mu0.reshape(self.n, 1).copy()

        if Sigma0 is None:
            self.Sigma = np.zeros((self.n, self.n))
            self.Sigma[0, 0] = 0.1**2
            self.Sigma[1, 1] = 0.1**2
            self.Sigma[2, 2] = np.deg2rad(5.0)**2
            self.Sigma[3:, 3:] = 1e6 * np.eye(self.n - 3)
        else:
            self.Sigma = Sigma0.copy()

        self.R = R_motion_filter
        self.Q = Q_meas_filter
        self.observed = [False] * n_landmarks

    def landmark_base_index(self, j):
        return 3 + 2 * j

    def get_landmark_from_state(self, mu, j):
        b = self.landmark_base_index(j)
        return mu[b, 0], mu[b + 1, 0]

    def motion_model_full(self, mu, u):
        mu_new = mu.copy()
        mu_new[:3, 0] = velocity_motion_model(mu[:3, 0], u, self.dt)
        return mu_new

    def motion_jacobian_full(self, mu, u):
        G = np.eye(self.n)
        G[:3, :3] = motion_jacobian_robot(mu[:3, 0], u, self.dt)
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
        self.mu = self.motion_model_full(self.mu, u)
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])

        G = self.motion_jacobian_full(self.mu, u)
        R_full = np.zeros((self.n, self.n))
        R_full[:3, :3] = self.R
        self.Sigma = G @ self.Sigma @ G.T + R_full
        return self.mu, self.Sigma

    def correct_one_landmark(self, j, z):
        z = np.asarray(z).reshape(2)
        if not self.observed[j]:
            self.initialize_landmark(j, z)
            return self.mu, self.Sigma

        H = self.measurement_jacobian_j(self.mu, j)
        z_pred = self.measurement_model_j(self.mu, j)
        S = H @ self.Sigma @ H.T + self.Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        innovation = z.reshape(2, 1) - z_pred
        innovation[1, 0] = wrap_angle(innovation[1, 0])

        self.mu = self.mu + K @ innovation
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])

        I = np.eye(self.n)
        self.Sigma = (I - K @ H) @ self.Sigma @ (I - K @ H).T + K @ self.Q @ K.T
        return self.mu, self.Sigma

    def step(self, u, measurements):
        self.predict(u)
        for j, z in measurements:
            self.correct_one_landmark(j, z)
        return self.mu, self.Sigma

def simulate_ground_truth_noiseless(q0, N, dt):
    q_hist = np.zeros((N, 3))
    q = q0.copy()
    for k in range(N):
        q = velocity_motion_model(q, control_law(k, dt), dt)
        q_hist[k] = q
    return q_hist

def simulate_slam_step(q, u, landmarks, R_motion_true, Q_meas_true, rng, dt, fov_deg=180.0, max_range=7.0):
    q_new = velocity_motion_model(q, u, dt) + rng.multivariate_normal(np.zeros(3), R_motion_true)
    q_new[2] = wrap_angle(q_new[2])

    measurements = []
    for j, lm in enumerate(landmarks):
        ok, _, _ = is_landmark_in_fov(q_new, lm, fov_deg=fov_deg, max_range=max_range)
        if ok:
            measurements.append((j, measure_range_bearing(q_new, lm, Q_meas_true, rng)))
    return q_new, measurements

def simulate_noisy_motion_and_ekf_slam(
    q0, N, dt, R_motion_true, Q_meas_true,
    R_motion_filter, Q_meas_filter, landmarks,
    fov_deg=180.0, max_range=7.0, seed=7
):
    rng = np.random.default_rng(seed)
    q = q0.copy()
    n_state = 3 + 2 * len(landmarks)

    q_true = np.zeros((N, 3))
    mu_hist = np.zeros((N, n_state))
    Sigma_hist = np.zeros((N, n_state, n_state))
    observed_hist = np.zeros((N, len(landmarks)), dtype=bool)

    mu0 = np.zeros(n_state)
    mu0[:3] = q0
    Sigma0 = np.zeros((n_state, n_state))
    Sigma0[0, 0] = 0.1**2
    Sigma0[1, 1] = 0.1**2
    Sigma0[2, 2] = np.deg2rad(5.0)**2
    Sigma0[3:, 3:] = 1e6 * np.eye(n_state - 3)

    ekf = EKFSLAM(R_motion_filter, Q_meas_filter, dt, len(landmarks), mu0, Sigma0)

    for k in range(N):
        u = control_law(k, dt)
        q, measurements = simulate_slam_step(
            q, u, landmarks, R_motion_true, Q_meas_true, rng, dt,
            fov_deg=fov_deg, max_range=max_range
        )
        q_true[k] = q
        ekf.step(u, measurements)
        mu_hist[k] = ekf.mu.flatten()
        Sigma_hist[k] = ekf.Sigma
        observed_hist[k] = np.array(ekf.observed, dtype=bool)

    return q_true, mu_hist, Sigma_hist, observed_hist


def compute_fov_measurements_for_animation(q_true, landmarks, fov_deg=180.0, max_range=7.0):
    visible_rays = []
    for q in q_true:
        x, y, _ = q
        rays = []
        for i, lm in enumerate(landmarks):
            ok, r, b = is_landmark_in_fov(q, lm, fov_deg=fov_deg, max_range=max_range)
            if ok:
                mx, my = rb_to_xy(q, r, b)
                rays.append((i, x, y, mx, my))
        visible_rays.append(rays)
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
    title, out_gif, fps=25, fov_deg=180.0, max_range=7.0
):
    N = q_true.shape[0]
    n_landmarks = landmarks.shape[0]
    est_landmarks_hist = extract_estimated_landmarks(mu_hist, n_landmarks)

    xmin, xmax, ymin, ymax = compute_bounds(
        [q_true, mu_hist[:, :3], q_gt_noiseless],
        landmarks=landmarks,
        est_landmarks_hist=est_landmarks_hist,
        pad=1.0
    )

    visible_rays = compute_fov_measurements_for_animation(q_true, landmarks, fov_deg, max_range)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker="s", color="black", s=70, label="true landmarks")

    (true_ln,) = ax.plot([], [], lw=2.2, color="black", label="true (noisy motion)")
    (ekf_ln,) = ax.plot([], [], lw=2.2, ls="--", color="purple", label="EKF-SLAM estimate")
    (gt_ln,) = ax.plot([], [], lw=2.0, ls="--", color="lightgray", label="ground truth (noiseless)")

    (true_pt,) = ax.plot([], [], "o", ms=6, color="black")
    (ekf_pt,) = ax.plot([], [], "o", ms=6, color="purple")

    (fov_left_ln,) = ax.plot([], [], lw=1.2, ls="-.", color="green", label="FOV")
    (fov_right_ln,) = ax.plot([], [], lw=1.2, ls="-.", color="green")

    (meas_ray_1,) = ax.plot([], [], lw=1.0, color="red", alpha=0.9, label="visible ray")
    (meas_pt_1,) = ax.plot([], [], "x", ms=7, color="red", alpha=0.9)
    (meas_ray_2,) = ax.plot([], [], lw=1.0, color="orange", alpha=0.9, label="2nd visible ray")
    (meas_pt_2,) = ax.plot([], [], "x", ms=7, color="orange", alpha=0.9)

    est_colors = ["blue", "magenta"]
    est_lm_pts, est_lm_texts = [], []
    for j in range(n_landmarks):
        (pt,) = ax.plot([], [], "D", ms=7, color=est_colors[j % len(est_colors)], label=f"estimated LM{j+1}")
        txt = ax.text(0.0, 0.0, "", fontsize=9, color=est_colors[j % len(est_colors)])
        est_lm_pts.append(pt)
        est_lm_texts.append(txt)

    half_fov = np.deg2rad(fov_deg / 2.0)
    ax.legend(loc="upper right")

    def init():
        for obj in [true_ln, ekf_ln, gt_ln, true_pt, ekf_pt,
                    fov_left_ln, fov_right_ln, meas_ray_1, meas_pt_1, meas_ray_2, meas_pt_2]:
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
        for obj in [meas_ray_1, meas_pt_1, meas_ray_2, meas_pt_2]:
            obj.set_data([], [])

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

    q0 = np.array([0.0, 0.0, 0.0])

    landmarks = np.array([
        [4.0, 3.0],
        [-3.5, 2.5]
    ])

    R_motion_true = np.diag([
        0.01**2,
        0.01**2,
        np.deg2rad(0.4)**2
    ])

    Q_meas_true = np.diag([
        0.03**2,
        np.deg2rad(1.0)**2
    ])

    R_motion_filter = np.diag([
        0.015**2,
        0.015**2,
        np.deg2rad(0.5)**2
    ])

    Q_meas_filter = np.diag([
        0.025**2,
        np.deg2rad(0.8)**2
    ])

    fov_deg = 180.0
    max_range = 7.0

    q_true, mu_hist, Sigma_hist, observed_hist = simulate_noisy_motion_and_ekf_slam(
        q0=q0,
        N=N,
        dt=dt,
        R_motion_true=R_motion_true,
        Q_meas_true=Q_meas_true,
        R_motion_filter=R_motion_filter,
        Q_meas_filter=Q_meas_filter,
        landmarks=landmarks,
        fov_deg=fov_deg,
        max_range=max_range,
        seed=7
    )

    q_gt_noiseless = simulate_ground_truth_noiseless(
        q0=q0,
        N=N,
        dt=dt
    )

    animate_ekf_slam(
        q_true=q_true,
        mu_hist=mu_hist,
        q_gt_noiseless=q_gt_noiseless,
        landmarks=landmarks,
        observed_hist=observed_hist,
        title="EKF-SLAM with slide motion model",
        out_gif="ekf_slam_slide_model.gif",
        fps=25,
        fov_deg=fov_deg,
        max_range=max_range
    )