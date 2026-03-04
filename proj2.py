import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

A = np.array([[0.0, 1.0],
                [0.0, 0.0]])
B = np.array([[0.0],
                [1.0]])

#C = np.eye(2)  # z = C x  -> measures full state [q, qdot]
C = np.array([[1.0, 0.0]])  # measure position only
# 3) Motion step: x_{k+1} = A x_k + B u + e,  e~N(0,R)
def motion_step(x, u, A, B, R, rng, dt):
    # continuous model: x_dot = A x + B u
    x_dot = A @ x + (B.flatten() * u)
    mean = x + x_dot * dt               # Euler integration

    e = rng.multivariate_normal(mean=np.zeros(x.shape[0]), cov=R)
    return mean + e

def measurement_step(x, C, Q, rng):
    z_mean = C @ x                
    v = rng.multivariate_normal(mean=np.zeros(1), cov=Q) 
    return z_mean + v             

def kalman_filter_step(mu_prev, Sigma_prev, u, z, A, B, C, R, Q, dt):
    # 2: mu_bar = A mu_prev + B u
    #mu_bar = A @ mu_prev + (B.flatten() * u)

    # 3: Sigma_bar = A Sigma_prev A^T + R
    #Sigma_bar = A @ Sigma_prev @ A.T + R
    # Euler-discretized prediction
    A_d = np.eye(A.shape[0]) + A * dt
    B_d = B * dt

    mu_bar = A_d @ mu_prev + (B_d.flatten() * u)
    Sigma_bar = A_d @ Sigma_prev @ A_d.T + R
    # 4: K = Sigma_bar C^T (C Sigma_bar C^T + Q)^(-1)
    S = C @ Sigma_bar @ C.T + Q
    K = Sigma_bar @ C.T @ np.linalg.inv(S)

    # 5: mu = mu_bar + K (z - C mu_bar)
    mu = mu_bar + K @ (z - C @ mu_bar)

    # 6: Sigma = (I - K C) Sigma_bar
    I = np.eye(A.shape[0])
    Sigma = (I - K @ C) @ Sigma_bar

    return mu, Sigma, mu_bar, Sigma_bar, K

def simulate_sequences(u=0.2, seed=0,dt=0.02):
    T = 10.0
    N = int(T / dt)
    rng = np.random.default_rng(seed)

    # Noise covariances
    # Motion noise covariance R (state noise)
    # Measurement noise covariance Q (measurement noise)
    R = np.diag([0.01**2, 0.02**2]) # το πρωτο ειναι θορυβος στη θεση, το δευτερο στη ταχυτητα
    Q = np.array([[0.3**2]])  # measurement noise variance for position

    # Initial true state x0 = [q, qdot]
    x0 = np.array([0.0, 1.0])

    # True clean motion (R=0)
    x_clean = np.zeros((N, 2))
    x = x0.copy()
    for k in range(N):
        x_dot = A @ x + (B.flatten() * u)
        x = x + x_dot * dt        
        x_clean[k] = x

    # True noisy motion (process noise R)
    x_noisy = np.zeros((N, 2))
    x = x0.copy()
    for k in range(N):
        x = motion_step(x, u, A, B, R, rng, dt)        
        x_noisy[k] = x

    # Measurements (with noise Q) from clean motion
    # (you could also measure x_noisy; εδώ κρατάω το clean->noisy-meas για να ξεχωρίζει καθαρά)
    z_noisy = np.zeros((N, 1))
    for k in range(N):
        z_noisy[k] = measurement_step(x_noisy[k], C, Q, rng)

    # Kalman on noisy measurements
    # Initialize KF mean using the first measurement z0
    z0 = z_noisy[0, 0]          # first noisy position measurement (scalar) for initial mu
    mu = np.array([z0, 0.0])    # mu0 = [q0_meas, v0_guess]
    Sigma = np.diag([2.0**2, 2.0**2])       # initial covariance

    mu_hist = np.zeros((N, 2))
    for k in range(N):
        mu, Sigma, *_ = kalman_filter_step(mu, Sigma, u, z_noisy[k], A, B, C, R, Q, dt)
        mu_hist[k] = mu

    return x_clean, x_noisy, z_noisy, mu_hist

def make_4panel_gif(filename="double_integrator_4panels.gif", u=0.2, seed=0, fps=20):
    x_clean, x_noisy, z_noisy, mu_hist = simulate_sequences(u=u, seed=seed)
  
    # Show only position q (first component) in each panel for clarity
    q_clean = x_clean[:, 0]
    q_noisy = x_noisy[:, 0]
    q_meas  = z_noisy[:, 0]
    q_kf    = mu_hist[:, 0]

    all_q = np.concatenate([q_clean, q_noisy, q_meas, q_kf])
    ymin, ymax = all_q.min() - 2.0, all_q.max() + 2.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    titles = [
        "1) True motion ",
        "2) True motion (with motion noise R)",
        "3) Measurements (z = Cx + v, measurement noise Q)",
        "4) Kalman estimate muinit=zinit"
    ]
    colors = ["green", "orange", "red", "purple"]

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_ylim(ymin, ymax)
        ax.grid(True)
        ax.set_ylabel("q (position)")

    N = len(x_noisy)   
    dt=0.02
    t_axis = np.arange(N) * dt
    axes[-1].set_xlabel("time (s)")

    lines = []
    dots = []

    for i, ax in enumerate(axes):
        (ln,) = ax.plot([], [], linewidth=2, color=colors[i])
        (pt,) = ax.plot([], [], marker="o", markersize=7, linestyle="None", color=colors[i])
        lines.append(ln)
        dots.append(pt)

    # Optional show noisy motion faintly to compare
    (ref_noisy,) = axes[3].plot(t_axis, q_noisy, linewidth=1.2, alpha=0.25, color="black", label="true noisy (ref)")
    axes[3].legend(loc="upper left")

    def init():
        for ln, pt in zip(lines, dots):
            ln.set_data([], [])
            pt.set_data([], [])
        return lines + dots + [ref_noisy]

    def update(k):
        series = [q_clean, q_noisy, q_meas, q_kf]
        artists = []
        for i in range(4):
            y = series[i]
            lines[i].set_data(t_axis[:k+1], y[:k+1])
            dots[i].set_data([t_axis[k]], [y[k]])
            artists.extend([lines[i], dots[i]])
        artists.append(ref_noisy)
        return artists

    anim = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=int(1000 / fps))
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF: {filename}")

if __name__ == "__main__":
    make_4panel_gif()