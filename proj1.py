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

def lidar_detect(q, landmarks, r_max=8.0, fov=np.deg2rad(180)):
    x, y, th = q
    dets = []
    for i, (lx, ly) in enumerate(landmarks):
        dx, dy = lx - x, ly - y
        r = np.hypot(dx, dy)
        if r > r_max:
            continue
        bearing = wrap_angle(np.arctan2(dy, dx) - th)
        if abs(bearing) <= fov / 2:
            dets.append(i)
    return dets

def simulate_robots(robots_init, N, dt, v_cmd, w_cmd, noisy=False, sigma_v=0.0, sigma_w=0.0, seed=7):
    rng = np.random.default_rng(seed)
    trajs = []
    for q0 in robots_init:
        q = q0.copy()
        traj = np.zeros((N, 3))
        for k in range(N):
            if noisy:
                v = v_cmd + rng.normal(0.0, sigma_v)
                w = w_cmd + rng.normal(0.0, sigma_w)
            else:
                v = v_cmd
                w = w_cmd
            q = step_unicycle(q, v, w, dt)
            traj[k] = q
        trajs.append(traj)
    return trajs

def compute_bounds(trajs, landmarks=None, pad=1.0):
    xs, ys = [], []
    for tr in trajs:
        xs.append(tr[:, 0])
        ys.append(tr[:, 1])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    if landmarks is not None:
        xs = np.concatenate([xs, landmarks[:, 0]])
        ys = np.concatenate([ys, landmarks[:, 1]])

    xmin, xmax = xs.min() - pad, xs.max() + pad
    ymin, ymax = ys.min() - pad, ys.max() + pad
    return xmin, xmax, ymin, ymax

dt = 0.05
T = 40.0
N = int(T / dt)

R = 3.0
v_cmd = 0.6
w_cmd = v_cmd / R

sigma_v = 0.25
sigma_w = 0.2

robots_init = [
    np.array([0.0, 0.0, 0.0]),
    np.array([2.0, 1.0, np.deg2rad(30)]),
    np.array([-1.5, 2.0, np.deg2rad(-60)]),
]

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

trajs_clean = simulate_robots(robots_init, N, dt, v_cmd, w_cmd, noisy=False)
trajs_noisy = simulate_robots(robots_init, N, dt, v_cmd, w_cmd, noisy=True, sigma_v=sigma_v, sigma_w=sigma_w, seed=7)

def animate_motion(trajs, title, out_gif, landmarks=None, show_lidar=False,
                   r_max=8.0, fov=np.deg2rad(180), fps=25):

    num_robots = len(trajs)
    N_local = trajs[0].shape[0]

    # Fixed robot colors
    robot_colors = ["red", "blue", "green"]

    xmin, xmax, ymin, ymax = compute_bounds(trajs, landmarks=landmarks, pad=1.0)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    # Landmarks
    if landmarks is not None:
        ax.scatter(landmarks[:, 0], landmarks[:, 1],
                   marker="s", color="black", label="landmarks")

    traj_lines = []
    robot_pts = []
    heading_lines = []

    # Create graphical objects per robot
    for i in range(num_robots):
        color = robot_colors[i]

        # Full trajectory line
        (traj_ln,) = ax.plot([], [], linewidth=2, color=color, label=f"robot {i}")
        traj_lines.append(traj_ln)

        # Robot body
        (pt,) = ax.plot([], [], marker="o", markersize=6,
                        linestyle="None", color=color)
        robot_pts.append(pt)

        # Heading line
        (hd,) = ax.plot([], [], linewidth=2.5, color=color)
        heading_lines.append(hd)

        # Start & End markers (same color)
        tr = trajs[i]
        x0, y0 = tr[0, 0], tr[0, 1]
        xf, yf = tr[-1, 0], tr[-1, 1]

        ax.plot(x0, y0, marker="o", markersize=10,
                markerfacecolor="none", markeredgewidth=2.5,
                markeredgecolor=color)

        ax.plot(xf, yf, marker="o", markersize=10,
                markerfacecolor="none", markeredgewidth=2.5,
                markeredgecolor=color)

    lidar_lines = []

    def clear_lidar_lines():
        nonlocal lidar_lines
        for ln in lidar_lines:
            ln.remove()
        lidar_lines = []

    def init():
        for i in range(num_robots):
            traj_lines[i].set_data([], [])
            robot_pts[i].set_data([], [])
            heading_lines[i].set_data([], [])
        clear_lidar_lines()
        return traj_lines + robot_pts + heading_lines

    def update(k):

        if show_lidar:
            clear_lidar_lines()

        artists = []

        for i, tr in enumerate(trajs):
            color = robot_colors[i]

            # FULL trajectory
            traj_lines[i].set_data(tr[:k+1, 0], tr[:k+1, 1])
            artists.append(traj_lines[i])

            x, y, th = tr[k]

            # Robot body
            robot_pts[i].set_data([x], [y])
            artists.append(robot_pts[i])

            # Heading
            hx = x + 0.4 * np.cos(th)
            hy = y + 0.4 * np.sin(th)
            heading_lines[i].set_data([x, hx], [y, hy])
            artists.append(heading_lines[i])

            # LiDAR rays
            if show_lidar and (landmarks is not None):
                det_idxs = lidar_detect(tr[k], landmarks,
                                        r_max=r_max, fov=fov)
                for idx in det_idxs:
                    lx, ly = landmarks[idx]
                    (ln,) = ax.plot([x, lx], [y, ly],
                                    linewidth=0.8, color=color)
                    lidar_lines.append(ln)
                    artists.append(ln)

        return artists

    ax.legend()

    anim = FuncAnimation(fig, update,
                         frames=N_local,
                         init_func=init,
                         blit=True,
                         interval=int(1000 / fps))

    writer = PillowWriter(fps=fps)
    anim.save(out_gif, writer=writer)
    plt.close(fig)

    print(f"Saved: {out_gif}")

animate_motion(
    trajs_clean,
    title="Simulation 1: Ground truth (no noise)",
    out_gif="sim1_clean.gif",
    landmarks=None,
    show_lidar=False,
    fps=25
)

animate_motion(
    trajs_noisy,
    title="Simulation 2: Motion with Gaussian noise",
    out_gif="sim2_noisy.gif",
    landmarks=None,
    show_lidar=False,
    fps=25
)

animate_motion(
    trajs_clean,
    title="Simulation 3: Ground truth + LiDAR detections to landmarks",
    out_gif="sim3_lidar.gif",
    landmarks=landmarks,
    show_lidar=True,
    r_max=8.0,
    fov=np.deg2rad(180),
    fps=25
)