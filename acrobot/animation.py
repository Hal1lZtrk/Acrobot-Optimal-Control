"""Task 5: animation of the robot executing the LQR-tracked swing-up.

* ``export_video``  -> GIF / MP4 file (matplotlib; works headless)
* ``interactive``   -> pygame window with play / pause / step / restart keys
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .params import DEFAULT, DT


def tip_positions(x, p=DEFAULT):
    """Elbow and tip in a frame with y pointing up (theta1 = 0 hangs down, theta1 = pi is upright)."""
    q1, q2 = x[0], x[1]
    e = np.array([p.l1 * np.sin(q1), -p.l1 * np.cos(q1)])
    t = e + np.array([p.l2 * np.sin(q1 + q2), -p.l2 * np.cos(q1 + q2)])
    return e, t


def export_video(X_real, U_real, X_ref, path, stride=4, fps=25, p=DEFAULT, label="closed loop (LQR)"):
    reach = p.l1 + p.l2 + 0.4
    T = X_real.shape[1]
    frames = list(range(0, T, stride)) + [T - 1]
    t_axis = np.arange(T) * DT

    fig = plt.figure(figsize=(8, 4.5), dpi=90)
    ax = fig.add_axes([0.02, 0.05, 0.50, 0.88])
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-0.5, 0.5], [0, 0], color="k", lw=3)
    ax.add_patch(plt.Circle((0, 0), reach - 0.4 + 0.0, fill=False, ls=":", color="0.8"))
    ref_line, = ax.plot([], [], "-o", color="0.65", lw=7, ms=7, solid_capstyle="round", label="optimal reference")
    real_line, = ax.plot([], [], "-o", color="#1f5fbf", lw=4, ms=6, solid_capstyle="round", label=label)
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    txt = ax.text(0.0, reach - 0.1, "", ha="center", va="top", fontsize=11)

    axu = fig.add_axes([0.58, 0.58, 0.39, 0.33])
    axu.plot(t_axis, U_real[0], color="#1f5fbf", lw=1.2)
    axu.set_ylabel(r"$\tau$ [Nm]")
    axu.set_xlim(0, t_axis[-1])
    axu.grid(alpha=0.3)
    mu, = axu.plot([], [], "o", color="#d1495b")
    axq = fig.add_axes([0.58, 0.14, 0.39, 0.33])
    axq.plot(t_axis, X_ref[0], color="0.6", lw=2)
    axq.plot(t_axis, X_real[0], color="#1f5fbf", lw=1.2)
    axq.set_ylabel(r"$\theta_1$ [rad]")
    axq.set_xlabel("time [s]")
    axq.set_xlim(0, t_axis[-1])
    axq.grid(alpha=0.3)
    mq, = axq.plot([], [], "o", color="#d1495b")
    for a in (axu, axq):
        a.spines[["top", "right"]].set_visible(False)

    def draw(i):
        for line, X in ((ref_line, X_ref), (real_line, X_real)):
            e, t = tip_positions(X[:, i], p)
            line.set_data([0, e[0], t[0]], [0, e[1], t[1]])
        txt.set_text(f"t = {i * DT:5.2f} s")
        mu.set_data([t_axis[i]], [U_real[0, i]])
        mq.set_data([t_axis[i]], [X_real[0, i]])
        return ref_line, real_line, txt, mu, mq

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".gif"):
        anim.save(path, writer="pillow", fps=fps)
    else:
        anim.save(path, writer="ffmpeg", fps=fps, dpi=120)
    plt.close(fig)
    return path


def interactive(X_real, X_ref=None, p=DEFAULT, fps=50):
    """pygame viewer.  Keys: SPACE pause/play, LEFT/RIGHT step 0.1 s, R restart, ESC quit."""
    import pygame

    pygame.init()
    W, H, S = 900, 700, 80
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Acrobot swing-up (Set 3) - SPACE pause, arrows step, R restart")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    cx, cy = W // 2, H // 2 + 20
    T = X_real.shape[1]
    i, paused, running = 0, False, True
    jump = int(0.1 / DT)

    def draw(X, color, width):
        e, t = tip_positions(X, p)
        pts = [(cx, cy), (cx + e[0] * S, cy - e[1] * S), (cx + t[0] * S, cy - t[1] * S)]
        pygame.draw.lines(screen, color, False, pts, width)
        for q in pts:
            pygame.draw.circle(screen, color, (int(q[0]), int(q[1])), width + 2)

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    i = 0
                elif ev.key == pygame.K_RIGHT:
                    i = min(T - 1, i + jump)
                elif ev.key == pygame.K_LEFT:
                    i = max(0, i - jump)
        screen.fill((250, 250, 250))
        pygame.draw.line(screen, (0, 0, 0), (cx - 40, cy), (cx + 40, cy), 4)
        if X_ref is not None:
            draw(X_ref[:, i], (170, 170, 170), 9)
        draw(X_real[:, i], (31, 95, 191), 5)
        screen.blit(font.render(f"t = {i * DT:5.2f} s", True, (0, 0, 0)), (20, 20))
        screen.blit(font.render("grey: optimal reference   blue: closed loop", True, (60, 60, 60)), (20, 48))
        pygame.display.flip()
        if not paused:
            i = min(T - 1, i + 2)
        clock.tick(fps)
    pygame.quit()
