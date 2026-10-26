"""Figures required by the assignment.

Every figure is saved as PNG in ``results/figures`` and, by default, also opened in a window
(call :func:`show` at the end of a script).  Pass ``--no-show`` on the command line or set the
environment variable ``ACROBOT_NO_SHOW=1`` for batch / headless runs (no display needed then).
"""
import os
import sys

import matplotlib

SHOW = not (os.environ.get("ACROBOT_NO_SHOW") or "--no-show" in sys.argv)
if SHOW and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    SHOW = False                              # no display available
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .newton import _closed_loop_rollout, total_cost
from .params import DT

DESIRED = dict(color="0.35", ls="--", lw=1.6)
OPT = dict(color="#1f5fbf", lw=2.0)
PALETTE = ["#d1495b", "#2a9d8f", "#e9a100", "#1f5fbf", "#7a4eab", "#555555"]
STATE_LABELS = [r"$\theta_1$ [rad]", r"$\theta_2$ [rad]", r"$\dot\theta_1$ [rad/s]", r"$\dot\theta_2$ [rad/s]"]
ERR_LABELS = [r"$e_{\theta_1}$ [rad]", r"$e_{\theta_2}$ [rad]", r"$e_{\dot\theta_1}$ [rad/s]", r"$e_{\dot\theta_2}$ [rad/s]"]

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "figure.dpi": 100})


def _finish(fig, path):
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=140)
    if not SHOW:
        plt.close(fig)


def show():
    """Open all figures created so far (blocks until the windows are closed)."""
    if SHOW and plt.get_fignums():
        plt.show()


def _time(X):
    return np.arange(X.shape[1]) * DT


def plot_trajectory(Xopt, Uopt, Xdes, Udes, path, title):
    """Optimal trajectory vs desired curve (states + input)."""
    t = _time(Xopt)
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    for i in range(4):
        ax[i].plot(t, Xdes[i], label="desired curve", **DESIRED)
        ax[i].plot(t, Xopt[i], label="optimal", **OPT)
        ax[i].set_ylabel(STATE_LABELS[i])
    ax[4].plot(t, Udes[0], **DESIRED)
    ax[4].plot(t, Uopt[0], **OPT)
    ax[4].set_ylabel(r"$\tau$ [Nm]")
    ax[4].set_xlabel("time [s]")
    ax[0].legend(loc="best", ncol=2)
    ax[0].set_title(title)
    _finish(fig, path)


def plot_intermediate(res, Xdes, iters, path, title):
    """Desired curve, a few intermediate iterates and the final trajectory (theta_1, theta_2, input)."""
    t = _time(Xdes)
    iters = [k for k in iters if k < len(res.x_hist) - 1]
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for j, (idx, lab) in enumerate([(0, STATE_LABELS[0]), (1, STATE_LABELS[1])]):
        ax[j].plot(t, Xdes[idx], label="desired curve", **DESIRED)
        for n, k in enumerate(iters):
            ax[j].plot(t, res.x_hist[k][idx], color=cmap(0.15 + 0.6 * n / max(1, len(iters) - 1)),
                       lw=1.2, label=f"iteration {k}")
        ax[j].plot(t, res.x_hist[-1][idx], label="final", **OPT)
        ax[j].set_ylabel(lab)
    for n, k in enumerate(iters):
        ax[2].plot(t, res.u_hist[k][0], color=cmap(0.15 + 0.6 * n / max(1, len(iters) - 1)), lw=1.0)
    ax[2].plot(t, res.u_hist[-1][0], **OPT)
    ax[2].set_ylabel(r"$\tau$ [Nm]")
    ax[2].set_xlabel("time [s]")
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_title(title)
    _finish(fig, path)


def plot_convergence(res, path, title):
    """Cost and norm of the descent direction along iterations (semilog)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(np.arange(len(res.J)), res.J, "o-", color=OPT["color"], ms=3)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("cost J")
    ax[0].set_title("Cost along iterations")
    ax[1].semilogy(np.arange(len(res.dnorm)), res.dnorm, "o-", color=PALETTE[0], ms=3)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel(r"$\|\Delta u\|$")
    ax[1].set_title("Norm of the descent direction")
    fig.suptitle(title)
    _finish(fig, path)


def plot_armijo(res, Xdes, Udes, Q, R, QT, path, title, n_grid=41):
    """Armijo plot J(gamma): true cost, tangent, and Armijo line, for early iterations and the last one."""
    keys = [k for k in sorted(k for k in res.ls_data if k != "last")]
    last = res.ls_data["last"]
    if last[0] not in keys:
        keys.append(last[0])
    data = {k: (res.ls_data[k] if k in res.ls_data else last) for k in keys}
    fig, ax = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
    gam = np.linspace(0, 1, n_grid)
    for a, k in zip(ax[0], keys):
        _, X, U, K, sig = data[k]
        J0 = total_cost(X, U, Xdes, Udes, Q, R, QT)
        slope = res.slope[k]
        Js = []
        for g in gam:
            Xn, Un = _closed_loop_rollout(X[:, 0], X, U, K, sig, g, res.params, DT)
            Js.append(total_cost(Xn, Un, Xdes, Udes, Q, R, QT))
        Js = np.array(Js)
        a.plot(gam, Js, color=OPT["color"], lw=2, label=r"$J(u+\gamma\,\Delta u)$")
        a.plot(gam, J0 + gam * slope, color="0.4", ls="--", label="tangent")
        c = res.armijo[k]["c"]
        a.plot(gam, J0 + c * gam * slope, color=PALETTE[0], ls=":", label=f"Armijo line (c={c:g})")
        a.set_xlabel(r"step size $\gamma$")
        a.set_title(f"iteration {k}")
        finite = Js[np.isfinite(Js)]
        lo, hi = finite.min(), min(finite.max(), J0 + 2 * abs(slope) + 1e-9) if finite.size else (0, 1)
        pad = 0.05 * (hi - lo + 1e-12)
        a.set_ylim(lo - pad, max(hi, J0) + pad)
    ax[0][0].set_ylabel("cost")
    ax[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(title)
    _finish(fig, path)


def plot_tracking(Xg, Ug, runs, path, title):
    """Desired (generated) trajectory and closed-loop trajectories for several scenarios.

    runs: list of (label, X, U)
    """
    t = _time(Xg)
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    for i in range(4):
        ax[i].plot(t, Xg[i], label="desired (optimal)", **DESIRED)
        for n, (lab, X, U) in enumerate(runs):
            ax[i].plot(t, X[i], color=PALETTE[n % len(PALETTE)], lw=1.3, label=lab)
        ax[i].set_ylabel(STATE_LABELS[i])
    ax[4].plot(t, Ug[0], **DESIRED)
    for n, (lab, X, U) in enumerate(runs):
        ax[4].plot(t, U[0], color=PALETTE[n % len(PALETTE)], lw=1.1)
    ax[4].set_ylabel(r"$\tau$ [Nm]")
    ax[4].set_xlabel("time [s]")
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_title(title)
    _finish(fig, path)


def plot_tracking_error(Xg, runs, path, title, logy=False):
    t = _time(Xg)
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for i in range(4):
        for n, (lab, X, U) in enumerate(runs):
            e = X[i] - Xg[i]
            ax[i].plot(t, e, color=PALETTE[n % len(PALETTE)], lw=1.4, label=lab)
        ax[i].set_ylabel(ERR_LABELS[i])
    ax[3].set_xlabel("time [s]")
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].set_title(title)
    _finish(fig, path)


def plot_constraints(Xg, Ug, runs, limits, path, title):
    """Constrained vs unconstrained MPC: theta_1 error and torque."""
    t = _time(Xg)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, Xg[0], label="desired", **DESIRED)
    ax[2].plot(t, Ug[0], label="desired", **DESIRED)
    for n, (lab, X, U) in enumerate(runs):
        c = PALETTE[n % len(PALETTE)]
        ax[0].plot(t, X[0], color=c, lw=1.4, label=lab)
        ax[1].plot(t, X[0] - Xg[0], color=c, lw=1.4)
        ax[2].plot(t, U[0], color=c, lw=1.2)
    for L in limits:
        ax[2].axhline(L, color="0.6", lw=0.8, ls=":")
        ax[2].axhline(-L, color="0.6", lw=0.8, ls=":")
    ax[0].set_ylabel(STATE_LABELS[0])
    ax[1].set_ylabel(ERR_LABELS[0])
    ax[2].set_ylabel(r"$\tau$ [Nm]")
    ax[2].set_xlabel("time [s]")
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].set_title(title)
    _finish(fig, path)
