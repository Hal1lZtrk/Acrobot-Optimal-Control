"""Acrobot dynamics, RK4 discretisation and exact (complex-step) Jacobians.

Model (assignment, Task 0)::

    M(q) ddq + C(q,dq) dq + F dq + G(q) = [0, tau]^T

All functions accept a *batch* of states ``x`` with shape ``(4, N)`` and
inputs ``u`` with shape ``(N,)`` or ``(1, N)`` so that a whole trajectory
can be propagated / linearised in one vectorised call.  A single state
``(4,)`` with scalar/``(1,)`` input also works.

Jacobians use the *complex-step* derivative: for a real-analytic ``f``,
``df/dx_j = Im f(x + i h e_j) / h`` with ``h = 1e-30`` is exact to machine
precision (no subtractive cancellation, unlike finite differences).
"""
import math

import numpy as np

from .params import DEFAULT, DT, NI, NS, AcrobotParams


def _as_batch(x, u):
    x = np.asarray(x)
    u = np.asarray(u)
    single = x.ndim == 1
    if single:
        x = x[:, None]
    u = u.reshape(-1) if u.ndim != 1 or u.size != x.shape[1] else u
    if u.size == 1 and x.shape[1] > 1:
        u = np.full(x.shape[1], u.item(), dtype=u.dtype)
    return x, u, single


def continuous_dynamics(x, u, p: AcrobotParams = DEFAULT):
    """x_dot = f_c(x, u) (batched)."""
    x, u, single = _as_batch(x, u)
    q1, q2, dq1, dq2 = x
    c2, s2 = np.cos(q2), np.sin(q2)

    M11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2 * p.l1 * p.lc2 * c2)
    M12 = p.I2 + p.m2 * p.lc2 * (p.l1 * c2 + p.lc2)
    M22 = p.I2 + p.m2 * p.lc2**2 * np.ones_like(q2)

    h = p.m2 * p.l1 * p.lc2
    C11, C12 = -h * dq2 * s2, -h * (dq1 + dq2) * s2
    C21 = h * dq1 * s2

    G1 = p.g * p.m1 * p.lc1 * np.sin(q1) + p.g * p.m2 * (p.l1 * np.sin(q1) + p.lc2 * np.sin(q1 + q2))
    G2 = p.g * p.m2 * p.lc2 * np.sin(q1 + q2)

    r1 = -(C11 * dq1 + C12 * dq2) - p.f1 * dq1 - G1
    r2 = u - (C21 * dq1) - p.f2 * dq2 - G2

    det = M11 * M22 - M12 * M12
    ddq1 = (M22 * r1 - M12 * r2) / det
    ddq2 = (-M12 * r1 + M11 * r2) / det
    out = np.stack([dq1, dq2, ddq1, ddq2])
    return out[:, 0] if single else out


def step(x, u, p: AcrobotParams = DEFAULT, dt: float = DT):
    """One RK4 step  x_{t+1} = f(x_t, u_t)  (batched)."""
    k1 = continuous_dynamics(x, u, p)
    k2 = continuous_dynamics(x + 0.5 * dt * k1, u, p)
    k3 = continuous_dynamics(x + 0.5 * dt * k2, u, p)
    k4 = continuous_dynamics(x + dt * k3, u, p)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def jacobians(X, U, p: AcrobotParams = DEFAULT, dt: float = DT):
    """Exact Jacobians of the RK4 map along a trajectory.

    X : (4, N), U : (1, N) or (N,).  Returns A (4, 4, N) and B (4, 1, N).
    """
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float).reshape(-1)
    N = X.shape[1]
    h = 1e-30
    A = np.empty((NS, NS, N))
    B = np.empty((NS, NI, N))
    for j in range(NS):
        Xp = X.astype(complex)
        Xp[j] += 1j * h
        A[:, j, :] = step(Xp, U.astype(complex), p, dt).imag / h
    B[:, 0, :] = step(X.astype(complex), U + 1j * h, p, dt).imag / h
    return A, B


def rollout(x0, U, p: AcrobotParams = DEFAULT, dt: float = DT):
    """Open-loop simulation. U has shape (1, T) (last column unused). Returns X (4, T)."""
    U = np.asarray(U).reshape(1, -1)
    T = U.shape[1]
    fs = make_scalar_step(p, dt)
    X = np.empty((NS, T))
    X[:, 0] = x0
    s = tuple(float(v) for v in x0)
    u_list = U[0].tolist()
    for t in range(T - 1):
        s = fs(*s, u_list[t])
        X[:, t + 1] = s
    return X


def make_scalar_step(p: AcrobotParams = DEFAULT, dt: float = DT):
    """Pure-Python RK4 step for a single state (about 10x faster than numpy on 4-vectors).

    Used inside long sequential rollouts (line search, closed-loop simulation).
    Returns ``step(x0, x1, x2, x3, u) -> (x0, x1, x2, x3)``; verified against
    :func:`step` in the unit tests.
    """
    sin, cos = math.sin, math.cos
    a = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2)
    b = 2 * p.m2 * p.l1 * p.lc2
    c = p.I2 + p.m2 * p.lc2**2
    e = p.m2 * p.lc2 * p.l1
    h = p.m2 * p.l1 * p.lc2
    gA = p.g * (p.m1 * p.lc1 + p.m2 * p.l1)
    gB = p.g * p.m2 * p.lc2
    f1, f2 = p.f1, p.f2

    def fc(q1, q2, v1, v2, u):
        c2, s2 = cos(q2), sin(q2)
        M11 = a + b * c2
        M12 = c + e * c2
        G1 = gA * sin(q1) + gB * sin(q1 + q2)
        G2 = gB * sin(q1 + q2)
        r1 = h * v2 * s2 * v1 + h * (v1 + v2) * s2 * v2 - f1 * v1 - G1
        r2 = u - h * v1 * s2 * v1 - f2 * v2 - G2
        det = M11 * c - M12 * M12
        return v1, v2, (c * r1 - M12 * r2) / det, (-M12 * r1 + M11 * r2) / det

    def step_s(q1, q2, v1, v2, u):
        k1 = fc(q1, q2, v1, v2, u)
        k2 = fc(q1 + 0.5 * dt * k1[0], q2 + 0.5 * dt * k1[1], v1 + 0.5 * dt * k1[2], v2 + 0.5 * dt * k1[3], u)
        k3 = fc(q1 + 0.5 * dt * k2[0], q2 + 0.5 * dt * k2[1], v1 + 0.5 * dt * k2[2], v2 + 0.5 * dt * k2[3], u)
        k4 = fc(q1 + dt * k3[0], q2 + dt * k3[1], v1 + dt * k3[2], v2 + dt * k3[3], u)
        d6 = dt / 6.0
        return (q1 + d6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                q2 + d6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                v1 + d6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
                v2 + d6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]))

    return step_s
