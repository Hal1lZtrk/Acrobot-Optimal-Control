"""Trajectory tracking around (x_gen, u_gen): time-varying LQR (Task 3) and linear MPC (Task 4)."""
import numpy as np

from .dynamics import jacobians, make_scalar_step
from .params import DEFAULT, DT, NI, NS


def lqr_gains(A, B, Q, R, QT, return_P=False):
    """Finite-horizon discrete LQR, backward Riccati.

    Returns K (1,4,T) with du_t = K_t dx_t  (and the cost-to-go matrices P (4,4,T) if requested).
    """
    T = A.shape[2]
    K = np.zeros((NI, NS, T))
    Ps = np.zeros((NS, NS, T))
    P = QT.copy()
    Ps[:, :, -1] = P
    for t in range(T - 2, -1, -1):
        At, Bt = A[:, :, t], B[:, :, t]
        PB = P @ Bt
        K[:, :, t] = -np.linalg.solve(R + Bt.T @ PB, PB.T @ At)
        P = Q + At.T @ P @ At + At.T @ PB @ K[:, :, t]
        P = 0.5 * (P + P.T)
        Ps[:, :, t] = P
    return (K, Ps) if return_P else K


def _simulate(Xg, Ug, x0, controller, params, dt, u_limits=None):
    """Closed-loop simulation on the *nonlinear* plant. controller(t, dx) -> du."""
    T = Xg.shape[1]
    fs = make_scalar_step(params, dt)
    X = np.empty((NS, T))
    U = np.zeros((NI, T))
    s = tuple(float(v) for v in x0)
    X[:, 0] = s
    for t in range(T - 1):
        dx = np.array(s) - Xg[:, t]
        u = Ug[0, t] + controller(t, dx)
        if u_limits is not None:
            u = min(max(u, u_limits[0]), u_limits[1])
        U[0, t] = u
        s = fs(*s, u)
        X[:, t + 1] = s
    U[0, -1] = U[0, -2]
    return X, U


def track_lqr(Xg, Ug, x0, Q, R, QT, params=DEFAULT, dt=DT):
    """Task 3: u_t = u_gen,t + K_t (x_t - x_gen,t) with K from the LQ problem linearised on the trajectory."""
    A, B = jacobians(Xg, Ug, params, dt)
    K = lqr_gains(A, B, Q, R, QT)
    return _simulate(Xg, Ug, x0, lambda t, dx: float(K[0, :, t] @ dx), params, dt)


def _condensed_qp(A, B, Q, R, QT, N):
    """Return (Phi, Gamma) with stacked dx_{1..N} = Phi dx_0 + Gamma du_{0..N-1}."""
    Phi = np.zeros((N * NS, NS))
    Gam = np.zeros((N * NS, N))
    Ak = np.eye(NS)
    rows = []
    for k in range(N):
        Ak = A[:, :, k] @ Ak
        Phi[k * NS:(k + 1) * NS] = Ak
    for j in range(N):
        col = B[:, 0, j]
        Gam[j * NS:(j + 1) * NS, j] = col
        for k in range(j + 1, N):
            col = A[:, :, k] @ col
            Gam[k * NS:(k + 1) * NS, j] = col
    return Phi, Gam


def track_mpc(Xg, Ug, x0, N, Q, R, QT, u_limits=None, params=DEFAULT, dt=DT):
    """Task 4: receding-horizon controller around the trajectory.

    At every step the LQ problem over the next ``N`` samples

        min  sum_{k<N} dx_k' Q dx_k + du_k' R du_k + dx_N' P_{t+N} dx_N
        s.t. dx_{k+1} = A_k dx_k + B_k du_k,  dx_0 = x_t - x_gen,t,
             u_min <= u_gen,t+k + du_k <= u_max          (optional)

    is solved (condensed into a dense QP) and only du_0 is applied to the
    nonlinear plant.  Without limits the QP is solved in closed form
    (equivalent to a Riccati recursion over the window); with limits it is
    solved with qpOASES through CasADi.  If the solver fails the clipped
    unconstrained solution is used (counted in ``info['fallbacks']``).
    """
    T = Xg.shape[1]
    A, B = jacobians(Xg, Ug, params, dt)
    # Pad with the last linearisation so the window never shrinks below N.
    A = np.concatenate([A, np.repeat(A[:, :, -1:], N, axis=2)], axis=2)
    B = np.concatenate([B, np.repeat(B[:, :, -1:], N, axis=2)], axis=2)
    Ug_pad = np.concatenate([Ug, np.repeat(Ug[:, -1:], N, axis=1)], axis=1)

    # Terminal weight = cost-to-go of the full-horizon LQ problem at t+N.  This is what makes a
    # short window behave like the infinite-sighted controller (without it a 0.2 s window loses
    # the swing-up).  Unconstrained, this MPC is then identical to the Task 3 LQR.
    _, Pfull = lqr_gains(A[:, :, :T], B[:, :, :T], Q, R, QT, return_P=True)
    Pfull = np.concatenate([Pfull, np.repeat(Pfull[:, :, -1:], N, axis=2)], axis=2)
    Qbar0 = np.zeros((N * NS, N * NS))
    for k in range(N - 1):
        Qbar0[k * NS:(k + 1) * NS, k * NS:(k + 1) * NS] = Q
    Rbar = np.kron(np.eye(N), R)

    solver = None
    if u_limits is not None:
        import casadi as ca
        solver = ca.conic("qp", "qpoases", {"h": ca.Sparsity.dense(N, N)}, {"printLevel": "none"})
    info = {"fallbacks": 0, "active": 0}

    def controller(t, dx):
        Phi, Gam = _condensed_qp(A[:, :, t:t + N], B[:, :, t:t + N], Q, R, QT, N)
        Qbar = Qbar0.copy()
        Qbar[(N - 1) * NS:, (N - 1) * NS:] = Pfull[:, :, t + N]
        H = 2 * (Gam.T @ Qbar @ Gam + Rbar)
        g = 2 * Gam.T @ Qbar @ Phi @ dx
        H = 0.5 * (H + H.T)
        du = -np.linalg.solve(H, g)
        if u_limits is None:
            return float(du[0])
        lb = u_limits[0] - Ug_pad[0, t:t + N]
        ub = u_limits[1] - Ug_pad[0, t:t + N]
        try:
            sol = solver(h=H, g=g, lbx=lb, ubx=ub)
            d0 = float(sol["x"][0])
            if not np.isfinite(d0):
                raise ValueError
        except Exception:
            info["fallbacks"] += 1
            d0 = float(np.clip(du[0], lb[0], ub[0]))
        if d0 <= lb[0] + 1e-6 or d0 >= ub[0] - 1e-6:
            info["active"] += 1
        return d0

    X, U = _simulate(Xg, Ug, x0, controller, params, dt, u_limits)
    return X, U, info
