"""Newton-like algorithm for optimal control (closed-loop version).

Problem::

    min  sum_{t<T-1} 0.5 (x_t-xr_t)' Q (x_t-xr_t) + 0.5 (u_t-ur_t)' R (u_t-ur_t)
         + 0.5 (x_{T-1}-xr_{T-1})' QT (x_{T-1}-xr_{T-1})
    s.t. x_{t+1} = f(x_t, u_t),  x_0 given

Each iteration

1. linearises the dynamics along the current trajectory (exact Jacobians),
2. solves the resulting *affine* LQ problem with a backward Riccati pass
   -> feedback gains ``K_t`` and feed-forward terms ``sigma_t``
   (the Newton direction, using the Gauss-Newton Hessian),
3. runs an Armijo backtracking line search on the **closed-loop** update

       x+_{t+1} = f(x+_t,  u_t + K_t (x+_t - x_t) + gamma * sigma_t)

   The feedback term keeps the new trajectory close to the linearisation
   point, which is what makes the method work for long, unstable
   horizons (swing-up) where the open-loop update ``u + gamma*du``
   would only admit vanishingly small steps.

``mu`` is a Levenberg-type regularisation added to ``R + B'PB``; it is
increased automatically when the direction is not a descent direction
or the line search fails and relaxed again after successful iterations.
"""
from dataclasses import dataclass, field

import numpy as np

from .dynamics import jacobians, make_scalar_step, rollout
from .params import DEFAULT, DT, NI, NS


@dataclass
class NewtonResult:
    x: np.ndarray
    u: np.ndarray
    J: list = field(default_factory=list)          # cost at start of each iteration (+ final)
    dnorm: list = field(default_factory=list)      # ||open-loop descent direction||
    gamma: list = field(default_factory=list)      # accepted step size
    slope: list = field(default_factory=list)      # directional derivative grad'du
    x_hist: list = field(default_factory=list)
    u_hist: list = field(default_factory=list)
    armijo: list = field(default_factory=list)     # per-iteration line-search records
    ls_data: dict = field(default_factory=dict)    # {iter: (X, U, K, sig)} for selected iterations + the last one
    converged: bool = False
    iters: int = 0
    params: object = None


def total_cost(X, U, Xr, Ur, Q, R, QT):
    dX = X[:, :-1] - Xr[:, :-1]
    dU = U[:, :-1] - Ur[:, :-1]
    J = 0.5 * np.einsum("it,ij,jt->", dX, Q, dX) + 0.5 * np.einsum("it,ij,jt->", dU, R, dU)
    dXT = X[:, -1] - Xr[:, -1]
    return J + 0.5 * dXT @ QT @ dXT


def _riccati(A, B, Q, R, QT, q, r, qT, mu):
    """Backward pass of the affine LQ problem. Returns K, sigma or None if H is not PD."""
    T = A.shape[2]
    K = np.zeros((NI, NS, T))
    sig = np.zeros((NI, T))
    P = QT.copy()
    p = qT.copy()
    for t in range(T - 2, -1, -1):
        At, Bt = A[:, :, t], B[:, :, t]
        PB = P @ Bt
        H = R + Bt.T @ PB + mu * np.eye(NI)
        if np.any(np.linalg.eigvalsh(H) <= 1e-12):
            return None, None
        Hinv = np.linalg.inv(H)
        K[:, :, t] = -Hinv @ PB.T @ At
        sig[:, t] = -Hinv @ (r[:, t] + Bt.T @ p)
        P = Q + At.T @ P @ At + At.T @ PB @ K[:, :, t]
        p = q[:, t] + At.T @ p + At.T @ PB @ sig[:, t]
        P = 0.5 * (P + P.T)
    return K, sig


def second_derivatives(X, U, params=DEFAULT, dt=DT, eps=1e-5):
    """Second derivatives of the RK4 map (central differences of the exact Jacobians).

    Returns fxx (4,4,4,N), fxu (4,4,N) [d^2 f_i / dx_j du], fuu (4,N).
    """
    N = X.shape[1]
    fxx = np.empty((NS, NS, NS, N))
    for k in range(NS):
        Xp, Xm = X.copy(), X.copy()
        Xp[k] += eps
        Xm[k] -= eps
        Ap, _ = jacobians(Xp, U, params, dt)
        Am, _ = jacobians(Xm, U, params, dt)
        fxx[:, :, k, :] = (Ap - Am) / (2 * eps)
    Ap, Bp = jacobians(X, U + eps, params, dt)
    Am, Bm = jacobians(X, U - eps, params, dt)
    fxu = (Ap - Am) / (2 * eps)                 # (4,4,N): d A / du
    fuu = ((Bp - Bm) / (2 * eps))[:, 0, :]      # (4,N)
    return fxx, fxu[:, :, :], fuu


def _riccati_second_order(A, B, Q, R, QT, q, r, qT, mu, fxx, fxu, fuu):
    """DDP-style backward pass including the second derivatives of the dynamics."""
    T = A.shape[2]
    K = np.zeros((NI, NS, T))
    sig = np.zeros((NI, T))
    Qu_all = np.zeros((NI, T))
    Vx = qT.copy()
    Vxx = QT.copy()
    for t in range(T - 2, -1, -1):
        At, Bt = A[:, :, t], B[:, :, t]
        Qx = q[:, t] + At.T @ Vx
        Qu = r[:, t] + Bt.T @ Vx
        Qxx = Q + At.T @ Vxx @ At + np.einsum("i,ijk->jk", Vx, fxx[:, :, :, t])
        Quu = R + Bt.T @ Vxx @ Bt + np.array([[Vx @ fuu[:, t]]])
        Qux = Bt.T @ Vxx @ At + (Vx @ fxu[:, :, t])[None, :]
        Quu_reg = Quu + mu * np.eye(NI)
        if np.any(np.linalg.eigvalsh(Quu_reg) <= 1e-9):
            return None, None, None
        Qi = np.linalg.inv(Quu_reg)
        Kt = -Qi @ Qux
        st = -Qi @ Qu
        K[:, :, t], sig[:, t], Qu_all[:, t] = Kt, st, Qu
        Vx = Qx + Kt.T @ Quu @ st + Kt.T @ Qu + Qux.T @ st
        Vxx = Qxx + Kt.T @ Quu @ Kt + Kt.T @ Qux + Qux.T @ Kt
        Vxx = 0.5 * (Vxx + Vxx.T)
    return K, sig, Qu_all


def _closed_loop_rollout(x0, X, U, K, sig, gamma, params, dt):
    """x+_{t+1} = f(x+_t, u_t + K_t (x+_t - x_t) + gamma sigma_t)  (pure-Python inner loop)."""
    T = U.shape[1]
    fs = make_scalar_step(params, dt)
    Xl = X.T.tolist()
    ul = U[0].tolist()
    Kl = K[0].T.tolist()                       # (T, 4)
    sl = (gamma * sig[0]).tolist()
    Xn = np.empty_like(X)
    Un = np.empty_like(U)
    s = tuple(float(v) for v in x0)
    Xn[:, 0] = s
    with np.errstate(all="ignore"):
        for t in range(T - 1):
            k, xr = Kl[t], Xl[t]
            u = ul[t] + sl[t] + k[0] * (s[0] - xr[0]) + k[1] * (s[1] - xr[1]) + k[2] * (s[2] - xr[2]) + k[3] * (s[3] - xr[3])
            Un[0, t] = u
            s = fs(*s, u)
            if not (abs(s[0]) < 1e6 and abs(s[2]) < 1e6):     # diverged: reject this step size
                Xn[:, t + 1:] = np.nan
                Un[:, t + 1:] = np.nan
                return Xn, Un
            Xn[:, t + 1] = s
    Un[:, -1] = Un[:, -2]
    return Xn, Un


def _open_loop_direction(A, B, K, sig):
    """Linearised closed-loop response of the Newton step -> descent direction du."""
    T = A.shape[2]
    dx = np.zeros((NS, T))
    du = np.zeros((NI, T))
    for t in range(T - 1):
        du[:, t] = K[:, :, t] @ dx[:, t] + sig[:, t]
        dx[:, t + 1] = A[:, :, t] @ dx[:, t] + B[:, :, t] @ du[:, t]
    return du


def newton_closed_loop(Xr, Ur, x0, U0, Q, R, QT, max_iters=100, tol=1e-6,
                       params=DEFAULT, dt=DT, beta=0.5, c=1e-4, max_ls=25,
                       step0=1.0, mu0=0.0, verbose=True, store_every=1,
                       second_order=False, ls_iters=(0, 1, 2)):
    """Run the closed-loop Newton method.  U0: (1,T) initial input guess."""
    T = Xr.shape[1]
    U = np.array(U0, dtype=float).reshape(NI, T)
    X = rollout(x0, U, params, dt)
    res = NewtonResult(x=X, u=U)
    res.x_hist.append(X.copy())
    res.u_hist.append(U.copy())
    mu = mu0
    J = total_cost(X, U, Xr, Ur, Q, R, QT)

    for k in range(max_iters):
        A, B = jacobians(X, U, params, dt)
        q = Q @ (X - Xr)
        r = R @ (U - Ur)
        qT = QT @ (X[:, -1] - Xr[:, -1])
        q[:, -1] = 0.0
        r[:, -1] = 0.0

        # gradient of J w.r.t. u (adjoint recursion)
        lam = qT.copy()
        grad = np.zeros((NI, T))
        for t in range(T - 2, -1, -1):
            grad[:, t] = r[:, t] + B[:, :, t].T @ lam
            lam = q[:, t] + A[:, :, t].T @ lam

        # Newton direction (retry with larger mu if not a descent direction)
        if second_order:
            fxx, fxu, fuu = second_derivatives(X, U, params, dt)
        while True:
            if second_order:
                K, sig, Qu = _riccati_second_order(A, B, Q, R, QT, q, r, qT, mu, fxx, fxu, fuu)
            else:
                K, sig = _riccati(A, B, Q, R, QT, q, r, qT, mu)
            if K is not None:
                du = _open_loop_direction(A, B, K, sig)
                slope = float(np.sum(grad[:, :-1] * du[:, :-1]))
                if slope < 0:
                    break
            mu = max(1e-6, mu * 10.0) if mu > 0 else 1e-3
            if mu > 1e8:
                raise RuntimeError("Newton direction is not a descent direction even with huge mu")
        dnorm = float(np.linalg.norm(du))
        res.J.append(J)
        res.dnorm.append(dnorm)
        res.slope.append(slope)

        if dnorm > 1e-2 or 'last' not in res.ls_data:
            res.ls_data.pop('last', None)
            res.ls_data['last'] = (k, X.copy(), U.copy(), K.copy(), sig.copy())
        if k in ls_iters:
            res.ls_data[k] = (k, X.copy(), U.copy(), K.copy(), sig.copy())

        # Armijo backtracking on the closed-loop update
        g = step0
        accepted = False
        trace = []
        for _ in range(max_ls):
            Xn, Un = _closed_loop_rollout(x0, X, U, K, sig, g, params, dt)
            Jn = total_cost(Xn, Un, Xr, Ur, Q, R, QT)
            ok = np.isfinite(Jn) and Jn <= J + c * g * slope
            trace.append((g, Jn))
            if ok:
                accepted = True
                break
            g *= beta
        res.armijo.append(dict(iter=k, J0=J, slope=slope, trace=trace, c=c, accepted=accepted))

        if verbose:
            print(f"iter {k:3d}  J = {J:14.6f}  |du| = {dnorm:10.4e}  gamma = {g:8.5f}  mu = {mu:.0e}")

        if not accepted:
            mu = max(1e-6, mu * 10.0) if mu > 0 else 1e-3
            if mu > 1e8:
                break
            continue

        X, U, J = Xn, Un, Jn
        res.gamma.append(g)
        if k % store_every == 0:
            res.x_hist.append(X.copy())
            res.u_hist.append(U.copy())
        mu = mu / 10.0 if mu > 1e-6 else 0.0
        res.iters = k + 1
        if dnorm < tol:
            res.converged = True
            break

    res.J.append(J)
    res.x, res.u = X, U
    res.params = params
    if not res.converged and verbose:
        print("stopped without reaching tol (see README -> 'Convergence')")
    return res
