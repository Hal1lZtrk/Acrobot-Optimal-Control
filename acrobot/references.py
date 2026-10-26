"""Reference (desired) curves for Task 1 and Task 2."""
import numpy as np
from scipy.optimize import fsolve

from .params import DEFAULT, DT, NI, NS, AcrobotParams


def equilibrium(theta1_deg: float, p: AcrobotParams = DEFAULT):
    """Static equilibrium with theta1 imposed.

    With zero velocities/accelerations the dynamics reduce to
    ``G1(q) = 0`` and ``tau = G2(q)``; solve for (theta2, tau).
    Returns (q1, q2, tau) in rad / rad / Nm.
    """
    q1 = np.deg2rad(theta1_deg)

    def eqs(v):
        q2, tau = v
        G1 = p.g * p.m1 * p.lc1 * np.sin(q1) + p.g * p.m2 * (p.l1 * np.sin(q1) + p.lc2 * np.sin(q1 + q2))
        G2 = p.g * p.m2 * p.lc2 * np.sin(q1 + q2)
        return [G1, G2 - tau]

    sol, _, ier, msg = fsolve(eqs, [0.0, 0.0], full_output=True, xtol=1e-13)
    if ier != 1 or np.linalg.norm(eqs(sol)) > 1e-8:
        raise ValueError(f"no equilibrium found for theta1={theta1_deg} deg: {msg}")
    return q1, float(sol[0]), float(sol[1])


def _smoothstep(s):
    """Cubic 3s^2-2s^3 on [0,1] (zero slope at both ends)."""
    s = np.clip(s, 0.0, 1.0)
    return 3 * s**2 - 2 * s**3


def equilibrium_transition(T, theta1_start_deg=-15.0, theta1_end_deg=15.0, p: AcrobotParams = DEFAULT):
    """Task 1 reference: hold - smooth transition - hold, symmetric in time.

    Returns Xr (4,T), Ur (1,T) with Ur the holding torque tau = G2(q) along the curve
    (so Ur is exactly feasible at the two end equilibria).
    """
    q1s, q2s, _ = equilibrium(theta1_start_deg, p)
    q1e, q2e, _ = equilibrium(theta1_end_deg, p)
    t_trans = T // 4
    t_hold = (T - t_trans) // 2
    s = np.zeros(T)
    s[t_hold:t_hold + t_trans] = _smoothstep(np.linspace(0, 1, t_trans))
    s[t_hold + t_trans:] = 1.0
    Xr = np.zeros((NS, T))
    Xr[0] = q1s + (q1e - q1s) * s
    Xr[1] = q2s + (q2e - q2s) * s
    Xr[2] = np.gradient(Xr[0], DT)
    Xr[3] = np.gradient(Xr[1], DT)
    Ur = np.zeros((NI, T))
    Ur[0] = p.m2 * p.g * p.lc2 * np.sin(Xr[0] + Xr[1])
    return Xr, Ur


def swing_up_reference(T_seconds=15.0, q1_end=np.pi, shape="cubic"):
    """Task 2 reference: smooth cubic path from hanging-down (0,0) to upright (pi,0).

    The reference velocity is the time derivative of the path and the reference
    torque is zero (the optimiser has to discover the pumping motion).
    """
    T = int(round(T_seconds / DT))
    tt = np.arange(T) / (T - 1)
    s = _smoothstep(tt)
    Xr = np.zeros((NS, T))
    Xr[0] = q1_end * s
    Xr[2] = np.gradient(Xr[0], DT)
    Xr[2, 0] = Xr[2, -1] = 0.0          # starts and ends at rest
    Xr[3] = 0.0
    Ur = np.zeros((NI, T))
    return Xr, Ur
