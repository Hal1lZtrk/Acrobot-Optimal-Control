import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acrobot import config as C                                    # noqa: E402
from acrobot import dynamics as d, references as R                 # noqa: E402
from acrobot.newton import newton_closed_loop, total_cost          # noqa: E402
from acrobot.params import DEFAULT, SET3                           # noqa: E402
from acrobot.tracking import lqr_gains, track_lqr, track_mpc       # noqa: E402


def test_default_is_set3():
    assert DEFAULT == SET3 and SET3.l1 == 2.0 and SET3.m1 == 1.5 and SET3.I1 == 2.0


def test_scalar_step_matches_vectorised():
    rng = np.random.default_rng(1)
    fs = d.make_scalar_step()
    for _ in range(20):
        x, u = rng.normal(size=4) * 2, rng.normal() * 10
        assert np.allclose(fs(*x, u), d.step(x, u), atol=1e-12)


def test_equilibrium_is_fixed_point():
    for deg in (-15, 15):
        q1, q2, tau = R.equilibrium(deg)
        x = np.array([q1, q2, 0, 0])
        assert np.allclose(d.step(x, tau), x, atol=1e-10)


def test_energy_conserved_without_friction_and_input():
    p = dataclasses.replace(SET3, f1=0.0, f2=0.0)

    def energy(x):
        c2 = np.cos(x[1])
        M11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2 * p.l1 * p.lc2 * c2)
        M12 = p.I2 + p.m2 * p.lc2 * (p.l1 * c2 + p.lc2)
        M22 = p.I2 + p.m2 * p.lc2**2
        K = 0.5 * (M11 * x[2]**2 + 2 * M12 * x[2] * x[3] + M22 * x[3]**2)
        U = -p.g * ((p.m1 * p.lc1 + p.m2 * p.l1) * np.cos(x[0]) + p.m2 * p.lc2 * np.cos(x[0] + x[1]))
        return K + U

    X = d.rollout([0.5, -0.3, 0.2, 0.1], np.zeros((1, 2000)), p)
    assert abs(energy(X[:, -1]) - energy(X[:, 0])) < 1e-6


def test_jacobians_match_finite_differences():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4, 5))
    U = rng.normal(size=(1, 5)) * 5
    A, B = d.jacobians(X, U)
    eps = 1e-6
    for n in range(5):
        for j in range(4):
            e = np.zeros(4); e[j] = eps
            fd = (d.step(X[:, n] + e, U[0, n]) - d.step(X[:, n] - e, U[0, n])) / (2 * eps)
            assert np.allclose(A[:, j, n], fd, atol=1e-7)
        fd = (d.step(X[:, n], U[0, n] + eps) - d.step(X[:, n], U[0, n] - eps)) / (2 * eps)
        assert np.allclose(B[:, 0, n], fd, atol=1e-7)


def _task1_problem(T=400):
    Xr, Ur = R.equilibrium_transition(T)
    return Xr, Ur


def test_newton_converges_on_equilibrium_transition_and_is_stationary():
    Xr, Ur = _task1_problem()
    res = newton_closed_loop(Xr, Ur, Xr[:, 0], np.zeros((1, 400)), C.T1_Q, C.T1_R, C.T1_QT,
                             max_iters=40, second_order=True, verbose=False)
    assert res.converged and res.dnorm[-1] < 1e-6
    # stationarity: finite-difference gradient of J wrt a few inputs ~ 0
    J0 = total_cost(res.x, res.u, Xr, Ur, C.T1_Q, C.T1_R, C.T1_QT)
    for t in (10, 150, 300):
        U = res.u.copy(); U[0, t] += 1e-4
        Jp = total_cost(d.rollout(Xr[:, 0], U), U, Xr, Ur, C.T1_Q, C.T1_R, C.T1_QT)
        U = res.u.copy(); U[0, t] -= 1e-4
        Jm = total_cost(d.rollout(Xr[:, 0], U), U, Xr, Ur, C.T1_Q, C.T1_R, C.T1_QT)
        assert abs((Jp - Jm) / 2e-4) < 1e-3
    assert J0 <= res.J[0]


def test_cost_decreases_monotonically():
    Xr, Ur = _task1_problem()
    res = newton_closed_loop(Xr, Ur, Xr[:, 0], np.zeros((1, 400)), C.T1_Q, C.T1_R, C.T1_QT,
                             max_iters=10, verbose=False)            # Gauss-Newton variant
    assert all(b <= a + 1e-9 for a, b in zip(res.J, res.J[1:]))


def test_lqr_tracking_recovers_swing_up_from_perturbation():
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "swing_up_trajectory.npz")
    if not os.path.exists(f):
        pytest.skip("run run_task2.py first")
    cache = np.load(f)
    U = cache["u"]
    X = d.rollout(cache["x"][:, 0], U)
    assert abs(X[0, -1] - np.pi) < 1e-2                       # the generated trajectory reaches the upright pose
    x0 = X[:, 0] + np.array([0.15, 0.15, 0.2, 0.2])
    Xc, _ = track_lqr(X, U, x0, C.TRACK_Q, C.TRACK_R, C.TRACK_QT)
    Xo = d.rollout(x0, U)                                        # same torques, no feedback
    assert np.linalg.norm(Xc[:, -1] - X[:, -1]) < 1e-3
    assert np.linalg.norm(Xo[:, -1] - X[:, -1]) > 10 * np.linalg.norm(Xc[:, -1] - X[:, -1])


def test_unconstrained_mpc_equals_lqr():
    Xr, Ur = _task1_problem(500)
    res = newton_closed_loop(Xr, Ur, Xr[:, 0], np.zeros((1, 500)), C.T1_Q, C.T1_R, C.T1_QT,
                             max_iters=30, second_order=True, verbose=False)
    x0 = res.x[:, 0] + np.array([0.05, -0.05, 0.1, 0.0])
    Xl, _ = track_lqr(res.x, res.u, x0, C.TRACK_Q, C.TRACK_R, C.TRACK_QT)
    Xm, _, _ = track_mpc(res.x, res.u, x0, 20, C.TRACK_Q, C.TRACK_R, C.TRACK_QT)
    assert np.abs(Xl - Xm).max() < 5e-3      # identical up to the last window, where the horizon is padded


def test_mpc_respects_torque_limits():
    pytest.importorskip("casadi")
    Xr, Ur = _task1_problem(300)
    res = newton_closed_loop(Xr, Ur, Xr[:, 0], np.zeros((1, 300)), C.T1_Q, C.T1_R, C.T1_QT,
                             max_iters=30, second_order=True, verbose=False)
    x0 = res.x[:, 0] + np.array([0.1, -0.1, 0.0, 0.0])
    _, U, info = track_mpc(res.x, res.u, x0, 20, C.TRACK_Q, C.TRACK_R, C.TRACK_QT, u_limits=(-12.0, 12.0))
    assert np.abs(U).max() <= 12.0 + 1e-9 and info["fallbacks"] == 0
