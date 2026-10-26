"""Shared helpers for the run_taskN.py scripts."""
import json
import os
import time

import numpy as np

from . import config as C
from . import references
from .dynamics import rollout
from .newton import newton_closed_loop

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SWING_UP_FILE = os.path.join(RESULTS, "swing_up_trajectory.npz")


def write_json(name, data):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=float)
    return path


def solve_transition(verbose=True):
    """Task 1: optimal transition between the two equilibria theta1 = -15 deg -> +15 deg."""
    T = int(round(C.T1_SECONDS / 0.01))
    Xr, Ur = references.equilibrium_transition(T, *C.T1_THETA1_DEG)
    t0 = time.time()
    res = newton_closed_loop(Xr, Ur, Xr[:, 0], np.zeros((1, T)), C.T1_Q, C.T1_R, C.T1_QT,
                             max_iters=60, second_order=True, verbose=verbose)
    return Xr, Ur, res, time.time() - t0


def solve_swing_up(max_iters=C.T2_MAX_ITERS, verbose=True):
    """Task 2: optimal swing-up (hanging -> upright), ~1-2 min on a laptop."""
    Xr, Ur = references.swing_up_reference(C.T2_SECONDS)
    T = Xr.shape[1]
    t0 = time.time()
    res = newton_closed_loop(Xr, Ur, Xr[:, 0], np.zeros((1, T)), C.T2_Q, C.T2_R, C.T2_QT,
                             max_iters=max_iters, tol=C.T2_TOL, second_order=True, verbose=verbose,
                             store_every=1)
    return Xr, Ur, res, time.time() - t0


def load_swing_up():
    """Load the cached optimal swing-up (x_gen, u_gen); recompute if the cache is missing."""
    if not os.path.exists(SWING_UP_FILE):
        print("No cached swing-up trajectory found -> running Task 2 first (about 1-2 minutes) ...")
        from run_task2 import main as run_task2
        run_task2()
    d = np.load(SWING_UP_FILE)
    U = d["u"]
    X = rollout(d["x"][:, 0], U)       # re-simulate: x_gen is exactly consistent with the dynamics
    return X, U, d["x_desired"], d["u_desired"]
