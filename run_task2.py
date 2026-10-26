"""Task 2 - optimal swing-up (hanging down -> upright) on a smooth desired curve, Set 3.

Takes about 1-2 minutes.  The result is cached in results/swing_up_trajectory.npz and re-used by Tasks 3-5.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acrobot import config as C                      # noqa: E402
from acrobot import plotting as P                    # noqa: E402
from acrobot.pipeline import RESULTS, SWING_UP_FILE, solve_swing_up, write_json  # noqa: E402


def main(max_iters=C.T2_MAX_ITERS):
    Xr, Ur, res, secs = solve_swing_up(max_iters)
    out = os.path.join(RESULTS, "figures")
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(SWING_UP_FILE, x=res.x, u=res.u, x_desired=Xr, u_desired=Ur)
    P.plot_trajectory(res.x, res.u, Xr, Ur, f"{out}/task2_trajectory.png", "Task 2: optimal swing-up vs desired curve")
    P.plot_intermediate(res, Xr, [0, 1, 5, 20, 60], f"{out}/task2_intermediate.png", "Task 2: intermediate iterates")
    P.plot_armijo(res, Xr, Ur, C.T2_Q, C.T2_R, C.T2_QT, f"{out}/task2_armijo.png", "Task 2: Armijo line search")
    P.plot_convergence(res, f"{out}/task2_convergence.png", "Task 2: convergence")
    xT = res.x[:, -1]
    summary = dict(task=2, converged=bool(res.converged), iterations=res.iters, first_cost=res.J[0], final_cost=res.J[-1],
                   final_descent_norm=res.dnorm[-1], seconds=round(secs, 1), peak_torque=float(np.abs(res.u).max()),
                   final_state=[float(v) for v in xT], final_angle_error_to_pi=float(abs(xT[0] - np.pi)),
                   backtracking_iterations=int(sum(g < 1.0 for g in res.gamma)),
                   max_abs_theta2=float(np.abs(res.x[1]).max()), max_abs_dtheta1=float(np.abs(res.x[2]).max()),
                   max_abs_dtheta2=float(np.abs(res.x[3]).max()), time_of_peak_torque=float(np.abs(res.u[0]).argmax() * 0.01),
                   time_theta1_within_0p1_of_pi=float(np.argmax(np.abs(res.x[0] - np.pi) < 0.1) * 0.01),
                   control_effort_int_u2=float(np.sum(res.u[0, :-1] ** 2) * 0.01),
                   cost_history=res.J, descent_norm_history=res.dnorm, gamma_history=res.gamma)
    write_json("task2_summary.json", summary)
    print(summary)
    P.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=C.T2_MAX_ITERS)
    ap.add_argument("--no-show", action="store_true", help="save figures only, open no windows")
    main(ap.parse_args().max_iters)
