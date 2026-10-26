"""Task 1 - optimal transition between two equilibria (theta1 = -15 deg -> +15 deg), Set 3."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acrobot import config as C                      # noqa: E402
from acrobot import plotting as P                    # noqa: E402
from acrobot.pipeline import RESULTS, solve_transition, write_json  # noqa: E402
from acrobot.references import equilibrium          # noqa: E402


def main():
    for deg in C.T1_THETA1_DEG:
        q1, q2, tau = equilibrium(deg)
        print(f"equilibrium theta1 = {deg:+.0f} deg -> theta2 = {np.rad2deg(q2):+.4f} deg, holding torque = {tau:+.4f} Nm")
    Xr, Ur, res, secs = solve_transition()
    out = os.path.join(RESULTS, "figures")
    P.plot_trajectory(res.x, res.u, Xr, Ur, f"{out}/task1_trajectory.png", "Task 1: optimal transition vs desired curve")
    P.plot_intermediate(res, Xr, [0, 1, 2], f"{out}/task1_intermediate.png", "Task 1: intermediate iterates")
    P.plot_armijo(res, Xr, Ur, C.T1_Q, C.T1_R, C.T1_QT, f"{out}/task1_armijo.png", "Task 1: Armijo line search")
    P.plot_convergence(res, f"{out}/task1_convergence.png", "Task 1: convergence")
    summary = dict(task=1, converged=bool(res.converged), iterations=res.iters, first_cost=res.J[0], final_cost=res.J[-1],
                   final_descent_norm=res.dnorm[-1], seconds=round(secs, 1), peak_torque=float(np.abs(res.u).max()),
                   max_deviation_from_desired=[float(v) for v in np.abs(res.x - Xr).max(axis=1)],
                   final_deviation_from_desired=[float(v) for v in (res.x[:, -1] - Xr[:, -1])],
                   cost_history=res.J, descent_norm_history=res.dnorm, gamma_history=res.gamma)
    write_json("task1_summary.json", summary)
    print(summary)
    P.show()


if __name__ == "__main__":
    main()
