"""Task 4 - trajectory tracking with MPC (unconstrained and with torque limits)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acrobot import config as C                      # noqa: E402
from acrobot import plotting as P                    # noqa: E402
from acrobot.pipeline import RESULTS, load_swing_up, write_json  # noqa: E402
from acrobot.tracking import track_lqr, track_mpc  # noqa: E402


def main():
    Xg, Ug, _, _ = load_swing_up()
    out = os.path.join(RESULTS, "figures")
    runs, rows = [], {}
    for name, noise in C.SCENARIOS.items():
        X, U, _ = track_mpc(Xg, Ug, Xg[:, 0] + noise, C.MPC_HORIZON, C.MPC_Q, C.MPC_R, C.MPC_QT)
        Xl, _ = track_lqr(Xg, Ug, Xg[:, 0] + noise, C.TRACK_Q, C.TRACK_R, C.TRACK_QT)
        runs.append((name, X, U))
        rows[name] = dict(max_abs_error_theta1=float(np.abs(X[0] - Xg[0]).max()),
                          final_error_norm=float(np.linalg.norm(X[:, -1] - Xg[:, -1])),
                          max_diff_to_lqr=float(np.abs(X - Xl).max()))
        print(f"MPC {name:15s} max|e_theta1| = {rows[name]['max_abs_error_theta1']:.3f} rad, "
              f"final ||e|| = {rows[name]['final_error_norm']:.2e}, max |x_MPC - x_LQR| = {rows[name]['max_diff_to_lqr']:.1e}")
    P.plot_tracking(Xg, Ug, runs, f"{out}/task4_tracking.png", f"Task 4: MPC tracking (horizon {C.MPC_HORIZON} = {C.MPC_HORIZON * 0.01:.1f} s)")
    P.plot_tracking_error(Xg, runs, f"{out}/task4_error.png", "Task 4: MPC tracking error")

    # constrained vs unconstrained (position-error scenario)
    noise = C.SCENARIOS["Position error"]
    x0 = Xg[:, 0] + noise
    X0, U0, _ = track_mpc(Xg, Ug, x0, C.MPC_HORIZON, C.MPC_Q, C.MPC_R, C.MPC_QT)
    cruns = [("unconstrained", X0, U0)]
    crow = {"unconstrained": dict(peak_torque=float(np.abs(U0).max()), max_abs_error_theta1=float(np.abs(X0[0] - Xg[0]).max()),
                                  final_error_norm=float(np.linalg.norm(X0[:, -1] - Xg[:, -1])))}
    for L in C.TORQUE_LIMITS:
        X, U, info = track_mpc(Xg, Ug, x0, C.MPC_HORIZON, C.MPC_Q, C.MPC_R, C.MPC_QT, u_limits=(-L, L))
        cruns.append((f"|tau| <= {L:g} Nm", X, U))
        crow[f"limit_{L:g}"] = dict(peak_torque=float(np.abs(U).max()), max_abs_error_theta1=float(np.abs(X[0] - Xg[0]).max()),
                                    final_error_norm=float(np.linalg.norm(X[:, -1] - Xg[:, -1])),
                                    steps_at_limit=int(info["active"]), solver_fallbacks=int(info["fallbacks"]))
        print(f"MPC |tau|<={L:g}: peak torque {np.abs(U).max():.1f} Nm, max|e_theta1| = {np.abs(X[0]-Xg[0]).max():.3f} rad, "
              f"final ||e|| = {np.linalg.norm(X[:, -1]-Xg[:, -1]):.2e}, steps at limit = {info['active']}, fallbacks = {info['fallbacks']}")
    P.plot_constraints(Xg, Ug, cruns, C.TORQUE_LIMITS, f"{out}/task4_constraints.png",
                       "Task 4: constrained vs unconstrained MPC (position-error start)")
    print(f"generated trajectory peak torque: {np.abs(Ug).max():.1f} Nm")
    write_json("task4_summary.json", dict(scenarios=rows, constraints=crow, reference_peak_torque=float(np.abs(Ug).max())))
    P.show()


if __name__ == "__main__":
    main()
