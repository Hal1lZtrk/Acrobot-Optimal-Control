"""Task 3 - trajectory tracking with a time-varying LQR from perturbed initial conditions."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acrobot import config as C                      # noqa: E402
from acrobot import plotting as P                    # noqa: E402
from acrobot.pipeline import RESULTS, load_swing_up, write_json  # noqa: E402
from acrobot.tracking import track_lqr              # noqa: E402


def main():
    Xg, Ug, _, _ = load_swing_up()
    runs, rows = [], {}
    for name, noise in C.SCENARIOS.items():
        X, U = track_lqr(Xg, Ug, Xg[:, 0] + noise, C.TRACK_Q, C.TRACK_R, C.TRACK_QT)
        runs.append((name, X, U))
        e = X - Xg
        rows[name] = dict(initial_error=noise.tolist(), max_abs_error_theta1=float(np.abs(e[0]).max()),
                          final_error_norm=float(np.linalg.norm(e[:, -1])), peak_torque=float(np.abs(U).max()))
        print(f"{name:15s} max|e_theta1| = {rows[name]['max_abs_error_theta1']:.3f} rad, "
              f"final ||e|| = {rows[name]['final_error_norm']:.2e}, peak torque = {rows[name]['peak_torque']:.1f} Nm")
    out = os.path.join(RESULTS, "figures")
    P.plot_tracking(Xg, Ug, runs, f"{out}/task3_tracking.png", "Task 3: LQR tracking of the swing-up")
    P.plot_tracking_error(Xg, runs, f"{out}/task3_error.png", "Task 3: LQR tracking error")
    write_json("task3_summary.json", rows)
    P.show()


if __name__ == "__main__":
    main()
