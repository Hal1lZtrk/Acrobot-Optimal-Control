"""Task 5 - animation of the LQR-tracked swing-up.

    python run_task5.py            # interactive pygame window
    python run_task5.py --gif      # writes results/task5_swing_up.gif (headless)
    python run_task5.py --mp4      # writes results/task5_swing_up.mp4 (needs ffmpeg)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acrobot import config as C                      # noqa: E402
from acrobot.animation import export_video, interactive  # noqa: E402
from acrobot.pipeline import RESULTS, load_swing_up  # noqa: E402
from acrobot.tracking import track_lqr              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--no-show", action="store_true", help="(accepted for symmetry; Task 5 opens a pygame window)")
    a = ap.parse_args()
    Xg, Ug, _, _ = load_swing_up()
    X, U = track_lqr(Xg, Ug, Xg[:, 0] + C.SCENARIOS["Mixed error"], C.TRACK_Q, C.TRACK_R, C.TRACK_QT)
    if a.gif:
        print(export_video(X, U, Xg, os.path.join(RESULTS, "task5_swing_up.gif"), stride=5, fps=20))
    elif a.mp4:
        print(export_video(X, U, Xg, os.path.join(RESULTS, "task5_swing_up.mp4"), stride=2, fps=25))
    else:
        interactive(X, Xg)


if __name__ == "__main__":
    main()
