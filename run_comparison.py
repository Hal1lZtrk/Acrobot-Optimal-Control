"""Cost along iterations: original course code (open-loop Newton, 80 iterations) vs this repository.

The baseline numbers (docs/baseline_open_loop_cost.csv) come from running the original course code with
the same Set 3 parameters, cost weights and 80 iterations as in its ``main_task2.py``.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acrobot import plotting as P                      # noqa: E402
from acrobot.pipeline import RESULTS, ROOT              # noqa: E402

plt = P.plt


def main():
    with open(os.path.join(ROOT, "docs", "baseline_open_loop_cost.csv")) as fh:
        rows = list(csv.DictReader(fh))
    base = [float(r["cost"]) for r in rows]
    with open(os.path.join(RESULTS, "task2_summary.json")) as fh:
        new = json.load(fh)["cost_history"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.semilogy(range(len(base)), base, color=P.PALETTE[0], lw=2, label=f"course code, open-loop Newton (80 its: J = {base[-1]:,.0f}, not converged)")
    ax.semilogy(range(len(new)), new, color=P.OPT["color"], lw=2, label=f"this repo, closed-loop Newton ({len(new) - 1} its: J = {new[-1]:,.0f}, converged)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost J (Task 2)")
    ax.set_title("Swing-up optimisation: same cost function, two optimisers")
    ax.legend(fontsize=8)
    P._finish(fig, os.path.join(RESULTS, "figures", "comparison_with_course_code.png"))
    P.show()


if __name__ == "__main__":
    main()
