"""All weights / horizons used by the run_taskN.py scripts in one place."""
import numpy as np

# ---- Task 1: equilibrium-to-equilibrium transition -------------------------
T1_SECONDS = 10.0
T1_THETA1_DEG = (-15.0, 15.0)
T1_Q = np.diag([1.0, 1.0, 0.1, 0.1])
T1_R = np.array([[0.01]])
T1_QT = 1000.0 * T1_Q

# ---- Task 2: swing-up (hanging -> upright) --------------------------------
T2_SECONDS = 15.0
T2_Q = np.diag([10.0, 10.0, 1.0, 1.0])
T2_R = np.array([[0.1]])
T2_QT = np.diag([1e4, 1e4, 1e3, 1e3])
T2_MAX_ITERS = 400
T2_TOL = 1e-6

# ---- Task 3: LQR tracking -------------------------------------------------
TRACK_Q = np.diag([1000.0, 1000.0, 10.0, 10.0])
TRACK_R = np.array([[1.0]])
TRACK_QT = 10.0 * TRACK_Q

# Perturbed initial conditions (x0 = x_gen[:,0] + noise)
SCENARIOS = {
    "Position error": np.array([0.10, -0.10, 0.0, 0.0]),
    "Velocity error": np.array([0.0, 0.0, 0.5, -0.5]),
    "Mixed error": np.array([0.15, 0.15, 0.2, 0.2]),
    "No error": np.zeros(4),
}

# ---- Task 4: MPC ----------------------------------------------------------
MPC_HORIZON = 20
MPC_Q = TRACK_Q
MPC_R = TRACK_R
MPC_QT = TRACK_QT
TORQUE_LIMITS = (30.0, 25.0)     # |tau| <= limit [Nm]; the generated swing-up peaks at about 36.6 Nm
