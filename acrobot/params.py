"""Physical and numerical parameters.

The tutor prescribed **Parameter Set 3** of Table 1 of the assignment.
Sets 1 and 2 are kept only so that the model can be explored; every
script in this repository uses ``SET3`` unless told otherwise.
"""
from dataclasses import dataclass

NS = 4          # states: theta1, theta2, dtheta1, dtheta2
NI = 1          # inputs: torque on joint 2
DT = 0.01       # RK4 step [s]


@dataclass(frozen=True)
class AcrobotParams:
    m1: float
    m2: float
    l1: float
    l2: float
    lc1: float
    lc2: float
    I1: float
    I2: float
    g: float = 9.81
    f1: float = 1.0
    f2: float = 1.0


SET1 = AcrobotParams(m1=1.0, m2=1.0, l1=1.0, l2=1.0, lc1=0.5, lc2=0.5, I1=0.33, I2=0.33)
SET2 = AcrobotParams(m1=2.0, m2=2.0, l1=1.5, l2=1.5, lc1=0.75, lc2=0.75, I1=1.5, I2=1.5)
SET3 = AcrobotParams(m1=1.5, m2=1.5, l1=2.0, l2=2.0, lc1=1.0, lc2=1.0, I1=2.0, I2=2.0)

DEFAULT = SET3
