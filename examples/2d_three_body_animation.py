# Native Libraries
import sys

# External Libraries
import numpy as np
from PyQt5 import QtWidgets

# Local Imports
from three_body.three_body import ThreeBody2D
from three_body.animation import ThreeBody2DGui

# 2D PENDULUM ANIMATION
# Simulation parameters
h = 0.001           # [s] Integration time step-size
t0 = 0              # [s] Starting time
m1 = 3     # [kg] Mass of particle 1
m2 = 4    # [kg] Mass of particle 2
m3 = 5     # [kg] Mass of particle 3

# Initial coordinates @ t=t0
# Initial positions-vectors for all particles i (i.e. 1, 2 and 3)
r1_0 = np.array([0,0], dtype=np.float64)      # [m] Initial position of particle 1
r2_0 = np.array([0,4], dtype=np.float64)      # [m] Initial position of particle 2
r3_0 = np.array([5,4], dtype=np.float64)      # [m] Initial position of particle 3

# Initial velocity-vectors for all particles i (i.e. 1, 2, and 3)
v1_0 = np.array([0,0], dtype=np.float64)      # [m/s] Initial velocity of particle 1
v2_0 = np.array([0,0], dtype=np.float64)      # [m/s] Initial velocity of particle 2
v3_0 = np.array([0,0], dtype=np.float64)      # [m/s] Initial velocity of particle 3

TB_2D = ThreeBody2D(m1, m2, m3, r1_0, r2_0, r3_0, v1_0, v2_0, v3_0, t0, h)

app = QtWidgets.QApplication(sys.argv)
gui_2D = ThreeBody2DGui(TB_2D, shown_points=100, iterations=10)
gui_2D.show()

sys.exit(app.exec_())
