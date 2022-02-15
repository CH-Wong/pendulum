# Native Libraries
import sys

# External Libraries
import numpy as np
from PyQt5 import QtWidgets

# Local Imports
from pendulum.animation import Pendulum2DGui
from pendulum.pendulum import Pendulum2D

# 2D PENDULUM ANIMATION
# Simulation parameters
h = 0.001   # [s] Integration time step-size
t0 = 0      # [s] Starting time
g = 9.81    # [m/s^2] Graviational acceleration
L = 1       # [m] Length of pendulum arm

# Initial conditions
theta_0 = np.pi/2       # [rad] Initial polar angle
omega_0 = -np.pi/2     # [rad/s] Initial polar angular velocity

app = QtWidgets.QApplication(sys.argv)

p_2D = Pendulum2D(h, g, L, theta_0, omega_0, t0)
gui_2D = Pendulum2DGui(p_2D, shown_points=1000, iterations=5)
gui_2D.show()

sys.exit(app.exec_())