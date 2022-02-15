# Native Libraries
import sys

# External Libraries
import numpy as np
from PyQt5 import QtWidgets

# Local Imports
from pendulum.animation import Pendulum3DGui
from pendulum.pendulum import Pendulum3D


# 3D PENDULUM ANIMATION
# Simulation parameters
h = 0.001   # [s] Integration time step-size
t0 = 0      # [s] Starting time
g = 9.81    # [m/s^2] Graviational acceleration
L = 1       # [m] Length of pendulum arm
m = 1       # [kg] Mass of ball

# Initial conditions
theta_0 = np.pi/2       # [rad] Initial polar angle
phi_0 = 0               # [rad] Initial azimuthal angle

omega_0 = -np.pi/2     # [rad/s] Initial polar angular velocity
sigma_0 = np.pi/2      # [rad/s] Initial azimuthal angular velocity 

p = Pendulum3D(h, g, L, m, theta_0, phi_0, omega_0, sigma_0, t0)

app = QtWidgets.QApplication(sys.argv)
gui = Pendulum3DGui(p, shown_points=5000, iterations=10)

sys.exit(app.exec_())