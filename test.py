from pendulum.pendulum import Pendulum
import numpy as np

# Simulation parameters
h = 0.001   # [s] Integration time step-size
t0 = 5      # [s] Starting time
g = 9.81    # [m/s^2] Graviational acceleration
L = 2       # [m] Length of pendulum arm

# Initial conditions
theta_0 = np.pi/4
omega_0 = -np.pi/2

p = Pendulum(h, g, L, theta_0, omega_0, t0)

for i in range(100):
    p.iterate()

print(p.cartesian())