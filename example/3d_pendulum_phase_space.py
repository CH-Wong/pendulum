from pendulum.pendulum import Pendulum3D
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

h = 0.001               # [s] Integration step
m = 1                   # [kg] Mass of ball
t0 = 0                  # [s] Starting time
tf = 10                 # [s] End time
g = 9.81                # [m/s^2] Gravitational acceleration
L = 1                   # [m] Pendulum length
theta_0 = np.pi/2       # [rad] Pendulum Starting position 
phi_0 = np.pi               # [rad] Pendulum starting position
omega_0 = np.pi/10      # [rad/s] Pendulum starting speed
sigma_0 = np.pi/10      # [rad/s] Pendulum starting speed


# Pre-initialize matrices
time = np.linspace(t0, tf, int((tf-t0)/h))

theta = np.zeros(np.shape(time))
phi = np.zeros(np.shape(time))

omega = np.zeros(np.shape(time))
sigma = np.zeros(np.shape(time))

# Insert initial conditions in vectors
theta[0] = theta_0
phi[0] = phi_0
omega[0] = omega_0
sigma[0] = sigma_0

p = Pendulum3D(h, g, L, theta_0, phi_0, omega_0, sigma_0, t0)

for num, t in enumerate(time[:-1]):
    p.iterate()
    theta[num+1] = p.theta
    phi[num+1] = p.phi

    omega[num+1] = p.omega
    sigma[num+1] = p.sigma


# Compute Conjugate Momentum in theta
p_theta = m*L**2*omega

# Translate radians to degrees
theta *= 180/np.pi
phi *= 180/np.pi

omega *= 180/np.pi
sigma *= 180/np.pi

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(theta, phi, p_theta)
ax.set_xlabel(r'$\theta$ [$\degree$]')
ax.set_ylabel(r'$\phi$ [$\degree$]')
ax.set_zlabel(r'$P_\theta$ [$kg m^2 s^{-1}$]')

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(theta, phi, p_theta)
ax.set_xlabel(r'$x$ [$m$]')
ax.set_ylabel(r'$y$ [$m$]')
ax.set_zlabel(r'$z$ [$m$]')

plt.show()

