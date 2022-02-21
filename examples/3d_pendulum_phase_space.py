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
phi_0 = np.pi           # [rad] Pendulum starting position
omega_0 = np.pi/10      # [rad/s] Pendulum starting speed
sigma_0 = np.pi/10      # [rad/s] Pendulum starting speed

p = Pendulum3D(h, g, L, m, theta_0, phi_0, omega_0, sigma_0, t0=t0)

# Pre-initialize matrices
time = np.linspace(t0, tf, int((tf-t0)/h))
phase_space = np.zeros((len(time), 3))
coordinates = np.zeros((len(time), 3))

# Insert initial conditions in vectors

print(p.phase_space())
print(p.cartesian())

phase_space[0,:] = p.phase_space()
coordinates[0,:] = p.cartesian()



for num, t in enumerate(time[:-1]):
    p.iterate()
    phase_space[num+1,:] = p.phase_space()
    coordinates[num+1,:] = p.cartesian()

print(coordinates)


plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xs=phase_space[:,0], ys=phase_space[:,1], zs=phase_space[:,2])
ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel(r'$\phi$ [rad]')
ax.set_zlabel(r'$P_\theta$ [$kg m^2 s^{-1}$]')

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xs=coordinates[:,0], ys=coordinates[:,1], zs=coordinates[:,2])
ax.set_xlabel(r'$x$ [$m$]')
ax.set_ylabel(r'$y$ [$m$]')
ax.set_zlabel(r'$z$ [$m$]')


plt.figure()
plt.plot(time, phase_space[:,0])
plt.xlabel(r't [s]')
plt.ylabel(r'$\theta$ [rad]')


plt.figure()
plt.plot(time, phase_space[:,1])
plt.xlabel(r't [s]')
plt.ylabel(r'$\phi$ [$rad$]')


plt.figure()
plt.plot(time, phase_space[:,2])
plt.xlabel(r't [s]')
plt.ylabel(r'$P_\theta$ [$kg m^2 s^{-1}$]')

plt.show()


