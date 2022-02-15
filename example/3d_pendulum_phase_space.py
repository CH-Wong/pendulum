import numpy as np
from pendulum.pendulum import Pendulum3D
import matplotlib.pyplot as plt

def phase_space():
    h = 0.001                # [s] Integration step
    t0 = 0                  # [s] Starting time
    tf = 100                  # [s] End time
    g = 9.81                # [m/s^2] Gravitational acceleration
    L = 1                   # [m] Pendulum length
    theta_0 = np.pi/2       # [rad] Pendulum Starting position 
    phi_0 = 0               # [rad] Pendulum starting position
    omega_0 = np.pi/10      # [rad/s] Pendulum starting speed
    sigma_0 = np.pi/10      # [rad/s] 


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
        omega[num+1] = p.omega

    # Translate radians to degrees
    theta = theta*180/np.pi
    omega = omega*180/np.pi

    plt.figure()
    plt.title(r"Phase Space Diagram($\theta,\omega$)")
    plt.plot(theta, omega, "-")
    plt.xlabel(r"$\theta$ [$\degree$]")
    plt.ylabel(r"$\omega$ [$\degree/s$]")

    plt.figure()
    plt.title(r"Position over time $\theta(t)$")
    plt.plot(time, theta, ".-")
    plt.xlabel("t [s]")
    plt.ylabel(r"$\theta(t)$ [$\degree$]")

    plt.show()


if __name__ == "__main__":
    phase_space()