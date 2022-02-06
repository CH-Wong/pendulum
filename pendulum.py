from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pyqtgraph

# Mathematics here: https://en.wikipedia.org/wiki/Spherical_pendulum

def spherical_to_cartesian(l, theta, phi):
    x = l*np.sin(theta)*np.cos(phi)
    y = l*np.sin(theta)*np.cos(phi)
    z = l*l(1-np.cos(theta))

    return x, y, z



def runga_kutta():
    # Implementation of 2nd order Runga-Kutta integration of simple pendulum equation
    # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
    # \frac{d\theta}{dt} = \omega
    # \frac{d\omega}{dt} = -\frac{g}{L}sin(\theta)

    # Simulation parameters
    h = 0.001   # [s] Integration time step-size
    t0 = 0      # [s] Starting time
    tf = 10     # [s] End time
    g = 9.81    # [m/s^2] Graviational acceleration
    L = 1       # [m] Length of pendulum arm

    # Pre-initialize matrices
    time = np.linspace(t0, tf, int((tf-t0)/h))
    theta = np.zeros(np.shape(time))
    omega = np.zeros(np.shape(time))

    # Initial conditions
    theta_0 = np.pi/2
    omega_0 = np.pi/10

    # Insert initial conditions in vectors
    theta[0] = theta_0
    omega[0] = omega_0
    
    for num, t in enumerate(time[:-1]):
        # Estimation of function slope (derivative) at previous time-step previous guessed value
        k1_theta = omega[num]
        # Estimation of function value half an integration step (h/2) later
        theta_1 = theta[num] + k1_theta*h/2

        # Estimation of function slope (derivative) at previous time-step previous guessed value
        k1_omega = - (g/L)*np.sin(theta[num])
        # Estimation of function value half an integration step (h/2) later
        omega_1 = omega[num] + k1_omega*h/2

        # Estimation of function slope at time t0 + (h/2)
        k2_theta = omega_1
        # Function estimation using slope k2
        theta[num+1] = theta[num] + k2_theta*h

        # Estimation of function slope at time t0 + (h/2)
        k2_omega = -(g/L)*np.sin(theta_1)
        # Function estimation using slope k2
        omega[num+1] = omega[num] + k2_omega*h

    # Translate to cartesion coordinates for plotting
    x = L*np.sin(theta)
    y = -L*np.cos(theta)

    # Translate radians to degrees
    theta = theta*180/np.pi
    omega = omega*180/np.pi


    plt.figure()
    plt.plot(theta, omega, ".-")
    plt.xlabel(r"$\theta$ [$\degree$]")
    plt.ylabel(r"$\omega$ [$\degree/s$]")
    
    plt.figure()
    plt.plot(time, theta, ".-")
    plt.xlabel("t [s]")
    plt.ylabel(r"$\theta(t)$ [$\degree$]")
    
    plt.figure()
    plt.plot(x, y, ".-")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")


    plt.show()



runga_kutta()

