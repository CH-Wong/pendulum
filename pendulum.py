from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Mathematics here: https://en.wikipedia.org/wiki/Spherical_pendulum

def spherical_to_cartesian(l, theta, phi):
    x = l*np.sin(theta)*np.cos(phi)
    y = l*np.sin(theta)*np.cos(phi)
    z = l*l(1-np.cos(theta))

    return x, y, z



def runga_kutta():
    # \frac{d\theta}{dt} = \omega
    # \frac{d\omega}{dt} = -\frac{g}{L}sin(\theta)

    h = 0.001   # [s]
    t0 = 0      # [s]
    tf = 10     # [s]
    g = 9.81    # [m/s^2]
    L = 1       # [m]

    # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
    time_steps = np.linspace(t0, tf, int((tf-t0)/h))
    print(len(time_steps))

    theta_0 = 0
    omega_0 = np.pi/10

    theta_approx = np.zeros(np.shape(time_steps))
    omega_approx = np.zeros(np.shape(time_steps))

    theta_approx[0] = theta_0
    omega_approx[0] = omega_0
    
    for num, t in enumerate(time_steps[:-1]):
        theta_approx[num+1] = theta_approx[num] + omega_approx[num]*h
        omega_approx[num+1] = omega_approx[num] - (g/L)*np.sin(theta_approx[num])*h

    # theta_approx = (theta_approx) % (2 * np.pi)
    x = L*np.sin(theta_approx)
    y = -L*np.cos(theta_approx)

    plt.figure()
    plt.plot(theta_approx, omega_approx, ".-")
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$\omega$ [rad/s]")
    
    plt.figure()
    plt.plot(time_steps, theta_approx, ".-")
    plt.xlabel("t [s]")
    plt.ylabel(r"$\theta(t)$ [rad]")
    
    plt.figure()
    plt.plot(x, y, ".-")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")


    plt.show()




runga_kutta()

