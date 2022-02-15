# Native libraries
import sys

# External Libraries
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from matplotlib.colors import Colormap

from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg


# Mathematics here: https://en.wikipedia.org/wiki/Spherical_pendulum

class Pendulum2D():
    def __init__(self, h, g, L, theta_0, omega_0, t0=0):
        self.h = h
        self.L = L
        self.g = g

        # Set variables to initial conditions
        self.time = t0
        self.theta = theta_0
        self.omega = omega_0
        self.coordinates = self.cartesian()

    def cartesian(self):
        # Translate to cartesion coordinates for plotting
        return (self.L*np.sin(self.theta), -self.L*np.cos(self.theta))

    def iterate(self):
        # Implementation of 4th order Runga-Kutta integration of simple pendulum equation
        # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
        # \frac{d\theta}{dt} = \omega
        # \frac{d\omega}{dt} = -\frac{g}{L}sin(\theta)
        # TODO: Add friction?

        self.time += self.h

        # Estimate function slope k using current known (or previously guessed) position (theta, omega) 
        # and the differential equation for dtheta/dt and domega/dt
        k1_theta = self.omega
        k1_omega = - (self.g/self.L)*np.sin(self.theta)
        
        # Use the estimated slopes k1 to guess where the next point will be after timestep dt = h
        theta_1 = self.theta + k1_theta*self.h/2
        omega_1 = self.omega + k1_omega*self.h/2

        # Use first estimate (theta_1, omega_1) to repeat the process
        k2_theta = omega_1
        k2_omega = - (self.g/self.L)*np.sin(theta_1)

        # Now use slope k2 to make another estimate, more accurate than (theta_1, omega_1)
        theta_2 = self.theta + k2_theta*self.h/2
        omega_2 = self.omega + k2_omega*self.h/2

        # Use first estimate (theta_1, omega_1) to repeat the process
        k3_theta = omega_2
        k3_omega = - (self.g/self.L)*np.sin(theta_2)

        # Now use slope k2 to make another estimate, more accurate than (theta_1, omega_1)
        theta_3 = self.theta + k3_theta*self.h/2
        omega_3 = self.omega + k3_omega*self.h/2

        # Use first estimate (theta_1, omega_1) to repeat the process
        k4_theta = omega_3
        k4_omega = - (self.g/self.L)*np.sin(theta_3)

        # Use weighted average k1, k2, k3, k4 to estimate the actual next time point
        self.theta += self.h*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)/6
        self.omega += self.h*(k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)/6

        return (self.omega, self.theta)

class Pendulum3D():
    def __init__(self, h, g, L, m, theta_0, phi_0, omega_0, sigma_0, t0=0):
        self.h = h
        self.L = L
        self.g = g
        self.m = m

        # Set variables to initial conditions
        self.time = t0

        self.theta = theta_0
        self.phi = phi_0

        self.omega = omega_0
        self.sigma = sigma_0
        
        self.coordinates = self.cartesian()


    def cartesian(self):
        # Translate to cartesion coordinates for plotting
        return (self.L*np.sin(self.theta)*np.cos(self.phi), self.L*np.sin(self.theta)*np.sin(self.phi), self.L*(1 - np.cos(self.theta)))

    def iterate(self):
        # Implementation of 4th order Runga-Kutta integration of simple pendulum equation
        # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
        # \frac{d\theta}{dt} = \omega
        # \frac{d\omega}{dt} = -\frac{g}{L}sin(\theta)
        # TODO: Add friction?
        self.time += self.h

        # Estimate function slope k using current known (or previously guessed) position (theta, omega) 
        # and the differential equation for dtheta/dt and domega/dt
        k1_theta = self.omega
        k1_phi  = self.sigma

        k1_omega = np.sin(self.theta)*np.cos(self.theta)*self.sigma**2 - (self.g/self.L)*np.sin(self.theta)
        k1_sigma = -2*self.omega*self.sigma/np.tan(self.theta)

        # Use the estimated slopes k1 to guess where the next point will be after timestep dt = h
        theta_1 = self.theta + k1_theta*self.h/2
        phi_1 = self.phi + k1_phi*self.h/2

        omega_1 = self.omega + k1_omega*self.h/2
        sigma_1 = self.sigma + k1_sigma*self.h/2

        # Use first estimate (theta_1, omega_1) to repeat the process
        k2_theta = omega_1
        k2_phi  = sigma_1

        k2_omega = np.sin(theta_1)*np.cos(theta_1)*sigma_1**2 - (self.g/self.L)*np.sin(theta_1)
        k2_sigma = -2*omega_1*sigma_1/np.tan(theta_1)

        # Now use slope k2 to make another estimate, more accurate than (theta_1, omega_1)
        theta_2 = self.theta + k2_theta*self.h/2
        phi_2 = self.phi + k2_phi*self.h/2

        omega_2 = self.omega + k2_omega*self.h/2
        sigma_2 = self.sigma + k2_sigma*self.h/2

        # Use first estimate (theta_2, omega_2) to repeat the process
        k3_theta = omega_2
        k3_phi  = sigma_2

        k3_omega = np.sin(theta_2)*np.cos(theta_2)*sigma_2**2 - (self.g/self.L)*np.sin(theta_2)
        k3_sigma = -2*omega_2*sigma_2/np.tan(theta_2)

        # Now use slope k2 to make another estimate, more accurate than (theta_1, omega_1)
        theta_3 = self.theta + k3_theta*self.h/2
        phi_3 = self.phi + k3_phi*self.h/2

        omega_3 = self.omega + k3_omega*self.h/2
        sigma_3 = self.sigma + k3_sigma*self.h/2

        # Use first estimate (theta_1, omega_1) to repeat the process
        k4_theta = omega_3
        k4_phi  = sigma_3

        k4_omega = np.sin(theta_3)*np.cos(theta_3)*sigma_3**2 - (self.g/self.L)*np.sin(theta_3)
        k4_sigma = -2*omega_3*sigma_3/np.tan(theta_3)

        # Use weighted average k1, k2, k3, k4 to estimate the actual next time point
        self.theta += self.h*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)/6
        self.phi += self.h*(k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)/6
        self.omega += self.h*(k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)/6
        self.sigma += self.h*(k1_sigma + 2*k2_sigma + 2*k3_sigma + k4_sigma)/6

        return (self.theta, self.phi, self.omega, self.sigma)


def main():
    # Simulation parameters
    h = 0.001   # [s] Integration time step-size
    t0 = 5      # [s] Starting time
    g = 9.81    # [m/s^2] Graviational acceleration
    L = 1       # [m] Length of pendulum arm

    # Initial conditions
    theta_0 = np.pi/2
    omega_0 = -np.pi/2

    p = Pendulum2D(h, g, L, theta_0, omega_0, t0)

    return p 


if __name__ == '__main__':
    main()