
# Native libraries
import sys

# External Libraries
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from matplotlib.colors import Colormap

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg


# Mathematics here: https://en.wikipedia.org/wiki/Spherical_pendulum

class Pendulum:
    app = QtGui.QApplication(sys.argv)
    points = dict()
    widget = gl.GLViewWidget()

    # create the background grids
    gx = gl.GLGridItem()
    gx.rotate(90, 0, 1, 0)
    gx.translate(-10, 0, 0)
    widget.addItem(gx)

    gy = gl.GLGridItem()
    gy.rotate(90, 1, 0, 0)
    gy.translate(0, -10, 0)
    widget.addItem(gy)

    gz = gl.GLGridItem()
    gz.translate(0, 0, -10)
    widget.addItem(gz)

    iteration = 0

    def __init__(self, h, t0, tf, g, L, theta_0, omega_0):
        # Pre-initialize matrices
        self.time = np.linspace(t0, tf, int((tf-t0)/h))
        self.theta = np.zeros(np.shape(self.time))
        self.omega = np.zeros(np.shape(self.time))
        
        # Insert initial conditions in vectors
        self.theta[0] = theta_0
        self.omega[0] = omega_0


        self.mask_plot = gl.GLScatterPlotItem(
            pos=mask_points, 
            color=pg.glColor((mask_color, 1000))
            )


    def runga_kutta(self):
        # Implementation of 2nd order Runga-Kutta integration of simple pendulum equation
        # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
        # \frac{d\theta}{dt} = \omega
        # \frac{d\omega}{dt} = -\frac{g}{L}sin(\theta)

        for num, t in enumerate(self.time[:-1]):
            # Estimation of function slope (derivative) at previous time-step previous guessed value
            k1_theta = self.omega[num]
            # Estimation of function value half an integration step (h/2) later
            theta_1 = self.theta[num] + k1_theta*self.h/2

            # Estimation of function slope (derivative) at previous time-step previous guessed value
            k1_omega = - (self.g/self.L)*np.sin(self.theta[num])
            # Estimation of function value half an integration step (h/2) later
            omega_1 = self.omega[num] + k1_omega*self.h/2

            # Estimation of function slope at time t0 + (h/2)
            k2_theta = omega_1
            # Function estimation using slope k2
            self.theta[num+1] = self.theta[num] + k2_theta*self.h

            # Estimation of function slope at time t0 + (h/2)
            k2_omega = -(self.g/self.L)*np.sin(theta_1)
            # Function estimation using slope k2
            self.omega[num+1] = self.omega[num] + k2_omega*self.h

        # Translate to cartesion coordinates for plotting
        self.x = self.L*np.sin(self.theta)
        self.y = -self.L*np.cos(self.theta)

        # Translate radians to degrees
        self.theta = self.theta*180/np.pi
        self.omega = self.omega*180/np.pi


    def plot(self):
        plt.figure()
        plt.plot(self.theta, self.omega, ".-")
        plt.xlabel(r"$\theta$ [$\degree$]")
        plt.ylabel(r"$\omega$ [$\degree/s$]")
        
        plt.figure()
        plt.plot(self.time, self.theta, ".-")
        plt.xlabel("t [s]")
        plt.ylabel(r"$\theta(t)$ [$\degree$]")
        
        plt.figure()
        plt.plot(x, y, ".-")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")


        


def main():
    # Simulation parameters
    h = 0.001   # [s] Integration time step-size
    t0 = 0      # [s] Starting time
    tf = 10     # [s] End time
    g = 9.81    # [m/s^2] Graviational acceleration
    L = 1       # [m] Length of pendulum arm

    # Initial conditions
    theta_0 = np.pi/2
    omega_0 = np.pi/10

    p = Pendulum(h, t0, tf, L, theta_0, omega_0)
    p.runga_kutta()
    plt.show()


if __name__ == '__main__':
    main()