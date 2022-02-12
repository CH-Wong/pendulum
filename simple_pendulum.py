
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

class Pendulum():
    # # Enable antialiasing for prettier plots
    # pg.setConfigOptions(antialias=True)
    def __init__(self, h, g, L, theta_0, omega_0, t0=0, num_shown_points = 100):
        self.num_shown_points = 100
        self.h = h
        self.L = L
        self.g = g

        # Pre-initialize matrices
        self.time = np.zeros(num_shown_points)
        self.theta = np.ones(num_shown_points)
        self.omega = np.ones(num_shown_points)
        self.x = np.zeros(num_shown_points)
        self.y = np.zeros(num_shown_points)
        
        # Insert initial conditions in vectors
        self.theta *= theta_0
        self.omega *= omega_0

        self.cartesian()

    def cartesian(self):
        # Translate to cartesion coordinates for plotting
        self.x = self.L*np.sin(self.theta)
        self.y = -self.L*np.cos(self.theta)


    def iterate(self):
        # Implementation of 2nd order Runga-Kutta integration of simple pendulum equation
        # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
        # \frac{d\theta}{dt} = \omega
        # \frac{d\omega}{dt} = -\frac{g}{L}sin(\theta)

        omega = self.omega[-1]
        theta = self.theta[-1]
        time = self.time[-1]

        self.omega[:-1] = self.omega[1:]
        self.theta[:-1] = self.theta[1:]
        self.time[:-1] = self.time[1:]
        

        self.time[-1] = time + self.h

        # Estimation of function slope (derivative) at previous time-step previous guessed value
        k1_theta = omega
        # Estimation of function value half an integration step (h/2) later
        theta_1 = theta + k1_theta*self.h/2

        # Estimation of function slope (derivative) at previous time-step previous guessed value
        k1_omega = - (self.g/self.L)*np.sin(theta)
        # Estimation of function value half an integration step (h/2) later
        omega_1 = omega + k1_omega*self.h/2

        # Estimation of function slope at time t0 + (h/2)
        k2_theta = omega_1
        # Function estimation using slope k2
        self.theta[-1]= theta + k2_theta*self.h

        # Estimation of function slope at time t0 + (h/2)
        k2_omega = -(self.g/self.L)*np.sin(theta_1)
        # Function estimation using slope k2
        self.omega[-1] = omega + k2_omega*self.h

        self.cartesian()

        
class pendulum_gui(QtWidgets.QWidget):

    def __init__(self, pendulum, interval = 100):
        # Initialize using built-in QWidget __init__ from inherited class
        super().__init__()

        # Set pendulum class in this class
        self.pendulum = pendulum
        # Set up PyQtGraph styling configurations        
        pg.setConfigOption('background', 0.95)
        pg.setConfigOptions(antialias=True)
        

        # Swinging pendulum animation
        # Create PyQTGraph plotting widget instance and configure layout
        self.pendulum_widget = pg.PlotWidget()
        self.pendulum_widget.setAspectLocked(lock=True, ratio=1)
        self.pendulum_widget.setYRange(-1, 1)
        self.pendulum_widget.setXRange(-1, 1)

        self.pendulum_widget.setLabel("left", "y [m]")
        self.pendulum_widget.setLabel("bottom", "x [m]")

        # Add the plotting widget to the QtWidget layout
        widget_layout = QtWidgets.QHBoxLayout(self)
        widget_layout.addWidget(self.pendulum_widget)

        self.trace = self.pendulum_widget.plot()
        self.trace.setPen(pg.mkPen(color=(255, 135, 135), width=2))

        self.stick = self.pendulum_widget.plot()
        self.stick.setPen(pg.mkPen(color=(0, 0, 0), width=2))

        self.ball = self.pendulum_widget.plot([], [], symbol='o', symbolSize=15, symbolBrush=(0, 0, 0))



        # Phase space animation
        # Create PyQTGraph plotting widget instance and configure layout
        self.phase_widget = pg.PlotWidget()
        self.phase_widget.setYRange(-2*np.pi, 2*np.pi)
        self.phase_widget.setXRange(-np.pi, np.pi)

        self.phase_widget.setLabel("left", "<font>&omega;</font> [deg/s]")
        self.phase_widget.setLabel("bottom", "<font>&theta;</font> [deg]")
        self.phase_widget.setTitle("Phase Space (<font>&theta;, &omega;</font>)")

        # Add the plotting widget to the QtWidget layout
        widget_layout.addWidget(self.phase_widget)

        self.phase_space_trace = self.phase_widget.plot()
        self.phase_space_trace.setPen(pg.mkPen(color=(255, 135, 135), width=2))

        self.phase_space_current = self.phase_widget.plot([], [], symbol='o', symbolSize=15, symbolBrush=(0, 0, 0))


        self._timer = QtCore.QTimer(self, timeout=self.update_plot)
        self._timer.setInterval(interval)
        self._timer.start()


    def update_plot(self):
        # Update the data using the pendulum class
        self.pendulum.iterate()

        # Update the graph for the pendulum animation
        self.trace.setData(self.pendulum.x, self.pendulum.y)
        self.ball.setData([self.pendulum.x[-1]], [self.pendulum.y[-1]])
        self.stick.setData([0, self.pendulum.x[-1]], [0, self.pendulum.y[-1]])
        self.pendulum_widget.setTitle(f"t = {self.pendulum.time[-1]:.2f}s")

        # Update Phase space animation
        self.phase_space_trace.setData(self.pendulum.theta, self.pendulum.omega)
        self.phase_space_current.setData([self.pendulum.theta[-1]], [self.pendulum.omega[-1]])




def phase_space():
    h = 0.001                # [s] Integration step
    t0 = 0                  # [s] Starting time
    tf = 100                  # [s] End time
    g = 9.81                # [m/s^2] Gravitational acceleration
    L = 1                   # [m] Pendulum length
    theta_0 = np.pi/2       # ['] Pendulum Starting position 
    omega_0 = np.pi/10      # [rad/s] Pendulum starting speed

    # Pre-initialize matrices
    time = np.linspace(t0, tf, int((tf-t0)/h))
    theta = np.zeros(np.shape(time))
    omega = np.zeros(np.shape(time))
    
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

def main():
    # Simulation parameters
    h = 0.001   # [s] Integration time step-size
    t0 = 0      # [s] Starting time
    g = 9.81    # [m/s^2] Graviational acceleration
    L = 1       # [m] Length of pendulum arm

    # Initial conditions
    theta_0 = np.pi/2
    omega_0 = -np.pi/2

    p = Pendulum(h, g, L, theta_0, omega_0, t0, num_shown_points=1000)

    app = QtWidgets.QApplication(sys.argv)
    gui = pendulum_gui(p, interval=1)
    gui.show()

    # # phase_space()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()



