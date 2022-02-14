
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
    def __init__(self, h, g, L, theta_0, omega_0, t0=0):
        self.h = h
        self.L = L
        self.g = g

        # Set variables to initial conditions
        self.time = t0
        self.theta = theta_0
        self.omega = omega_0
        self.x, self.y = self.cartesian()

        self.cartesian()

    def cartesian(self):
        # Translate to cartesion coordinates for plotting
        return (self.L*np.sin(self.theta), -self.L*np.cos(self.theta))

    def iterate(self):
        # Implementation of 2nd order Runga-Kutta integration of simple pendulum equation
        # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
        # \frac{d\theta}{dt} = \omega
        # \frac{d\omega}{dt} = -\frac{g}{L}sin(\theta)

        self.time += self.h

        # Estimation of function slope (derivative) at previous time-step previous guessed value
        k1_theta = self.omega
        # Estimation of function value half an integration step (h/2) later
        theta_1 = self.theta + k1_theta*self.h/2

        # Estimation of function slope (derivative) at previous time-step previous guessed value
        k1_omega = - (self.g/self.L)*np.sin(self.theta)
        # Estimation of function value half an integration step (h/2) later
        omega_1 = self.omega + k1_omega*self.h/2

        # Estimation of function slope at time t0 + (h/2)
        k2_theta = omega_1
        # Function estimation using slope k2
        self.theta += k2_theta*self.h

        # Estimation of function slope at time t0 + (h/2)
        k2_omega = -(self.g/self.L)*np.sin(theta_1)
        # Function estimation using slope k2
        self.omega += k2_omega*self.h

        return (self.omega, self.theta)

        
class pendulum_gui(QtWidgets.QWidget):
    def __init__(self, Pendulum, shown_points = 100, interval = None, iterations=1):
        # Initialize using built-in QWidget __init__ from inherited class
        super().__init__()

        # Number of iterations before a the plot is updated
        self.iterations = iterations

        # Set pendulum class in this class
        self.Pendulum = Pendulum

        # Get initial phase-space (theta, omega) coordinates from current coordinates in Pendulum class
        self.theta = np.ones(shown_points)*self.Pendulum.theta
        self.omega = np.ones(shown_points)*self.Pendulum.omega


        # Get inital (x,y) coordinates from current coordinates in Pendulum class
        x0, y0 = self.Pendulum.cartesian()

        # Initialize data matrices
        self.x = np.ones(shown_points)*x0
        self.y = np.ones(shown_points)*y0



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

        # Interval = 0 gives computation-limited simulation speed
        if interval == None:
            # Set wait-time to be equal to the actual travel time of the pendulum
            interval = self.iterations*self.Pendulum.h*1000
            

        self._timer = QtCore.QTimer(self, timeout=self.update_plot)
        self._timer.setInterval(interval)
        self._timer.start()


    def update_plot(self):
        # Update the data using the pendulum class
        for i in range(self.iterations):
            self.Pendulum.iterate()

        # Update (x,y) coordinates by shifting registers down
        self.x[:-1] = self.x[1:]
        self.y[:-1] = self.y[1:]
        # Add new (x,y) coordinates as last entry
        self.x[-1], self.y[-1] = self.Pendulum.cartesian()

        # Update (theta, omega) coordinates by shifting registers down
        self.theta[:-1] = self.theta[1:]
        self.omega[:-1] = self.omega[1:]
        # Add new (theta,omega) coordinates as last entry
        self.theta[-1] = self.Pendulum.theta
        self.omega[-1] = self.Pendulum.omega

        # Set current time as the pendulum simulation title
        self.pendulum_widget.setTitle(f"t = {self.Pendulum.time:.2f}s")

        # Update the graph for the pendulum animation
        self.trace.setData(self.x, self.y)
        self.ball.setData([self.x[-1]], [self.y[-1]])
        self.stick.setData([0, self.x[-1]], [0, self.y[-1]])

        # Update Phase space animation
        self.phase_space_trace.setData(self.theta, self.omega)
        self.phase_space_current.setData([self.theta[-1]], [self.omega[-1]])




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

    p = Pendulum(h, g, L, theta_0, omega_0, t0)

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

def main():
    # Simulation parameters
    h = 0.001   # [s] Integration time step-size
    t0 = 5      # [s] Starting time
    g = 9.81    # [m/s^2] Graviational acceleration
    L = 1       # [m] Length of pendulum arm

    # Initial conditions
    theta_0 = np.pi/2
    omega_0 = -np.pi/2

    p = Pendulum(h, g, L, theta_0, omega_0, t0)

    app = QtWidgets.QApplication(sys.argv)
    gui = pendulum_gui(p, shown_points=100, iterations=5)
    gui.show()

    # phase_space()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()



