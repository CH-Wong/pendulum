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

from pendulum.pendulum import Pendulum2D

# Mathematics here: https://en.wikipedia.org/wiki/Spherical_pendulum
       
class pendulum_gui(QtWidgets.QWidget):
    def __init__(self, Pendulum, shown_points = 100, interval = None, iterations=1):
        # Initialize using built-in QWidget __init__ from inherited class
        super().__init__()

        # Number of iterations before a the plot is updated
        self.iterations = iterations

        # Set pendulum class in this class
        self.Pendulum = Pendulum2D

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

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()



