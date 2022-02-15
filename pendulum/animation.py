# Native libraries
import sys

# External Libraries
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from pendulum.pendulum import Pendulum2D, Pendulum3D

class Pendulum2DGui(QtWidgets.QWidget):
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
            interval = int(self.iterations*self.Pendulum.h*1000)
            

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


class Pendulum3DGui(QtWidgets.QWidget):
    def __init__(self, Pendulum, shown_points = 100, interval = None, iterations=1):
        # Initialize using built-in QWidget __init__ from inherited class
        super().__init__()

        # Number of iterations before a the plot is updated
        self.iterations = iterations

        # Set pendulum class in this class
        self.Pendulum = Pendulum

        # Get initial phase-space (theta, omega) coordinates from current coordinates in Pendulum class
        self.theta = np.ones(shown_points)*self.Pendulum.theta
        self.phi = np.ones(shown_points)*self.Pendulum.phi

        self.omega = np.ones(shown_points)*self.Pendulum.omega
        self.sigma = np.ones(shown_points)*self.Pendulum.sigma

        # Get inital (x,y) coordinates from current coordinates in Pendulum class
        x0, y0, z0 = self.Pendulum.cartesian()

        # Initialize data matrices
        self.coordinates = np.ones((shown_points,3))
        self.coordinates[:,0] *= x0
        self.coordinates[:,1] *= y0
        self.coordinates[:,2] *= z0

        # # Set up PyQtGraph styling configurations        
        # pg.setConfigOption('background', 0.95)
        # pg.setConfigOptions(antialias=True)
        
        # Swinging pendulum animation
        # https://pyqtgraph.readthedocs.io/en/latest/3dgraphics/
        # Create PyQTGraph plotting widget instance and configure layout
        self.plot_widget = gl.GLViewWidget()
        self.plot_widget.show()
        self.grid_widget = gl.GLGridItem()
        self.grid_widget.setSize(3*self.Pendulum.L, 3*self.Pendulum.L, 3*self.Pendulum.L)
        self.grid_widget.setSpacing(0.1, 0.1, 0.1)
        self.plot_widget.addItem(self.grid_widget)


        self.trace = gl.GLLinePlotItem(width=5, color=(1.0, 0.5, 0.5, 1.0), antialias=True)
        self.plot_widget.addItem(self.trace)
        
        self.stick = gl.GLLinePlotItem(width=3, color=(1.0, 1.0, 1.0, 1.0), antialias=True)
        self.plot_widget.addItem(self.stick)
        
        self.ball = gl.GLScatterPlotItem(size=15, color=(1.0, 0.5, 0.5, 1.0))
        self.plot_widget.addItem(self.ball)

        self.shadow = gl.GLLinePlotItem(width=5, color=(1.0, 1.0, 1.0, 0.5), antialias=True)
        self.plot_widget.addItem(self.shadow)

        # self.projection_y = gl.GLLinePlotItem(width=5, color=(1.0, 0, 1.0, 0.5), antialias=True)
        # self.plot_widget.addItem(self.projection_y)

        # self.projection_x = gl.GLLinePlotItem(width=5, color=(1.0, 1.0, 0, 0.5), antialias=True)
        # self.plot_widget.addItem(self.projection_x)

        # # Phase space animation
        # # Create PyQTGraph plotting widget instance and configure layout
        # self.phase_widget = pg.PlotWidget()
        # self.phase_widget.setYRange(-2*np.pi, 2*np.pi)
        # self.phase_widget.setXRange(-np.pi, np.pi)

        # self.phase_widget.setLabel("left", "<font>&omega;</font> [deg/s]")
        # self.phase_widget.setLabel("bottom", "<font>&theta;</font> [deg]")
        # self.phase_widget.setTitle("Phase Space (<font>&theta;, &omega;</font>)")

        # # Add the plotting widget to the QtWidget layout
        # widget_layout.addWidget(self.phase_widget)

        # self.phase_space_trace = self.phase_widget.plot()
        # self.phase_space_trace.setPen(pg.mkPen(color=(255, 135, 135), width=2))

        # self.phase_space_current = self.phase_widget.plot([], [], symbol='o', symbolSize=15, symbolBrush=(0, 0, 0))

        # Interval = 0 gives computation-limited simulation speed
        if interval == None:
            # Set wait-time to be equal to the actual travel time of the pendulum
            interval = int(self.iterations*self.Pendulum.h*1000)
            

        self._timer = QtCore.QTimer(self, timeout=self.update_plot)
        self._timer.setInterval(interval)
        self._timer.start()


    def update_plot(self):
        # Update the data using the pendulum class
        for i in range(self.iterations):
            self.Pendulum.iterate()

        # Update (x,y) coordinates by shifting registers down
        self.coordinates[:-1,:] = self.coordinates[1:,:]

        # Add new (x,y) coordinates as last entry
        self.coordinates[-1,:] = self.Pendulum.cartesian()

        # Update (theta, omega) coordinates by shifting registers down
        self.theta[:-1] = self.theta[1:]
        self.omega[:-1] = self.omega[1:]

        # Add new (theta,omega) coordinates as last entry
        self.theta[-1] = self.Pendulum.theta
        self.omega[-1] = self.Pendulum.omega

        # # Set current time as the pendulum simulation title
        # self.plot_widget.setTitle(f"t = {self.Pendulum.time:.2f}s")

        # # Update the graph for the pendulum animation

        self.trace.setData(pos=self.coordinates)

        stick_data = np.array([
            [0, 0, self.Pendulum.L],
            self.coordinates[-1,:]
        ])

        self.stick.setData(pos = stick_data)

        self.ball.setData(pos = self.coordinates[-1,:])

        shadow_mask = np.ones(np.shape(self.coordinates))
        shadow_mask[:,2] *= 0

        self.shadow.setData(pos=self.coordinates*shadow_mask)

        # projection_x = np.ones(np.shape(self.coordinates))*self.Pendulum.L
        # projection_x[:,[1,2]] = self.coordinates[:,[1,2]]
        # self.projection_x.setData(pos=projection_x)

        # projection_y = np.ones(np.shape(self.coordinates))*self.Pendulum.L
        # projection_y[:,[0,2]] = self.coordinates[:,[0,2]]
        # self.projection_y.setData(pos=projection_y)

        # # Update Phase space animation
        # self.phase_space_trace.setData(self.theta, self.omega)
        # self.phase_space_current.setData([self.theta[-1]], [self.omega[-1]])


def main():
    # 3D PENDULUM ANIMATION
    # Simulation parameters
    h = 0.001   # [s] Integration time step-size
    t0 = 0      # [s] Starting time
    g = 9.81    # [m/s^2] Graviational acceleration
    L = 1       # [m] Length of pendulum arm
    m = 1       # [kg] Mass of ball

    # Initial conditions
    theta_0 = np.pi/2       # [rad] Initial polar angle
    phi_0 = 0               # [rad] Initial azimuthal angle

    omega_0 = -np.pi/2     # [rad/s] Initial polar angular velocity
    sigma_0 = np.pi/2      # [rad/s] Initial azimuthal angular velocity 

    app = QtWidgets.QApplication(sys.argv)

    p_3D = Pendulum3D(h, g, L, m, theta_0, phi_0, omega_0, sigma_0, t0)
    gui_3D = Pendulum3DGui(p_3D, shown_points=5000, iterations=10)


    p_2D = Pendulum2D(h, g, L, theta_0, omega_0, t0)
    gui_2D = Pendulum2DGui(p_2D, shown_points=1000, iterations=5)
    gui_2D.show()


    sys.exit(app.exec_())

if __name__ == '__main__':
    main()