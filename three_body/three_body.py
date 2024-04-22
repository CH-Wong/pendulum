# External Libraries
import numpy as np

class ThreeBody2D():
    # G = 6.6743e-11  # [m^3 kg^–1 s^–2] Gravitational constant
    G = 1

    def __init__(self, m1, m2, m3, r1_0, r2_0, r3_0, v1_0, v2_0, v3_0, t0=0, h=0.001):
        self.h = h  # [s] Integration time-step, default = 0.001

        # Store user-given masses of each particle
        self.m1 = m1        # [kg] Mass of particle 1
        self.m2 = m2        # [kg] Mass of particle 2
        self.m3 = m3        # [kg] Mass of particle 3

        # Store all user-given initial conditions to class variables to be used during the computation loop
        # Initial time
        self.time = t0      # [s] Starting time, default = 0
 
        # Initial positions-vectors for all particles i (i.e. 1, 2 and 3)
        self.r1 = r1_0      # [m] Initial position of particle 1
        self.r2 = r2_0      # [m] Initial position of particle 2
        self.r3 = r3_0      # [m] Initial position of particle 3

        # Initial velocity-vectors for all particles i (i.e. 1, 2, and 3)
        self.v1 = v1_0      # [m/s] Initial velocity of particle 1
        self.v2 = v2_0      # [m/s] Initial velocity of particle 2
        self.v3 = v3_0      # [m/s] Initial velocity of particle 3

    def phase_space(self):
        return None

    def gravitational_accelaration(self, r0, r1, m1, r2, m2):
        return -self.G*m1*(r0-r1)/np.linalg.norm(r0-r1)**3 - self.G*m2*(r0-r2)/np.linalg.norm(r0-r2)**3

    def iterate(self):
        # Implementation of 4th order Runga-Kutta integration of three-body problem
        # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
        # https://en.wikipedia.org/wiki/Three-body_problem?
        self.time += self.h

        # Estimate function slope k1 at time t0 for all coordinates using known initial conditions, 
        # or previously guessed coordinates r_i and v_i for all particles i 
        
        k1_r1 = self.v1
        k1_v1 = self.gravitational_accelaration(self.r1, self.r2, self.m2, self.r3, self.m3)
        
        k1_r2 = self.v2
        k1_v2 = self.gravitational_accelaration(self.r2, self.r1, self.m1, self.r3, self.m3)

        k1_r3 = self.v3
        k1_v3 = self.gravitational_accelaration(self.r3, self.r1, self.m1, self.r2, self.m2)


        self.r1 += k1_r1*self.h
        self.v1 += k1_v1*self.h

        self.r2 += k1_r2*self.h
        self.v2 += k1_v2*self.h

        self.r3 += k1_r3*self.h
        self.v3 += k1_v3*self.h


        # k1_omega = - (self.g/self.L)*np.sin(self.theta)
        
        # # Use the estimated slopes k1 to guess where the next point will be after timestep dt = h
        # theta_1 = self.theta + k1_theta*self.h/2
        # omega_1 = self.omega + k1_omega*self.h/2

        # # Use first estimate (theta_1, omega_1) to repeat the process
        # k2_theta = omega_1
        # k2_omega = - (self.g/self.L)*np.sin(theta_1)

        # # Now use slope k2 to make another estimate, more accurate than (theta_1, omega_1)
        # theta_2 = self.theta + k2_theta*self.h/2
        # omega_2 = self.omega + k2_omega*self.h/2

        # # Use first estimate (theta_1, omega_1) to repeat the process
        # k4_theta = omega_2
        # k4_omega = - (self.g/self.L)*np.sin(theta_2)

        # # Now use slope k2 to make another estimate, more accurate than (theta_1, omega_1)
        # theta_3 = self.theta + k4_theta*self.h/2
        # omega_3 = self.omega + k4_omega*self.h/2

        # # Use first estimate (theta_1, omega_1) to repeat the process
        # k4_theta = omega_3
        # k4_omega = - (self.g/self.L)*np.sin(theta_3)

        # # Use weighted average k1, k2, k4, k4 to estimate the actual next time point
        # self.theta += self.h*(k1_theta + 2*k2_theta + 2*k4_theta + k4_theta)/6
        # self.omega += self.h*(k1_omega + 2*k2_omega + 2*k4_omega + k4_omega)/6

        return (self.r1, self.v1), (self.r2, self.v2), (self.r3, self.v3)

class ThreeBody3D():
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
        
    def cartesian(self):
        # Translate to cartesion coordinates for plotting
        return (self.L*np.sin(self.theta)*np.cos(self.phi), self.L*np.sin(self.theta)*np.sin(self.phi), self.L*(1 - np.cos(self.theta)))

    def phase_space(self):
        # Compute conjugate momentum theta
        momentum_theta = self.m*self.L**2*self.omega
        return (self.theta, self.phi, momentum_theta)

    def iterate(self):
        # Mathematics here: https://en.wikipedia.org/wiki/Spherical_pendulum
        # Implementation of 4th order Runga-Kutta integration of simple pendulum equation
        # https://lpsa.swarthmore.edu/NumInt/NumIntFirst.html
        # \frac{d\theta}{dt} = \omega
        # \frac{d\phi}{dt} = \sigma

        # \frac{d\omega}{dt} = sin(\theta)cos(\theta)\sigma^2-\frac{g}{L}sin(\theta)
        # \frac{d\sigma}{dt} = -2*\frac{\sigma\omega}{tan(\theta)}
        
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
        k4_theta = omega_2
        k4_phi  = sigma_2

        k4_omega = np.sin(theta_2)*np.cos(theta_2)*sigma_2**2 - (self.g/self.L)*np.sin(theta_2)
        k4_sigma = -2*omega_2*sigma_2/np.tan(theta_2)

        # Now use slope k2 to make another estimate, more accurate than (theta_1, omega_1)
        theta_3 = self.theta + k4_theta*self.h/2
        phi_3 = self.phi + k4_phi*self.h/2

        omega_3 = self.omega + k4_omega*self.h/2
        sigma_3 = self.sigma + k4_sigma*self.h/2

        # Use first estimate (theta_1, omega_1) to repeat the process
        k4_theta = omega_3
        k4_phi  = sigma_3

        k4_omega = np.sin(theta_3)*np.cos(theta_3)*sigma_3**2 - (self.g/self.L)*np.sin(theta_3)
        k4_sigma = -2*omega_3*sigma_3/np.tan(theta_3)

        # Use weighted average k1, k2, k4, k4 to estimate the actual next time point
        self.theta += self.h*(k1_theta + 2*k2_theta + 2*k4_theta + k4_theta)/6
        self.phi += self.h*(k1_phi + 2*k2_phi + 2*k4_phi + k4_phi)/6
        self.omega += self.h*(k1_omega + 2*k2_omega + 2*k4_omega + k4_omega)/6
        self.sigma += self.h*(k1_sigma + 2*k2_sigma + 2*k4_sigma + k4_sigma)/6

        return (self.theta, self.phi, self.omega, self.sigma)



def main():
    # Simulation parameters
    h = 0.001   # [s] Integration time step-size
    t0 = 0      # [s] Starting time

    # Initial coordinates @ t=t0
    # Store user-given masses of each particle
    m1 = 1        # [kg] Mass of particle 1
    m2 = 1        # [kg] Mass of particle 2
    m3 = 1        # [kg] Mass of particle 3

    # Initial positions-vectors for all particles i (i.e. 1, 2 and 3)
    r1_0 = np.vector(0,0)      # [m] Initial position of particle 1
    r2_0 = np.vector(0,1)      # [m] Initial position of particle 2
    r3_0 = np.vector(1,0)      # [m] Initial position of particle 3

    # Initial velocity-vectors for all particles i (i.e. 1, 2, and 3)
    v1_0 = (0,0)      # [m/s] Initial velocity of particle 1
    v2_0 = (0,0)      # [m/s] Initial velocity of particle 2
    v3_0 = (0,0)      # [m/s] Initial velocity of particle 3

    B = ThreeBody2D(m1, m2, m3, r1_0, r2_0, r3_0, v1_0, v2_0, v3_0, t0, h)

    return B


if __name__ == '__main__':
    main()