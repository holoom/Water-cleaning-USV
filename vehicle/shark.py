import numpy as np
class Shark:
    """
    Shark class to simulate the dynamics of an underwater vehicle with adjustable parameters.
    """
    def __init__(self, V_current=0, beta_current=0):
        
        D2R = np.pi/180
        self.V_c = V_current
        self.beta_c = beta_current * D2R

        self.m = 85  # mass in kg
        self.max_thrust = 33.5
        self.min_thrust = 7.5
        self.d_motor_left = -0.43  # Distance from port thruster to center line in m
        self.d_motor_right = 0.43  # Distance from starboard thruster to center line in m
        self.l = 0.43 * 2
        self.rho = 1000  # Density of water in kg/m^3

        # Maximum and drag coefficients
        Max_surge = 0.69  # maximum surge speed in m/s
        Max_yaw = 0.22  # maximum yaw rate in rad/s
        Max_sway = -0.0221  # maximum sway speed in m/s

        # Initial dynamic parameters
        self.Xudot = 300
        self.Yvdot = 150
        self.Nrdot_Iz = 325
        
        # self.Xu = 140

        self.Xu = (self.max_thrust * 2) / Max_surge
        self.Yv = -(Max_yaw * 0.3) * (self.m + 100) / Max_sway
        self.Nr = self.l * self.min_thrust / Max_yaw 
        
        self.Xuu = 0
        self.Yvv = 0
        self.Nrr = 0

    def dynamics(self, x, u):
        # xs, ys, phis, us, vs, rs, ml, mr = x
        xs = x[0]
        ys = x[1]
        phis = x[5]
        us = x[6]
        vs= x[7]
        rs = x[11]

        nu = np.array([us, vs, rs], float)
        ml, mr = u
        # F_p, F_s = u
        # Current velocities
        u_c = self.V_c * np.cos(self.beta_c - phis)  # current surge vel.
        v_c = self.V_c * np.sin(self.beta_c - phis)  # current sway vel.

        nu_c = np.array([u_c, v_c, 0], float)  # current velocity vector
        Dnu_c = np.array([rs*v_c, -rs*u_c, 0],float) # derivative
        nu_r = nu - nu_c  # relative velocity vector
        us_r, vs_r , rs_r = nu_r 
        dxs = us * np.cos(phis) - vs * np.sin(phis)
        dys = us * np.sin(phis) + vs * np.cos(phis)
        dphis = rs
        dus = Dnu_c[0] + ((ml + mr) + (self.m * vs_r * rs) + (self.Yvdot * rs * vs_r) -
               (self.Xu * us_r) - (self.Xuu * abs(us_r) * us_r)) / (self.m + self.Xudot)  
        dvs = Dnu_c[1] + (-(self.m * us_r * rs) - (self.Yv * vs_r) - (self.Yvv * abs(vs_r) * vs_r)) / (self.m + self.Yvdot) 
        drs = (-(self.d_motor_left * ml + self.d_motor_right * mr) -
               (self.Nr * rs) - (self.Nrr * abs(rs) * rs)) / self.Nrdot_Iz

        xdot = np.array([dxs, dys,0,0,0, dphis, dus, dvs,0,0,0, drs])
        return xdot
    def Smatrx (self, a):

        return np.array([
            [0, -a[2], a[1]], 
            [a[2], 0, -a[0]], 
            [-a[1], a[0], 0]
        ])


    def step(self, st, thrust, dt):
        k1 = self.dynamics(st, thrust)
        k2 = self.dynamics(st + 0.5 * dt * k1, thrust)
        k3 = self.dynamics(st + 0.5 * dt * k2, thrust)
        k4 = self.dynamics(st + dt * k3, thrust)
        st_next = st + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return st_next

    def add_trash(self, trash_properties):
        # Directly update dynamics parameters with new values provided in trash_properties
        # self.m = trash_properties.get('m')
        self.Xudot += trash_properties.get('Xudot')
        # self.Yvdot = trash_properties.get('Yvdot')
        # self.Nrdot_Iz = trash_properties.get('Nrdot_Iz')
        self.Xu += trash_properties.get('Xu')
        # self.Yv = trash_properties.get('Yv')
        # self.Nr = trash_properties.get('Nr')
    def thrust_allocation (self, tau_x, tau_r):
        B = np.array([[1, 1],           
                    [0, 0],   
                    [- self.d_motor_left, - self.d_motor_right]])

        tau = np.array([tau_x[0], 0, tau_r[0]])
        B_inv = np.linalg.pinv(B)
        Thrust = B_inv @ tau
        Thrust = np.clip(Thrust, -self.min_thrust, self.max_thrust)
        return Thrust 
    def ssa(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi