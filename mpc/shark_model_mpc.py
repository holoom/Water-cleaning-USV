#!/usr/bin/env python3
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, fabs, inv, types
import numpy as np
from scipy.linalg import block_diag
import math

def shark_model_mpc()-> AcadosModel:
    model = types.SimpleNamespace()
    # states
    x = SX.sym('x')                 # earth position x
    y = SX.sym('y')                 # earth position y
    psi = SX.sym('psi')             # yaw angle
    u = SX.sym('u')                 # earth velocity x
    v = SX.sym('v')                 # earth velocity y
    r = SX.sym('r')                 # yaw velocity
    ml = SX.sym('ml')           # motor left
    mr = SX.sym('mr')           # motor right
    sym_x = vertcat(x,y,psi,u,v,r,ml,mr)

    # controls
    u1 = SX.sym('u1')               # Left propeller shaft speed (rad/s)
    u2 = SX.sym('u2')               # Right propeller shaft speed (rad/s)
   
    sym_u = vertcat(u1,u2)

    # Parameters for disturbances
    disturbance_x = SX.sym('disturbance_x')
    disturbance_y = SX.sym('disturbance_y')
    disturbance_psi = SX.sym('disturbance_n')

    # Parameters for added mass
    added_mass_x = SX.sym('added_mass_x')
    added_mass_y = SX.sym('added_mass_y')
    added_mass_psi = SX.sym('added_mass_psi')

    # Parameters for linear drag
    linear_d_x = SX.sym('linear_d_x')
    linear_d_y = SX.sym('linear_d_y')
    linear_d_psi = SX.sym('linear_d_psi')

    # Parameters for nonlinear drag
    nonlinear_d_x = SX.sym('nonlinear_d_x')
    nonlinear_d_y = SX.sym('nonlinear_d_y')
    nonlinear_d_psi = SX.sym('nonlinear_d_psi')

    
    sym_p = vertcat(
        added_mass_x, added_mass_y, added_mass_psi,
        linear_d_x, linear_d_y, linear_d_psi,
        nonlinear_d_x, nonlinear_d_y, nonlinear_d_psi,
        disturbance_x, disturbance_y, disturbance_psi
    )


    # xdot for f_impl
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot = SX.sym('u_dot')
    v_dot = SX.sym('v_dot')
    r_dot = SX.sym('r_dot')
    ml_dot = SX.sym('ml_dot')
    mr_dot = SX.sym('mr_dot')
    sym_xdot = vertcat(x_dot,y_dot,psi_dot,u_dot,v_dot,r_dot,ml_dot,mr_dot)

    # algebraic variables
    z = vertcat([])

    # system parameters
    m = 55                                      # mass
    Iz = 15.10
    ZG = 0
    g = 9.81

    # Experimental propeller data including lever arms
    y_pont = 0.395      # distance from centerline to waterline centroid (m)
    Cw_pont = 0.75      # waterline area coefficient (-)
    Cb_pont = 0.4       # block coefficient, computed from m = 55 kg
    l_motor_left = -y_pont  # lever arm, left propeller (m)
    l_motor_right = y_pont  # lever arm, right propeller (m)
    k_pos = 0.02216 / 2  # Positive Bollard, one propeller
    k_neg = 0.01289 / 2  # Negative Bollard, one propeller
    n_max = math.sqrt((0.5 * 24.4 *  g) /  k_pos)  # max. prop. rev.
    n_min = -math.sqrt((0.5 * 13.6 *  g) /  k_neg) # min. prop. rev.


    # Propeller configuration/input matrix
    # B = np.array([
    #         [50, 50],
    #         [0, 0],
    #         [-0.39 * 50, 0.39 * 50]
    #     ])
    # B = k_pos * np.array([[1, 1], [-l1, -l2]])
    # u_alloc = np.array([fabs(ml) * ml ,fabs(mr) * mr])
    # B = np.array([[1, 1], [-l1, -l2]]) 
    # u_alloc = np.array([u1 ,u2])
    # tau = np.matmul(B, u_alloc)

    tau_x = ml + mr
    tau_y = 0
    tau_r = -(l_motor_left * ml + l_motor_right * mr)
    

    # dynamics
    dx = u*cos(psi) - v*sin(psi) 
    dy = u*sin(psi) + v*cos(psi)
    dpsi = r

    du = (tau_x + m*r*v  +added_mass_y*r*v + linear_d_x*u + nonlinear_d_x*fabs(u)*u + disturbance_x)/(m + added_mass_x)
    dv = (tau_y - m*r*u - added_mass_x *r*u + linear_d_y*v + nonlinear_d_y*fabs(v)*v + disturbance_y)/(m + added_mass_y)
    
    dr = (tau_r - added_mass_y*u*v + added_mass_x*u*v + linear_d_psi*r + nonlinear_d_psi*fabs(r)*r + disturbance_psi)/(Iz+added_mass_psi)


    dml = u1
    dmr = u2

    f_expl = vertcat(dx,dy,dpsi,du,dv,dr,dml,dmr)
    f_impl = sym_xdot - f_expl
  
    # constraints
    # Model bounds
    model.u_min = -3
    model.u_max = 3

    # state bounds
    model.thrust_min = -100
    model.thrust_max = 100

    model.r_min = -2 # minimum angular velocity [rad/s]
    model.r_max = 2  # maximum angular velocity [rad/s]

    # input bounds
    model.dF_min = -30 # minimum throttle change rate
    model.dF_max = 30 # maximum throttle change rate
    
    # Define initial conditions
    model.x0 = np.array([0.001, 00.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])


    # nonlinear least sqares
    
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = sym_x
    model.xdot = sym_xdot
    model.u = sym_u
    model.p = sym_p
    model.z = z
    model.name = "shark_model"


    return model




