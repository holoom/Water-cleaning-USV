from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, diag, inv, DM, MX, fabs
import numpy as np
from scipy.linalg import block_diag
import math

def export_mhe_model_with_bias_body(params):
    model_name = 'otter_mhe_model_with_bias_b'
    
    
    # states
    x = SX.sym('x')                 # earth position x
    y = SX.sym('y')                 # earth position y
    psi = SX.sym('psi')             # yaw angle
    u = SX.sym('u')                 # earth velocity x
    v = SX.sym('v')                 # earth velocity y
    r = SX.sym('r')                 # yaw velocity
    # add parameter  as state
    b_x = SX.sym('b_x')
    b_y = SX.sym ('b_y')
    b_psi = SX.sym('b_psi')
    x = vertcat(x, y, psi, u, v, r, b_x, b_y, b_psi)
    nx_augmented = x.size()[0]
    nx = 6

    # state noise
    w_x = SX.sym('w_x')
    w_y = SX.sym('w_y')
    w_psi = SX.sym('w_psi')
    w_u = SX.sym('w_u')
    w_v = SX.sym('w_v')
    w_r = SX.sym('w_r')
    w = vertcat(w_x, w_y, w_psi, w_u, w_v, w_r)
    nw = w.size()[0]
    # controls
    ml = SX.sym('ml')
    mr = SX.sym('mr')               
    F = vertcat(ml, mr)
    p = F
    nu = u.size()[0]
    


    # xdot
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot = SX.sym('u_dot')
    v_dot = SX.sym('v_dot')
    r_dot = SX.sym('r_dot')
    b_x_dot = SX.sym('b_x_dot')
    b_y_dot = SX.sym('b_y_dot')
    b_psi_dot = SX.sym('b_psi_dot')
    xdot = vertcat(x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot, b_x_dot, b_y_dot, b_psi_dot)
    
    # algebraic variables
    z = []

    ### known system parameters
    # Parameters for added mass
    added_mass_x= params.get('added_mass_x', 0)
    added_mass_y = params.get('added_mass_y', 0)
    added_mass_psi = params.get('added_mass_psi', 0)

    # Parameters for linear drag
    linear_d_x = params.get('linear_d_x', 0)
    linear_d_y = params.get('linear_d_y', 0)
    linear_d_psi = params.get('linear_d_psi', 0)

    # Parameters for nonlinear drag
    nonlinear_d_x = params.get('nonlinear_d_x', 0)
    nonlinear_d_y = params.get('nonlinear_d_y', 0)
    nonlinear_d_psi = params.get('nonlinear_d_psi', 0)

    # system parameters
    m = 85                                      # mass
    Iz = 15
    ZG = 0
    g = 9.81

    # Experimental propeller data including lever arms
    y_pont = 0.395     # distance from centerline to waterline centroid (m)

    l_motor_left = -y_pont  # lever arm, left propeller (m)
    l_motor_right = y_pont  # lever arm, right propeller (m)



    tau_x = ml + mr
    tau_y = 0
    tau_r = -(l_motor_left * ml + l_motor_right * mr)
    

    # dynamics
    dx = u*cos(psi) - v*sin(psi) 
    dy = u*sin(psi) + v*cos(psi)
    dpsi = r

    du = (tau_x + m*r*v  +added_mass_y*r*v + linear_d_x*u + nonlinear_d_x*fabs(u)*u + b_x)/(m + added_mass_x)
    dv = (tau_y - m*r*u - added_mass_x *r*u + linear_d_y*v + nonlinear_d_y*fabs(v)*v + b_y)/(m + added_mass_y)
    
    dr = (tau_r - added_mass_y*u*v + added_mass_x*u*v + linear_d_psi*r + nonlinear_d_psi*fabs(r)*r + b_psi)/(Iz+added_mass_psi)

    

    f_expl = vertcat(dx,
                     dy,
                     dpsi,
                     du, 
                     dv,
                     dr,
                     0,
                     0,
                     0) 
  
    # constraints
    # Model bounds
    # model.u_min = -0.5
    # model.u_max = 2

    # # state bounds
    # model.thrust_min = -30
    # model.thrust_max = 100

    # model.r_min = -1.5 # minimum angular velocity [rad/s]
    # model.r_max = 1.5  # maximum angular velocity [rad/s]

    # # input bounds
    # model.dF_min = -30 # minimum throttle change rate
    # model.dF_max = 30 # maximum throttle change rate
    
    # add additive state noise
    f_expl[:nx] = f_expl[:nx] + w
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = w
    model.z = z
    model.p = p
    model.name = model_name

    return model

def main():
    params = {

        'added_mass_x':  5.5,
        'added_mass_y':  82.5,
        'added_mass_psi': 25.67,
        'linear_d_x': -77.55,
        'linear_d_y': -162.5,
        'linear_d_psi': -42.65,
        'nonlinear_d_x':  0,
        'nonlinear_d_y':0,
        'nonlinear_d_psi': 0,
        'disturbance_x': 0,
        'disturbance_y':  0,
        'disturbance_psi':  0
    }
    model = export_mhe_model_with_bias_body(params)
    print("Model name:", model.name)
    print("State vector:", model.x)
    print("Control:", model.p)
    print("F_impl expression:", model.f_impl_expr)
    print("F_expl expression:", model.f_expl_expr)



if __name__ == '__main__':
    main()