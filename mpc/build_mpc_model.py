from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
import casadi
#from utils import plot_pendulum
import math
from scipy.linalg import block_diag
import os

from shark_model_mpc import *


def build_mpc_model(h, N):
    ocp = AcadosOcp()
    model = shark_model_mpc()


    # Set model
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # set dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_0 = nx + nu
    ny_e = nx  # no action at the final state
    nz = 0
    nparam = model.p.size()[0]

    ocp.dims.N = N
    # Set parameters
    ocp.parameter_values = np.zeros((nparam, ))
    
    # set cost
    # Q = np.diag([1e3, 1e3, 0, 1e1, 1e-3, 1e1, 1e-1, 1e-1])
    # Q = np.diag([1e5, 1e5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    
    # R = np.eye(nu)
    # R[0, 0] = 1e-2
    # R[1, 1] = 1e-2

    # Qe = np.diag([ 5e5, 5e5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])

    Q = np.diag([10, 10, 10, 1e-3, 1e-3, 1e-3, 1e-5, 1e-5])
    
    R = np.eye(nu)
    R[0, 0] = 1e-4
    R[1, 1] = 1e-4

    Qe = np.diag([ 5e5, 5e5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.cost_type_0 = "LINEAR_LS"

    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_e = Qe
    ocp.cost.W_0 = block_diag(Q, R)

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx
    ocp.cost.Vx_0 = Vx

    Vu = np.zeros((ny, nu))
    Vu[8, 0] = 1.0
    Vu[9, 1] = 1.0
    ocp.cost.Vu = Vu
    ocp.cost.Vu_0 = Vu
    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    '''ocp.cost.zl = 0 * np.ones((ns,)) #previously 100
    ocp.cost.Zl = 0 * np.ones((ns,))
    ocp.cost.zu = 0 * np.ones((ns,)) #previously 100
    ocp.cost.Zu = 0 * np.ones((ns,))'''

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)
    ocp.cost.yref_0 = np.zeros(ny_0)

    # setting constraints
    ocp.constraints.lbx = np.array([model.u_min, model.u_min, model.r_min, model.thrust_min, model.thrust_min])
    ocp.constraints.ubx = np.array([model.u_max, model.u_max, model.r_max, model.thrust_max, model.thrust_max])
    ocp.constraints.idxbx = np.array([3,4,5,6,7])
    
    # ocp.constraints.lbx = np.array([model.thrust_min, model.thrust_min])
    # ocp.constraints.ubx = np.array([model.thrust_max, model.thrust_max])
    # ocp.constraints.idxbx = np.array([6,7])
    ocp.constraints.lbu = np.array([model.dF_min, model.dF_min])
    ocp.constraints.ubu = np.array([model.dF_max, model.dF_max])
    ocp.constraints.idxbu = np.array([0, 1])


    # ocp.constraints.lsbx=np.zero s([1])
    # ocp.constraints.usbx=np.zeros([1])
    # ocp.constraints.idxsbx=np.array([1])
    '''ocp.constraints.lh = np.array(
        [
            model.u_min,
            model.u_min,
            model.r_min,
            model.Tport_min,
            model.Tstbd_min,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            model.u_max,
            model.u_max,
            model.r_max,
            model.Tport_max,
            model.Tstbd_max,
        ]
    )'''
    '''ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array([0, 2])'''

    # set intial condition
    ocp.constraints.x0 = model.x0

    ## set QP solver
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_max_iter = 100
    cost = 1e-6
    ocp.solver_options.__nlp_solver_tol_stat = cost
    ocp.solver_options.nlp_solver_tol_eq = cost
    ocp.solver_options.__nlp_solver_tol_ineq = cost
    ocp.solver_options.__nlp_solver_tol_comp = cost

    # set prediction horizon
    ocp.solver_options.tf = h * N
    ocp.solver_options.nlp_solver_type = 'SQP'


    # Show some info during generation
    print('Size of state vector=', nx)
    print('Size of input vector=', nu)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    export_dir = dir_path + "/c_generated_code"
    print("Saving to: " + export_dir)
    ocp.code_export_directory = export_dir
    print("Saving to: " + ocp.code_export_directory)
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_"+model.name+".json")
    print('>> NMPC exported')
    return acados_solver


if __name__ == '__main__':
    Tf = 5
    N = 50
    h = Tf/N
    acados_solver = build_mpc_model(h, N)
   
