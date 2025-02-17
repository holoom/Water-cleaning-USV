from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
import casadi as ca
import math
from scipy.linalg import block_diag
import os
import numpy as np
from scipy.linalg import block_diag
from acados_template import AcadosOcpSolver, AcadosOcp
from casadi import vertcat
from export_mhe_ode_model import export_mhe_model


def export_mhe_solver(model, N, h, Q, Q0, R):

    ocp_mhe = AcadosOcp()

    ocp_mhe.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    nparam = model.p.size()[0]

    ny = 3*nx     # h(x), w and arrival cost
    ny_e = 0

    ocp_mhe.dims.N = N

    # set cost
    ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
    ocp_mhe.cost.cost_type_e = 'LINEAR_LS'

    ocp_mhe.cost.W = block_diag(R, Q, np.zeros((nx, nx)))

    x = ocp_mhe.model.x
    u = ocp_mhe.model.u

    ocp_mhe.model.cost_y_expr = vertcat(x, u, x)

    ocp_mhe.parameter_values = np.zeros((nparam, ))

    ocp_mhe.cost.yref = np.zeros((3*nx,))
    ocp_mhe.cost.yref_e = np.zeros((0, ))

    ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp_mhe.solver_options.integrator_type = 'ERK'

    # set prediction horizon
    ocp_mhe.solver_options.tf = N*h

    ocp_mhe.solver_options.nlp_solver_type = 'SQP'
    # ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp_mhe.solver_options.nlp_solver_max_iter = 200

    acados_solver_mhe = AcadosOcpSolver(ocp_mhe, json_file = 'acados_ocp.json')

    # set arrival cost weighting matrix
    acados_solver_mhe.cost_set(0, "W", block_diag(R, Q, Q0))

    return acados_solver_mhe

if __name__ == '__main__':
    Tf = 5
    N = 50
    h = Tf/N

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
    # mhe model and solver
    model_mhe = export_mhe_model(params)

    nx = model_mhe.x.size()[0]
    nw = model_mhe.u.size()[0]
    ny = nx
    # Q0_mhe= 1 * np.diag([1, 1, 1, 1, 1, 1, 5e-3, 1e-2]) 
    # Q_mhe = 1e6*np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    # R_mhe = 1e2*np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    Q0_mhe = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
    Q_mhe  = 10.*np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
    R_mhe  = 1e3*np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

    acados_solver_mhe = export_mhe_solver(model_mhe, N, h, Q_mhe, Q0_mhe, R_mhe) 