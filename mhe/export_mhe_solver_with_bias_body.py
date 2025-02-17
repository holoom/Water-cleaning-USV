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
from export_mhe_ode_model_with_bias_body import export_mhe_model_with_bias_body


def export_mhe_solver_with_bias_body(model, N, h, Q, Q0, R, use_cython=False):

    # create render arguments
    ocp_mhe = AcadosOcp()

    ocp_mhe.model = model

    nx_augmented = model.x.size()[0]
    nu = model.u.size()[0]
    nparam = model.p.size()[0]
    nx = 6

    ny = R.shape[0] + Q.shape[0]                    # h(x), w
    ny_e = 0
    ny_0 = R.shape[0] + Q.shape[0] + Q0.shape[0]    # h(x), w and arrival cost

    # set number of shooting nodes
    ocp_mhe.dims.N = N

    x = ocp_mhe.model.x
    u = ocp_mhe.model.u

    # set cost type
    ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
    ocp_mhe.cost.cost_type_e = 'LINEAR_LS'
    ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'

    ocp_mhe.cost.W_0 = block_diag(R, Q, Q0)
    ocp_mhe.model.cost_y_expr_0 = vertcat(x[:nx], u, x)
    ocp_mhe.cost.yref_0 = np.zeros((ny_0,))


    # cost intermediate stages
    ocp_mhe.cost.W = block_diag(R, Q)

    ocp_mhe.model.cost_y_expr = vertcat(x[0:nx], u)

    ocp_mhe.parameter_values = np.zeros((nparam, ))

    # set y_ref for all stages
    ocp_mhe.cost.yref  = np.zeros((ny,))
    ocp_mhe.cost.yref_e = np.zeros((ny_e, ))
    ocp_mhe.cost.yref_0  = np.zeros((ny_0,))

    # set constraints
    # std_devs = np.array([
    # 0.02, 0.02, 0.005,  # (x, y, psi)
    # 0.1, 0.1, 0.01  # (u, v, r)
    # ])

    # # Using 3 standard deviations to set bounds
    # u_min = -3 * std_devs
    # u_max = 3 * std_devs

    # Apply these bounds to the OCP constraints
    # ocp_mhe.constraints.lbu = u_min
    # ocp_mhe.constraints.ubu = u_max
    # ocp_mhe.constraints.idxbu = np.arange(nu)        # indices of bounds on u

    # ocp_mhe.constraints.lbx = np.array([200, 90])
    # ocp_mhe.constraints.ubx = np.array([400, 120])
    # ocp_mhe.constraints.idxbx = np.array([6,7])

    # set QP solver
    # ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp_mhe.solver_options.integrator_type = 'ERK'

    # set prediction horizon
    ocp_mhe.solver_options.tf = N*h

    ocp_mhe.solver_options.nlp_solver_type = 'SQP'
    # ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp_mhe.solver_options.nlp_solver_max_iter = 200
    ocp_mhe.code_export_directory = 'mhe_generated_code'

    if use_cython:
        AcadosOcpSolver.generate(ocp_mhe, json_file='acados_ocp_mhe.json')
        AcadosOcpSolver.build(ocp_mhe.code_export_directory, with_cython=True)
        acados_solver_mhe = AcadosOcpSolver.create_cython_solver('acados_ocp_mhe.json')
    else:
        acados_solver_mhe = AcadosOcpSolver(ocp_mhe, json_file = 'acados_ocp_mhe.json')

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
    model_mhe = export_mhe_model_with_bias_body(params)

    nx = model_mhe.x.size()[0]
    nw = model_mhe.u.size()[0]
    ny = nx
    # Q0_mhe= 1 * np.diag([1, 1, 1, 1, 1, 1, 5e-3, 1e-2]) 
    # Q_mhe = 1e6*np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    # R_mhe = 1e2*np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    Q0_mhe = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    Q_mhe  = 10.*np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
    R_mhe  = 1e3*np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

    acados_solver_mhe = export_mhe_solver_with_bias_body(model_mhe, N, h, Q_mhe, Q0_mhe, R_mhe) 