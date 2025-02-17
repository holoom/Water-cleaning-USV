/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_shark_mhe.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

#define NX     SHARK_MHE_NX
#define NZ     SHARK_MHE_NZ
#define NU     SHARK_MHE_NU
#define NP     SHARK_MHE_NP
#define NBX    SHARK_MHE_NBX
#define NBX0   SHARK_MHE_NBX0
#define NBU    SHARK_MHE_NBU
#define NSBX   SHARK_MHE_NSBX
#define NSBU   SHARK_MHE_NSBU
#define NSH    SHARK_MHE_NSH
#define NSG    SHARK_MHE_NSG
#define NSPHI  SHARK_MHE_NSPHI
#define NSHN   SHARK_MHE_NSHN
#define NSGN   SHARK_MHE_NSGN
#define NSPHIN SHARK_MHE_NSPHIN
#define NSBXN  SHARK_MHE_NSBXN
#define NS     SHARK_MHE_NS
#define NSN    SHARK_MHE_NSN
#define NG     SHARK_MHE_NG
#define NBXN   SHARK_MHE_NBXN
#define NGN    SHARK_MHE_NGN
#define NY0    SHARK_MHE_NY0
#define NY     SHARK_MHE_NY
#define NYN    SHARK_MHE_NYN
#define NH     SHARK_MHE_NH
#define NPHI   SHARK_MHE_NPHI
#define NHN    SHARK_MHE_NHN
#define NPHIN  SHARK_MHE_NPHIN
#define NR     SHARK_MHE_NR


int main()
{

    shark_mhe_solver_capsule *acados_ocp_capsule = shark_mhe_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = SHARK_MHE_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = shark_mhe_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("shark_mhe_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = shark_mhe_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = shark_mhe_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = shark_mhe_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = shark_mhe_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = shark_mhe_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = shark_mhe_acados_get_nlp_opts(acados_ocp_capsule);

    // initial condition
    int idxbx0[NBX0];

    double lbx0[NBX0];
    double ubx0[NBX0];

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;
    x_init[3] = 0.0;
    x_init[4] = 0.0;
    x_init[5] = 0.0;
    x_init[6] = 0.0;
    x_init[7] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    u0[4] = 0.0;
    u0[5] = 0.0;
    // set parameters
    double p[NP];
    p[0] = 0;
    p[1] = 0;

    for (int ii = 0; ii <= N; ii++)
    {
        shark_mhe_acados_update_params(acados_ocp_capsule, ii, p, NP);
    }
  

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];


    // solve ocp in loop
    int rti_phase = 0;

    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
        status = shark_mhe_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat( NX, N+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat( NU, N, utraj, NU );
    // ocp_nlp_out_print(nlp_solver->dims, nlp_out);

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("shark_mhe_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("shark_mhe_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);

    shark_mhe_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);

    // free solver
    status = shark_mhe_acados_free(acados_ocp_capsule);
    if (status) {
        printf("shark_mhe_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = shark_mhe_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("shark_mhe_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
