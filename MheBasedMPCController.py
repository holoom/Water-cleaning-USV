import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from tf.transformations import quaternion_from_euler
from collections import deque
mhe_solver_path = os.path.abspath('/root/catkin_ws/src/otter_usv/complete_simulation/mhe')
mpc_solver_path = os.path.abspath('/root/catkin_ws/src/otter_usv/complete_simulation/mpc')
sys.path.append(mhe_solver_path)
sys.path.append(mpc_solver_path)
# mhe
from export_mhe_solver_with_bias_body import export_mhe_solver_with_bias_body
from export_mhe_ode_model_with_bias_body import export_mhe_model_with_bias_body
#mpc
from build_mpc_model import build_mpc_model

class MheBasedMPCController:
    def __init__(self, horizon_length=100, num_states=8, num_inputs=2, sample_time=0.1):
        """
        Initializes the MHE-based MPC controller.
        """
        self.config = {
            'N': horizon_length,
            'Nx': num_states,
            'Nu': num_inputs,
            'dt': sample_time
        }

        # Define model parameters
        self.param = {
            'added_mass_x': 5.5,
            'added_mass_y': 82.5,
            'added_mass_psi': 25.67,
            'linear_d_x': -77.55,
            'linear_d_y': -162.5,
            'linear_d_psi': -42.65,
            'nonlinear_d_x': 0,
            'nonlinear_d_y': 0,
            'nonlinear_d_psi': 0,
            'b_x': 0,
            'b_y': 0,
            'b_psi': 0
        }
        # self.param_arr = np.array(list(self.param.values()))

        # Initialize state variables
        self.x_current = np.zeros(8)
        self.xref = None
        self.yaw_sum = 0.0
        self.pre_yaw = 0.0
        self.last_solution = None
        self.ref_data = []
        self.actual_data = []
        self.error_data = []

        # Initialize MPC & MHE solvers
        self.mpc_solver = build_mpc_model(h=sample_time, N=horizon_length)
        model_mhe = export_mhe_model_with_bias_body(self.param)
        self.nx = 6
        self.nu = 2
        self.nx_augmented = model_mhe.x.size()[0]
        self.nw = model_mhe.u.size()[0]
        self.ny = self.nx
        

        # Covariance Matrices
        Q0_mhe = 10 * np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 5e-2, 1e-3, 1e-2])
        Q_mhe = 10.0 * np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        R_mhe = 1e6 * np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

        self.mhe_solver = export_mhe_solver_with_bias_body(
            model_mhe, self.config['N'], self.config['dt'], Q_mhe, Q0_mhe, R_mhe
        )

        # Data Buffers
        self.poseBuffer = deque(maxlen=self.config['N'] + 1)
        self.thrustBuffer = deque(maxlen=self.config['N'])

        # Control inputs
        self.control_inputs = np.zeros(2)
        self.mhe_iter = 0

        self.x0_bar = np.zeros((self.nx_augmented,))
        # self.x0_bar[-2:] = np.array([
        #                                 0.75 * self.param['added_mass_x'],  # 75% of added mass in X
        #                                 0.75 * self.param['linear_d_x']     # 75% of linear drag in X
        #                             ])

        self.bias_arr = np.zeros(3)
        self.est_x_arr = np.zeros(6) 


    def update_yaw_sum(self, psi: float) -> float:
        """        
        :param psi: Current yaw angle (radians), expected in [-pi, +pi]
        :return: The updated cumulative (unwrapped) yaw, yaw_sum
        """
        # For convenience
        M_PI = np.pi
        
        if self.pre_yaw >= 0 and psi >= 0:
            # Case 1
            yaw_diff = psi - self.pre_yaw
            
        elif self.pre_yaw >= 0 and psi < 0:
            # Case 2
            # Compare going "forward" (2*pi + psi - pre_yaw) vs. "backward" (-(pre_yaw + abs(psi)))
            if (2 * M_PI + psi - self.pre_yaw) >= (self.pre_yaw + abs(psi)):
                yaw_diff = -(self.pre_yaw + abs(psi))
            else:
                yaw_diff = 2 * M_PI + psi - self.pre_yaw
                
        elif self.pre_yaw < 0 and psi >= 0:
            # Case 3
            # Compare forward vs. backward
            if (2 * M_PI - psi + self.pre_yaw) >= (abs(self.pre_yaw) + psi):
                yaw_diff = abs(self.pre_yaw) + psi
            else:
                yaw_diff = -(2 * M_PI - psi + self.pre_yaw)
                
        else:
            # Case 4
            yaw_diff = psi - self.pre_yaw
        
        # Accumulate and update
        self.yaw_sum += yaw_diff
        self.pre_yaw = psi
        
        return self.yaw_sum

    def update_state(self, state):
        psi = self.update_yaw_sum(state[2])
        self.x_current[:6] = np.array([state[0], state[1], psi, state[3], state[4], state[5]])
        self.poseBuffer.append(state)

    def update_reference_trajectory(self, ref_poses):
        """
        Updates reference path.
        """
        self.xref = ref_poses
        # self.plot_heading(self.xref[0][2], self.x_current[2])
    def run_mpc(self):
        """
        Runs MPC to compute optimal control inputs.
        - If solver fails, uses the second element of the last solution for control inputs.
        - Predicted states are the last solution shifted by one element, with the last element repeated.
        """
        if self.xref is None:
            print("Reference path not set!")
            return
        
        # errors
        x_error = self.x_current[0] - self.xref[0][0]
        y_error = self.x_current[1] - self.xref[0][1]
        yaw_error = self.x_current[2] - self.xref[0][2]
        self.error_data.append((x_error, y_error, yaw_error))

        # Set initial state constraints
        self.mpc_solver.set(0, "lbx", self.x_current)
        self.mpc_solver.set(0, "ubx", self.x_current)

        # Warm-start the solver with the last solution if available
        if hasattr(self, 'last_solution') and self.last_solution is not None:
            for j in range(self.config['N'] + 1):
                if j < len(self.last_solution) - 1:
                    x_guess = self.last_solution[j + 1]  # Shift by one element
                else:
                    x_guess = self.last_solution[-1]  # Repeat the last element
                self.mpc_solver.set(j, "x", x_guess)
        # update parameters
        self.param['b_x'], self.param['b_y'], self.param['b_psi'] = self.bias_arr
        # Set reference trajectories and parameters
        yref_e = self.xref[self.config['N']][:self.config['Nx']]
        for j in range(self.config['N']):
            self.mpc_solver.set(j, "yref", self.xref[j])
            self.mpc_solver.set(j, "p", np.array(list(self.param.values())))
        self.mpc_solver.set(self.config['N'], "yref", yref_e)

        # Solve MPC
        start_time = time.time()
        status = self.mpc_solver.solve()
        solver_time = time.time() - start_time

        if status != 0:
            print("MPC solver failed!")
            if hasattr(self, 'last_solution') and self.last_solution is not None:
                # Use the second element of the last solution for control inputs
                u0 = self.last_solution[1][6:8]

                # Predicted states are the last solution shifted by one element, with the last element repeated
                predicted_states = np.vstack([
                    self.last_solution[1:],  # Shift by one element
                    [self.last_solution[-1]]  # Repeat the last element
                ])

                self.x_current = self.last_solution[2]

                # Print fallback results
                print("\n--- MPC Results (Fallback) ---")
                print(f"⚠️ Using previous solution due to solver failure!")
                print(f"🔹 Control Inputs: Thrust 1 = {u0[0]:.3f}, Thrust 2 = {u0[1]:.3f}")
                print("-------------------\n")
            else:
                print("No previous solution available for fallback")
                return
        else:
            # If solve successful
            x0 = self.mpc_solver.get(0, "x")
            u0 = x0[6:8]

            # Store solution for next iteration
            predicted_states = np.array([self.mpc_solver.get(i, "x") for i in range(self.config['N'] + 1)])
            self.last_solution = predicted_states

            # Update current state
            self.x_current = self.mpc_solver.get(1, "x")

            # Print results
            print("\n--- MPC Results ---")
            print(f"✅ Solver Time: {solver_time:.6f} seconds")
            print(f"🔹 Control Inputs: Thrust 1 = {u0[0]:.3f}, Thrust 2 = {u0[1]:.3f}")
            print(f"📉 Errors: X = {x_error:.3f}, Y = {y_error:.3f}, Yaw = {yaw_error:.3f}")
            print("param_arr",np.array(list(self.param.values())))

            print("-------------------\n")
        self.thrustBuffer.append((u0[0], u0[1]))


        ### MHE
        # self.run_mhe()
        return u0, predicted_states, x_error, y_error, yaw_error
    

    '''
    def run_mhe(self):

        N = self.config['N']

        if len(self.poseBuffer) < N+1 or len(self.thrustBuffer) < N:
            print("Not enough data for MHE.")
            print("----------------------------------------------")
            return
        
        print("----------------------------------------------")
        print("mhe iteration:",self.mhe_iter)
        simY = np.array(self.poseBuffer)     # shape should be (N+1, n_meas)
        simU = np.array(self.thrustBuffer)   # shape should be (N, nu)
        print("simY shape:", simY.shape)
        print("simU shape:", simU.shape)
        # self.plot_xy(simY)
        
        # define arrays to store the solution for each of the N+1 nodes
        simXest = np.zeros((N+1, self.nx))
        simWest = np.zeros((N, self.nx))

        for j in range(N):
            yref = np.zeros((3*self.nx, ))
            # set measurements and controls
            yref[:self.nx] = simY[j, :]
            yref[2*self.nx:] = self.x0_bar
            self.mhe_solver.set(j, "yref", yref)
            self.mhe_solver.set(j, "p", simU[j,:])

        status = self.mhe_solver.solve()
        if status != 0:
            raise Exception(f'ACADOS returned status {status} in MHE solve.')

        
        # get solution
        for i in range(N):
            x_augmented = self.mhe_solver.get(i, "x")
            simXest[i,:] = x_augmented
            simWest[i,:] = self.mhe_solver.get(i, "u")

            
        x_augmented = self.mhe_solver.get(N, "x")
        simXest[N,:] = x_augmented

        print('difference |x0_est - x0_bar|', np.linalg.norm(self.x0_bar - simXest[0, :]))
        print('difference |x_est - x_true|', np.linalg.norm(simXest - simY))
        
        self.x0_bar = self.mhe_solver.get(1, "x")
        self.mhe_iter+=1
    '''
        
    def run_mhe(self):

        N = self.config['N']

        if len(self.poseBuffer) < N+1 or len(self.thrustBuffer) < N:
            print("Not enough data for MHE.")
            print("----------------------------------------------")
            self.est_x_arr = self.x_current[:6]
            return self.est_x_arr, self.bias_arr
        
        print("----------------------------------------------")
        print("mhe iteration:",self.mhe_iter)
        simY = np.array(self.poseBuffer)     # shape should be (N+1, n_meas)
        simU = np.array(self.thrustBuffer)   # shape should be (N, nu)
        print("simY shape:", simY.shape)
        print("simU shape:", simU.shape)
        # self.plot_xy(simY)
        
        # define arrays to store the solution for each of the N+1 nodes
        simXest = np.zeros((N+1, self.nx))
        simWest = np.zeros((N, self.nx))
        simBias_est = np.zeros((N+1, 3))

        # set measurements and controls
        yref_0 = np.zeros((2*self.nx + self.nx_augmented, ))
        yref_0[:self.nx] = simY[0, :]
        yref_0[2*self.nx:] = self.x0_bar
        self.mhe_solver.set(0, "yref", yref_0)
        self.mhe_solver.set(0, "p", simU[0,:])

        # set initial guess to x0_bar
        self.mhe_solver.set(0, "x", self.x0_bar)

        yref = np.zeros((2*self.nx, ))

        for j in range(1, N):
            yref[:self.nx] = simY[j, :]
            self.mhe_solver.set(j, "yref", yref)
            self.mhe_solver.set(j, "p", simU[j, :])

            # set initial guess to x0_bar
            self.mhe_solver.set(j, "x", self.x0_bar)
        
        self.mhe_solver.set(N, "x", self.x0_bar)

        status = self.mhe_solver.solve()
        if status != 0:
            raise Exception(f'ACADOS returned status {status} in MHE solve.')

        
        # get solution
        for i in range(N):
            x_augmented = self.mhe_solver.get(i, "x")
            simXest[i,:] = x_augmented[0:self.nx]
            simBias_est[i,:] = x_augmented[self.nx:]
            simWest[i,:] = self.mhe_solver.get(i, "u")
            
        x_augmented = self.mhe_solver.get(N, "x")
        simXest[N,:] = x_augmented[0:self.nx]
        simBias_est[N,:] = x_augmented[self.nx:]

        print('difference |x0_est - x0_bar|', np.linalg.norm(self.x0_bar[0:self.nx] - simXest[0, :]))
        print('difference |x_est - x_true|', np.linalg.norm(simXest - simY))
        print('bias_est', simBias_est[0] )

        self.x0_bar = self.mhe_solver.get(1, "x")
        self.est_x_arr = simXest[N,:]
        self.bias_arr = simBias_est[0,:]
        self.mhe_iter+=1
        return self.est_x_arr, self.bias_arr
   
    def plot_heading(self, ref, actual):
        self.ref_data.append(ref)
        self.actual_data.append(actual)

        plt.cla()  

        ref_degrees = self.ref_data
        actual_degrees = self.actual_data

        plt.plot(ref_degrees, "-r", label="reference heading")
        plt.plot(actual_degrees, "ob", label="actual heading")
        

        plt.legend()  
        plt.grid(True)
        plt.title("Heading Comparison")
        plt.xlabel('Time Steps')
        plt.ylabel('Degrees')

        plt.pause(0.0001) 

if __name__ == "__main__":
    N = 100
    h = 0.1
    ref_poses = np.zeros((N + 1, 10))  # N+1 by 8 matrix
    # Move in x-direction only (set increasing values for x)
    ref_poses[:, 0] = np.linspace(0, 10, N + 1)  # Move 10m in x-direction
    current_state = np.array([0, 0, 0, 0, 0, 0])  # Initial state
    controller = MheBasedMPCController(horizon_length=N, sample_time=h)
    controller.update_state(current_state)
    controller.update_reference_trajectory(ref_poses)
    controller.run_mpc()
