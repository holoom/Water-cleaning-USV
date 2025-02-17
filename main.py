#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from vehicle.otter import Otter
from vehicle.shark import Shark
from vehicle.aquadrone import Aquadrone
from ref_path_generation import PathPlanner
from MheBasedMPCController import MheBasedMPCController

# Vehicle parameters
LENGTH = 2  # [m]
WIDTH = 1.0  # [m]
BACKTOWHEEL = 0.5  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.15  # [m]
TREAD = 0.3  # [m]
WB = 2.5  # [m]

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
    outline = np.matrix([[np.nan, np.nan, (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                         [np.nan, np.nan, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    rr_wheel = np.matrix([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                          [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    # Create the semicircle for the front side to indicate the direction of movement
    num_points = 20  # Number of points to create the semicircle
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # Keep only the lower semicircle
    semicircle_x = -1 * (WIDTH / 2) * np.cos(theta) + (LENGTH - BACKTOWHEEL)
    semicircle_y = (WIDTH / 2) * np.sin(theta)
    semicircle = np.vstack((semicircle_x, semicircle_y))

    # Connect the outline to the semicircle
    outline = np.hstack((outline, semicircle[:, ::-1]))

    Rot1 = np.matrix([[math.cos(yaw), math.sin(yaw)],
                      [-math.sin(yaw), math.cos(yaw)]])

    rr_wheel = (rr_wheel.T * Rot1).T
    rl_wheel = (rl_wheel.T * Rot1).T
    outline = (outline.T * Rot1).T

    outline[0, :] += x
    outline[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def animate(state, x, y, ox, oy, xref, cx, cy, target_ind, di, time):
    plt.cla()
    if ox is not None:
        plt.plot(ox, oy, "xr", label="MPC")
    plt.plot(cx, cy, "-r", label="course")
    plt.plot(x, y, "ob", label="trajectory")
    plt.plot(xref[:,0], xref[:,1], "xk", label="xref")
    plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
    plot_car(state[0], state[1], state[2], steer=di)
    plt.axis("equal")
    # Calculate axis limits to keep the reference trajectory centered
    padding = 10  # Padding around the trajectory for better visualization
    x_min, x_max = np.min(cx) - padding, np.max(cx) + padding
    y_min, y_max = np.min(cy) - padding, np.max(cy) + padding
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()  
    plt.grid(True)
    plt.title(f"Time[s]: {time:.2f}, speed[km/h]: {state[3] * 3.6:.2f}")
    plt.pause(0.0001)

        
def plot_results(time_vals, states, control_inputs):
    """
    Plots simulation results.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # 1. Plot vehicle trajectory (XY path)
    axs[0].plot(states[:, 0], states[:, 1], label='Trajectory', color='b')
    axs[0].set_xlabel("$x$ (m)")
    axs[0].set_ylabel("$y$ (m)")
    axs[0].set_title("Vehicle XY Path")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_vals, np.degrees(states[:, 2]), label='$\psi$ (deg)')
    axs[1].set_ylabel("Heading (deg)")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time_vals[:-1], control_inputs[:, 0], label='Thrust Left (N)', linestyle='dashed')
    axs[2].plot(time_vals[:-1], control_inputs[:, 1], label='Thrust Right (N)', linestyle='dashed')
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Thrust (N)")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()

def run_simulation(vehicle, planner, controller, sim_time=200, sample_time=0.1):
    """
    Runs the simulation with constant control inputs.
    """
    # Define reference path
    cx, cy = planner.cx, planner.cy
    num_steps = int(sim_time / sample_time) + 1

    # store the states and control inputs
    simX_all = np.zeros((num_steps, 6))  # (x, y, psi, u, v, r)
    simU_all = np.zeros((num_steps - 1, 2))  # (thrust_left, thrust_right)
    time_vals = np.linspace(0, sim_time, num_steps)
    error_data = np.zeros((num_steps, 3))  # (x_error, y_error, yaw_error)
    # store change in hydrodynamic coefficients
    # Store results for plotting
    mp_data = np.zeros((num_steps, 1))
    rp_x_data = np.zeros((num_steps, 1))
    rp_y_data = np.zeros((num_steps, 1))
    MA_data = np.zeros((num_steps, 6))
    D_data = np.zeros((num_steps, 6))
    # Store values
    mp_data[0] = 25
    rp_x_data[0] = 0.05
    rp_y_data[0] = 0
    MA_data[0] = vehicle.MA_coef.copy()
    D_data[0] = vehicle.D_coef.copy()

    #mhe estimator 
    estX_all = np.zeros((num_steps, 6))
    estBias_all = np.zeros((num_steps, 3))

    # Initial state
    full_state = np.zeros(12)  # Full state vector
    simX_all[0] = full_state[[0, 1, 5, 6, 7, 11]]

    
    for i in range(num_steps - 1):
        st = full_state[[0, 1, 5, 6, 7, 11]]  # Extract (x, y, psi, u, v, r) from full state
        # Get reference trajectory
        ref_trajectory, nearest_index = planner.get_reference_trajectory(st)
        print("ref_trajectory shape ",ref_trajectory.shape)
        goal_reached = planner.check_goal(st)
        # Apply controller
        controller.update_state(st)
        controller.update_reference_trajectory(ref_trajectory)
        thrust, predicted_states, x_error, y_error, yaw_error = controller.run_mpc()
        error_data[i] = [x_error, y_error, yaw_error]
        simU_all[i] = thrust
        est_state, est_bias = controller.run_mhe()
        estX_all[i+1] = est_state
        estBias_all[i+1] = est_bias
        ### simulate trash collection
        
        if i*sample_time % 10 == 0:  # Change `rp` every 10 seconds
            new_rp = random_rp()
        new_mp = linear_mp(i*sample_time)

        if not goal_reached:
            vehicle.simulate_trash_collection(new_mp, new_rp)
        # Store values
        mp_data[i+1] = new_mp
        rp_x_data[i+1] = new_rp[0]
        rp_y_data[i+1] = new_rp[1]
        MA_data[i+1] = vehicle.MA_coef.copy()
        D_data[i+1] = vehicle.D_coef.copy()
        # Update state using the vehicle model (RK4 integration inside `step`)
        full_state = vehicle.step(full_state, thrust, sample_time)
        # Store current state
        simX_all[i+1] = full_state[[0, 1, 5, 6, 7, 11]]  # Extract (x, y, psi, u, v, r)
        
        # Visualize the simulation
        real_x = simX_all[:i+1, 0]
        real_y = simX_all[:i+1, 1]
        prediction_horizon_x = predicted_states[:, 0]
        prediction_horizon_y = predicted_states[:, 1]

        animate(st, real_x, real_y, prediction_horizon_x, prediction_horizon_y, ref_trajectory, cx, cy, nearest_index, 0.0, i * sample_time)

    return time_vals, simX_all, simU_all, error_data, mp_data, rp_x_data, rp_y_data, MA_data, D_data, estX_all, estBias_all

# Define a linear function for `mp`
def linear_mp(t):
    return 25.0 + 0.1 * t  # Example: Increasing payload mass over time

# Generate random positions `rp` every 10 seconds within a circle of radius 0.2
def random_rp():
    angle = random.uniform(0, 2 * np.pi)
    radius = random.uniform(0, 0.2)
    return np.array([radius * np.cos(angle), radius * np.sin(angle), -0.35])




if __name__ == "__main__":
    # Set fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    output_dir = "/root/catkin_ws/src/otter_usv/complete_simulation/data/aquadrone/bias_body"
    os.makedirs(output_dir, exist_ok=True)
    # Define simulation parameters
    sim_time = 320 # seconds
    sample_time = 0.1  # seconds
    N = 50  # MPC horizon

    # Initialize vehicle model
    enable_current=False
    V_current=0.2
    beta_current=30
    aquadrone = Aquadrone(V_current=V_current if enable_current else 0, beta_current=beta_current if enable_current else 0)

    ### path planner setup
    # Define waypoints
    # ax = [0.0, 48.0, 64.0, 40.0, 48.0, 24.0, -8.0]
    # ay = [0.0, 0.0, 40.0, 56.0, 32.0, 40.0, -16.0]
    ax = [0.0, 60.0, 60.0, 0.0, 0.0] 
    ay = [0.0, 0.0, 60.0, 60.0, 0.0]

    # Initialize path planner
    planner = PathPlanner(ax, ay, horizon=N, dl=0.1, use_spline=False, min_dist=2, n_ind_search=50)
    cx, cy = planner.cx, planner.cy
    ### controller setup
    mpc_controller = MheBasedMPCController(horizon_length=N, sample_time=sample_time)


    # Run simulation
    time_vals, states, control_inputs, error_data, mp_data, rp_x_data, rp_y_data, MA_data, D_data , estX_all, estBias_all= run_simulation(
        aquadrone, planner=planner, controller=mpc_controller, sample_time=sample_time, sim_time=sim_time
    )


    # Save results
    case_name = "w_trash"
    filename = os.path.join(output_dir, f"{case_name}.npz")
    np.savez(filename, 
            t_vals=time_vals, 
            simY_all=states, 
            simU_all=control_inputs,
            error_data=error_data,
            reference_x=np.array(planner.cx), 
            reference_y=np.array(planner.cy),
            mp_data=mp_data,
            rp_x_data=rp_x_data,
            rp_y_data=rp_y_data,
            MA_data=MA_data,
            D_data=D_data,
            estX_all = estX_all,
            estBias_all = estBias_all)

    print(f"✅ Results saved at {filename}")
