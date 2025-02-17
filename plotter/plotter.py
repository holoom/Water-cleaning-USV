import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os

# Apply LaTeX styling for plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

import numpy as np
import matplotlib.pyplot as plt

def plot_states(time_vals, X_true, X_est=None, save_path=None):
    """
    Plots the true and estimated states (x, y, psi, u, v, r) over time.
    Arranges position states in the left column and velocity states in the right column.

    Parameters:
    - time_vals: Time values (array).
    - X_true: True state values (Nx6 array).
    - X_est: Estimated state values (Nx6 array, optional).
    - save_path: Path to save the figure (optional).
    """

    state_labels = [r"$x$ (m)", r"$y$ (m)", r"$\psi$ (rad)", 
                    r"$u$ (m/s)", r"$v$ (m/s)", r"$r$ (rad/s)"]

    fig, axs = plt.subplots(3, 2, figsize=(8, 4), dpi=300, constrained_layout=True)  

    for i, ax in enumerate(axs.flat):
        if i < 3:
            ax.plot(time_vals, X_true[:, i], label="True", color="blue", linewidth=1.2)
            if X_est is not None:
                ax.plot(time_vals[:len(X_est)], X_est[:, i], '--', label="Estimated", color="red", linewidth=1)
        else:
            ax.plot(time_vals, X_true[:, i], color="blue", linewidth=1.2)
            if X_est is not None:
                ax.plot(time_vals[:len(X_est)], X_est[:, i], '--', color="red", linewidth=1)

        ax.set_ylabel(state_labels[i], fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.8)
        ax.tick_params(axis='both', labelsize=8)

    # Add a single legend outside the plot
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=True)

    if save_path:
        plt.savefig(f"{save_path}/state_variables.png", dpi=300, bbox_inches='tight')
    
    plt.show()



def plot_controls(time_vals, U, u_max, u_min, save_path=None):
    """
    Plots the control inputs (thrust left and thrust right) in a single plot.
    
    Parameters:
    - time_vals: Time values (array).
    - U: Control inputs (Nx2 array).
    - u_max, u_min: Max and min thrust values.
    - save_path: Path to save the figure (optional).
    """

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300, constrained_layout=True)

    # Plot thrust inputs
    ax.plot(time_vals[:-1], U[:, 0], label=r"Left Thrust ($m_l$)", linestyle='solid', color='r', linewidth=0.8)
    ax.plot(time_vals[:-1], U[:, 1], label=r"Right Thrust ($m_r$)", linestyle='solid', color='b', linewidth=0.8)

    # Plot upper/lower thrust limits
    ax.hlines([u_max, u_min], time_vals[0], time_vals[-1], linestyles='dotted', color='black', alpha=0.7, linewidth=1)

    # Labels & Grid
    ax.set_ylabel("Thrust (N)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.grid(True, linestyle='dotted', linewidth=0.7, alpha=0.8)
    ax.tick_params(axis='both', labelsize=8)

    # Legend outside
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    # Save figure
    if save_path:
        plt.savefig(f"{save_path}/control_inputs.png", dpi=300, bbox_inches='tight')

    plt.show()


def plot_parameters(time_vals, est_added_mass, est_linear_drag, true_added_mass, true_linear_drag, save_path=None):
    """
    Plots the estimated parameters (Added Mass and Linear Drag) against time with their true values.
    """
    plt.figure(figsize=(6, 4))
    
    # Plot Added Mass
    plt.subplot(2, 1, 1)
    plt.plot(time_vals, est_added_mass, label="Estimated", color="purple")
    plt.axhline(true_added_mass, linestyle="dashed", color="black", label="True")
    plt.axvline(x=10, color='red', linestyle='dashed', label="Horizon")
    plt.ylabel("Added Mass (kg)")
    plt.xlabel("Time (s)")
    plt.grid(True, linestyle='dotted')
    plt.legend(loc="upper right")
    
    # Plot Linear Drag
    plt.subplot(2, 1, 2)
    plt.plot(time_vals, est_linear_drag, label="Estimated", color="orange")
    plt.axhline(true_linear_drag, linestyle="dashed", color="black", label="True Value")
    plt.axvline(x=10, color='red', linestyle='dashed', label="Horizon")
    plt.ylabel("Linear Drag (kg/s)")
    plt.xlabel("Time (s)")
    plt.grid(True, linestyle='dotted')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/estimated_parameters.png", dpi=300)
    plt.show()

def plot_trajectory(X_true, reference_x, reference_y, save_path=None):
    """
    Plots the actual vehicle trajectory against the reference path in the XY plane.
    """
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot reference path
    plt.plot(reference_x, reference_y, "--r", label="Reference", linewidth=1.1)

    # Plot actual trajectory
    plt.plot(X_true[:, 0], X_true[:, 1], "-b", label="Trajectory", linewidth=1)

    # Mark start and end points
    plt.scatter(X_true[0, 0], X_true[0, 1], color='green', marker='o', label='Start', s=50)
    plt.scatter(X_true[-1, 0], X_true[-1, 1], color='red', marker='x', label='End', s=50)

    # Labels, grid, and legend
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.grid(True, linestyle='dotted', linewidth=0.7, alpha=0.8)
    plt.legend(loc="upper right")
    # plt.title("Vehicle Trajectory vs Reference Path")

    # Save the plot
    if save_path:
        plt.savefig(f"{save_path}/trajectory.png", dpi=300)
    
    plt.show()

def plot_disturbances(time_vals, W_data, save_path=None):
    """
    Plots the linear damping coefficients over time.
    """
    state_labels = [r"$X_{u}$", r"$Y_{v}$ ", r"$Z_{w}$", 
                    r"$K_{p}$", r"$M_{q}$", r"$N_{r}$"]

    fig, axs = plt.subplots(3, 2, figsize=(8, 4), dpi=300, constrained_layout=True)  

    for i, ax in enumerate(axs.flat):
        ax.plot(time_vals, W_data[:, i], color="blue", linewidth=1.2)
        ax.set_ylabel(state_labels[i], fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.8)
        ax.tick_params(axis='both', labelsize=8)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=True)

    if save_path:
        plt.savefig(f"{save_path}/disturbances.png", dpi=300, bbox_inches='tight')
    
    plt.show()


# Load simulation data
# Define case path
case_path = "/root/catkin_ws/src/otter_usv/complete_simulation/data/aquadrone/"

# Load data from the specified path
# data = np.load(os.path.join(case_path, "mhe_results.npz"))
data = np.load(os.path.join(case_path, "mhe_results_wo_trash.npz"))

# Define save directory
save_directory = os.path.join(case_path, "plots_mhe")
os.makedirs(save_directory, exist_ok=True)

# Extract relevant data
time_vals = data["t_vals"]
simU_all = data["simU_all"]
simX_all = data.get("simY_all", None)
estX_all = data.get("estX_all", None)
estW_all = data.get("estW_all", None)
# reference_x = data["reference_x"]
# reference_y = data["reference_y"]
error_data = data.get("error_data", None)



# Define u_max and u_min
u_max = 100
u_min = -100


# Plot and save figures
plot_states(time_vals, simX_all, estX_all, save_path=save_directory)
plot_controls(time_vals, simU_all, u_max, u_min, save_path=save_directory)
plot_disturbances(time_vals, estW_all, save_path=save_directory)
# plot_parameters(time_vals, data["est_added_mass_all"], data["est_linear_drag_all"], true_added_mass, true_linear_drag, save_path=save_directory)
# plot_trajectory(simX_all, reference_x, reference_y, save_path=save_directory)