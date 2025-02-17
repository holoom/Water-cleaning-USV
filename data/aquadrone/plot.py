import matplotlib.pyplot as plt
import numpy as np
import os

# Apply LaTeX styling for plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 6,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

def plot_states(time_vals, X_true, X_est=None, save_path=None):
    """
    Plots the true and estimated states (x, y, psi, u, v, r) over time.
    """
    state_labels = [r"$x$ (m)", r"$y$ (m)", r"$\psi$ (rad)", 
                    r"$u$ (m/s)", r"$v$ (m/s)", r"$r$ (rad/s)"]

    fig, axs = plt.subplots(3, 2, figsize=(8, 4), dpi=300, constrained_layout=True)  

    for i, ax in enumerate(axs.flat):
        ax.plot(time_vals, X_true[:, i], color="blue", linewidth=1.2)
        if X_est is not None:
            ax.plot(time_vals[:len(X_est)], X_est[:, i], '--', label="Estimated", color="red", linewidth=1)
        ax.set_ylabel(state_labels[i], fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.8)
        ax.tick_params(axis='both', labelsize=8)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=True)

    if save_path:
        plt.savefig(f"{save_path}/state_variables.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_added_mass_coef(time_vals, MA_data, save_path=None):
    """
    Plots the added mass coefficients over time.
    """

    state_labels = [r"$X_{\dot{u}}$ (kg)", r"$Y_{\dot{v}}$ (kg)", r"$Z_{\dot{w}}$ (kg)", 
                    r"$K_{\dot{p}}$ (kg)", r"$M_{\dot{q}}$ (kg)", r"$N_{\dot{r}}$ (kg)"]

    fig, axs = plt.subplots(3, 2, figsize=(8, 4), dpi=300, constrained_layout=True)  

    for i, ax in enumerate(axs.flat):
        ax.plot(time_vals, -1 * MA_data[:, i], color="blue", linewidth=1.2)
        ax.set_ylabel(state_labels[i], fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.8)
        ax.tick_params(axis='both', labelsize=8)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=True)

    if save_path:
        plt.savefig(f"{save_path}/added_mass_coefficients.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_linear_damping_coef(time_vals, D_data, save_path=None):
    """
    Plots the linear damping coefficients over time.
    """
    state_labels = [r"$X_{u}$", r"$Y_{v}$ ", r"$Z_{w}$", 
                    r"$K_{p}$", r"$M_{q}$", r"$N_{r}$"]

    fig, axs = plt.subplots(3, 2, figsize=(8, 4), dpi=300, constrained_layout=True)  

    for i, ax in enumerate(axs.flat):
        ax.plot(time_vals, D_data[:, i], color="blue", linewidth=1.2)
        ax.set_ylabel(state_labels[i], fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.8)
        ax.tick_params(axis='both', labelsize=8)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=True)

    if save_path:
        plt.savefig(f"{save_path}/linear_damping_coefficients.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_controls(time_vals, U, u_max, u_min, save_path=None):
    """
    Plots the control inputs (thrust left and thrust right).
    """
    fig, ax = plt.subplots(figsize=(6, 1.5), dpi=300, constrained_layout=True)

    ax.plot(time_vals[:-1], U[:, 0], label=r"Left Thrust ($m_l$)", linestyle='solid', color='r', linewidth=0.8)
    ax.plot(time_vals[:-1], U[:, 1], label=r"Right Thrust ($m_r$)", linestyle='solid', color='b', linewidth=0.8)

    ax.hlines([u_max, u_min], time_vals[0], time_vals[-1], linestyles='dotted', color='black', alpha=0.7, linewidth=1)

    ax.set_ylabel("Thrust (N)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.grid(True, linestyle='dotted', linewidth=0.7, alpha=0.8)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    if save_path:
        plt.savefig(f"{save_path}/control_inputs.png", dpi=300, bbox_inches='tight')

    plt.show()

def plot_trajectory(X_true, reference_x, reference_y, save_path=None):
    """
    Plots the actual vehicle trajectory against the reference path.
    """
    plt.figure(figsize=(8, 6), dpi=300)

    plt.plot(reference_x, reference_y, "--r", label="Reference", linewidth=1.1)
    plt.plot(X_true[:, 0], X_true[:, 1], "-b", label="Trajectory", linewidth=1)

    plt.scatter(X_true[0, 0], X_true[0, 1], color='green', marker='o', label='Start', s=50)
    plt.scatter(X_true[-1, 0], X_true[-1, 1], color='red', marker='x', label='End', s=50)

    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.grid(True, linestyle='dotted', linewidth=0.7, alpha=0.8)
    plt.legend(loc="upper right")

    if save_path:
        plt.savefig(f"{save_path}/trajectory.png", dpi=300)
    
    plt.show()

def plot_simulated_trash_and_position(time_vals, mp_data, rp_x_data, rp_y_data, save_path=None):
    """
    Plots:
    1. Simulated trash (kg) over time
    2. Trash position (rp_x, rp_y) over time in separate subplots.
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), dpi=300, constrained_layout=True)

    # (1) Simulated Trash Over Time
    axs[0].plot(time_vals, mp_data, label="Simulated Trash (kg)", color='b', linewidth=1.2)
    axs[0].set_ylabel("Simulated Trash (kg)")
    axs[0].set_xlabel("Time (s)", fontsize=9)
    axs[0].grid(True, linestyle="dotted", linewidth=0.7, alpha=0.8)
    axs[0].legend(loc="upper left")

    # (2) Trash Position Over Time
    axs[1].plot(time_vals, rp_x_data, label=r"$r_{p_x}$", color='r', linestyle="dashed", linewidth=1)
    axs[1].plot(time_vals, rp_y_data, label=r"$r_{p_y}$", color='g', linestyle="dashed", linewidth=1)
    axs[1].set_ylabel("Trash Position (m)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True, linestyle="dotted", linewidth=0.7, alpha=0.8)
    axs[1].legend(loc="upper left")

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(f"{save_path}/simulated_trash_and_position.png", dpi=300, bbox_inches='tight')

    plt.show()
def plot_bias(time_vals, estBias, save_path=None):
    """
    Plots the estimated bias components (b_x, b_y, b_psi) over time.

    Parameters:
        time_vals (array): Time values.
        estBias (array): Estimated bias values (201 x 3 array).
        save_path (str, optional): Path to save the plot.
    """
    labels = [r"$b_x$ (N)", r"$b_y$ (N)", r"$b_{\psi}$ (N·m)"]

    fig, axs = plt.subplots(3, 1, figsize=(6, 2.5), dpi=300, constrained_layout=True)  

    for i, ax in enumerate(axs):
        ax.plot(time_vals, estBias[:, i], color="blue", linewidth=1.2)
        ax.set_ylabel(labels[i], fontsize=9)
        ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.8)
        ax.tick_params(axis='both', labelsize=8)

        # Only add the x-axis label to the last subplot
        if i == len(axs) - 1:
            ax.set_xlabel("Time (s)", fontsize=9)

    if save_path:
        plt.savefig(f"{save_path}/estimated_bias.png", dpi=300, bbox_inches='tight')

    plt.show()

# Load simulation data
case_path = "/root/catkin_ws/src/otter_usv/complete_simulation/data/aquadrone"
case_name = "w_trash_and_current"
data = np.load(os.path.join(case_path, f"{case_name}.npz"))

# Define save directory
save_directory = os.path.join(case_path, "plots")
os.makedirs(save_directory, exist_ok=True)

# Extract relevant data
time_vals = data["t_vals"]
simU_all = data["simU_all"]
simX_all = data.get("simY_all", None)
estX_all = data.get("estX_all", None)
reference_x = data["reference_x"]
reference_y = data["reference_y"]
mp_data = data.get("mp_data", None)
rp_x_data = data.get("rp_x_data", None)
rp_y_data = data.get("rp_y_data", None)
MA_data = data.get("MA_data", None)
D_data = data.get("D_data", None)
estBias_all = data.get("estBias_all", None)

print(estBias_all)

# Define u_max and u_min
u_max = 100
u_min = -100

# Plot and save figures
# plot_states(time_vals, simX_all, estX_all, save_path=save_directory)
# plot_added_mass_coef(time_vals, MA_data, save_path=save_directory)
# plot_linear_damping_coef(time_vals, D_data, save_path=save_directory)
plot_controls(time_vals, simU_all, u_max, u_min, save_path=save_directory)
# plot_trajectory(simX_all, reference_x, reference_y, save_path=save_directory)
# plot_simulated_trash_and_position(time_vals, mp_data, rp_x_data, rp_y_data, save_path=save_directory)
# plot_bias(time_vals, estBias_all, save_path=save_directory)