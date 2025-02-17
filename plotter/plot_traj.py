import matplotlib.pyplot as plt
import numpy as np
import os

# Apply LaTeX styling for plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 6,  # Reduced legend font size
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

def plot_trajectory(X_adaptive, X_nominal, reference_x, reference_y, case_name, save_path=None):
    """
    Plots the actual vehicle trajectories (Adaptive and Nominal MPC) against the reference path.
    """
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot reference path
    plt.plot(reference_x, reference_y, "--r", label="Reference", linewidth=1.1)

    # Plot Adaptive MPC trajectory
    plt.plot(X_adaptive[:, 0], X_adaptive[:, 1], "-b", label="Adaptive MPC", linewidth=1)

    # Plot Nominal MPC trajectory
    plt.plot(X_nominal[:, 0], X_nominal[:, 1], "-g", label="Nominal MPC", linewidth=1)

    # Mark start and end points
    plt.scatter(X_adaptive[0, 0], X_adaptive[0, 1], color='black', marker='o', label='Start', s=50)
    plt.scatter(X_adaptive[-1, 0], X_adaptive[-1, 1], color='purple', marker='x', label='End (Adaptive)', s=50)
    plt.scatter(X_nominal[-1, 0], X_nominal[-1, 1], color='orange', marker='x', label='End (Nominal)', s=50)

    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.grid(True, linestyle='dotted', linewidth=0.7, alpha=0.8)

    # Adjust legend size and placement
    legend = plt.legend(loc="upper right", prop={'size': 6}, framealpha=0.8)  # Reduced font size
    legend.get_frame().set_linewidth(0.5)  # Thinner legend border
    legend.get_frame().set_alpha(0.7)  # Make legend background slightly transparent

    if save_path:
        save_filename = f"trajectory_comparison_{case_name}.png"
        plt.savefig(os.path.join(save_path, save_filename), dpi=300)
    
    plt.show()

# Load simulation data for both cases
case1_path = "/root/catkin_ws/src/otter_usv/complete_simulation/data/aquadrone/bias_body/"
case2_path = "/root/catkin_ws/src/otter_usv/complete_simulation/data/aquadrone/"

case_name = "w_current"  # Example case name
data_case1 = np.load(os.path.join(case1_path, f"{case_name}.npz"))
data_case2 = np.load(os.path.join(case2_path, f"{case_name}.npz"))

# Define save directory
save_directory = os.path.join(case1_path, "plotssss")
os.makedirs(save_directory, exist_ok=True)

# Extract relevant data
reference_x = data_case1["reference_x"]
reference_y = data_case1["reference_y"]
simX_adaptive = data_case1.get("simY_all", None)  # Adaptive MPC trajectory
simX_nominal = data_case2.get("simY_all", None)   # Nominal MPC trajectory

# Plot and save the trajectory comparison with case name in filename
plot_trajectory(simX_adaptive, simX_nominal, reference_x, reference_y, case_name, save_path=save_directory)
