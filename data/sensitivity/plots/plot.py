import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# List of simulation result files (updated for added mass)
new_files = [
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_x_1.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_psi_0.5.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_psi_0.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_psi_1.5.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_x_0.5.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_x_0.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_x_1.5.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_y_0.5.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_y_0.npz",
    "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/added_mass_y_1.5.npz"
]

# Updated legend labels for added mass (with proper LaTeX formatting)
legend_labels = [
    "Exact",
    r"$N_{\dot{r}} = 0.5 N_{\dot{r}0}$", r"$N_{\dot{r}} = 0 N_{\dot{r}0}$", r"$N_{\dot{r}} = 1.5 N_{\dot{r}0}$",
    r"$X_{\dot{u}} = 0.5 X_{\dot{u}0}$", r"$X_{\dot{u}} = 0 X_{\dot{u}0}$", r"$X_{\dot{u}} = 1.5 X_{\dot{u}0}$", 
    r"$Y_{\dot{v}} = 0.5 Y_{\dot{v}0}$", r"$Y_{\dot{v}} = 0 Y_{\dot{v}0}$", r"$Y_{\dot{v}} = 1.5 Y_{\dot{v}0}$"
]


# Function to compute RMSE
def compute_rmse(errors):
    return np.sqrt(np.mean(errors**2))

# Collect RMSE values
rmse_results = []

for idx, file_path in enumerate(new_files):
    data = np.load(file_path)
    error_data = data['error_data']  # Assuming shape (N, 3) for x_e, y_e, psi_e
    
    rmse_x = compute_rmse(error_data[:, 0])
    rmse_y = compute_rmse(error_data[:, 1])
    rmse_psi = compute_rmse(error_data[:, 2])
    
    rmse_results.append([legend_labels[idx], rmse_x, rmse_y, rmse_psi])

# Create a DataFrame
rmse_df = pd.DataFrame(rmse_results, columns=["Condition", "RMSE_x (m)", "RMSE_y (m)", "RMSE_psi (rad)"])



# Create a figure for error plots with a size of 6x4 inches
fig, ax = plt.subplots(3, 1, figsize=(6, 4), sharex=True)

# Adjust spacing to make room for the legend
plt.subplots_adjust(right=0.75)

# Plot all errors in one figure
for idx, file_path in enumerate(new_files):
    data = np.load(file_path)
    time_vals = data['t_vals']
    error_data = data['error_data']
    
    ax[0].plot(time_vals, error_data[:, 0], label=legend_labels[idx], linewidth=0.8)
    ax[1].plot(time_vals, error_data[:, 1], linewidth=0.8)
    ax[2].plot(time_vals, error_data[:, 2], linewidth=0.8)

ax[0].set_ylabel(r"$x_e$ (m)")
ax[1].set_ylabel(r"$y_e$ (m)")
ax[2].set_ylabel(r"$\psi_e$ (rad)")
ax[2].set_xlabel("Time (s)")

ax[0].grid()
ax[1].grid()
ax[2].grid()

plt.suptitle("Errors Over Time", y=1)  # Adjust y-value to move the title higher

# Single legend for all subplots (placed outside the plot)
legend = fig.legend(loc='upper left', bbox_to_anchor=(0.95, 0.75))

plt.tight_layout()

# Save the figure and ensure the legend is included properly
error_plot_path = "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/errors_over_time.png"
plt.savefig(error_plot_path, dpi=300, bbox_inches="tight", bbox_extra_artists=[legend])

plt.show()


# Trajectory Plot
plt.figure(figsize=(8, 6))
for idx, file_path in enumerate(new_files):
    data = np.load(file_path)
    ref_x, ref_y = data['reference_x'], data['reference_y']
    states = data['simY_all']
    
    plt.plot(states[:, 0], states[:, 1], label=legend_labels[idx])

plt.plot(ref_x, ref_y, label="Reference Trajectory", linestyle="dashed", color="black")
plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$y$ (m)")
plt.title("Trajectory Comparison")
plt.legend(loc="upper left")
plt.grid()

# Save the trajectory plot
trajectory_plot_path = "/root/catkin_ws/src/otter_usv/complete_simulation/data/sensitivity/trajectory_comparison.png"
plt.savefig(trajectory_plot_path, dpi=300, bbox_inches="tight")

plt.show()
