# Adaptive Control for Underactuated Water-Cleaning USVs

This repository contains the code implementation for **Adaptive Model Predictive Control (MPC) with Real-Time Parameter Estimation** applied to **underactuated Unmanned Surface Vehicles (USVs)** for marine debris collection. The work is based on the research paper (soon to be published) 

## **Overview**

This work proposes an **adaptive MPC framework** integrated with **Moving Horizon Estimation (MHE)** to handle dynamic parameter variations due to floating trash. The study includes:
- **Model Predictive Control (MPC)** for robust path-following.
- **Moving Horizon Estimation (MHE)** to track real-time hydrodynamic changes.
- **Python-based simulation environment** for performance validation.
- **Real-world experiments** conducted on USVs for validation.

## **Project Structure**

```plaintext
water-cleaning-USV/
│── acados/                  # Acados submodule for MPC/MHE
│── lib/                     # External libraries 
│── mhe/                     # Moving Horizon Estimation (MHE) implementation
│── mpc/                     # Model Predictive Control (MPC) implementation
│── plotter/                 # Visualization scripts
│── vehicle/                 # Vehicle dynamics and simulation models
│── .gitignore               # Git ignore file for unnecessary files
│── .gitmodules              # Git submodules configuration
│── MheBasedMPCController.py # Main controller script
│── README.md                # Project documentation
│── cubic_spline_planner.py  # Path planning using cubic splines
│── main.py                  # Entry point for simulation
│── ref_path_generation.py    # Reference trajectory generation
```

## **Installation**

### **1. Clone the Repository**
```bash
# git clone --recurse-submodules https://github.com/your_username/water-cleaning-USV.git
# cd water-cleaning-USV
```

### **2. Install and Setup acados**

<!-- The project depends on acados for solving MPC and MHE. Follow these steps to install acados: -->
```bash
# cd acados
# git submodule update --init --recursive
# mkdir build
# cd build
# cmake ..
# make install
```

<!-- After installation, update your environment variables: -->
```bash
# export ACADOS_SOURCE_DIR=$(pwd)
# export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
```

<!-- Running the Simulation -->
```bash
# python3 main.py
```
```
