import numpy as np
import math
import time
import matplotlib.pyplot as plt
from cubic_spline_planner import calc_spline_course


# Global variables
stuck_counter = 0
goal_reached = False


class PathPlanner:
    """
    Path planning class that generates a reference trajectory and computes the nearest reference index.
    """
    def __init__(self, waypoints_x, waypoints_y, dl=0.1, min_dist=2.0, use_spline=True, target_speed=1, horizon=10, nx=6, nu=2, goal_dis=1.5, stop_speed=0.5/3.6, n_ind_search=20):
        """
        Initialize the path planner with configurable parameters.
        """
        # Configurable parameters
        self.NX = nx  # Number of state variables
        self.NU = nu  # Number of control variables
        self.GOAL_DIS = goal_dis  # Goal distance
        self.STOP_SPEED = stop_speed  # Stop speed
        self.TARGET_SPEED = target_speed  # Target speed
        self.HORIZON = horizon  # Number of reference points to return
        self.N_IND_SEARCH = n_ind_search  # Number of indices to search for nearest point
        self.pind = 0  # Initialize previous index for tracking progress

        self.use_spline = use_spline  # Option to choose path type
        if use_spline:
            self.cx, self.cy, self.cyaw, _, self.traj = self.generate_reference_path(waypoints_x, waypoints_y, dl)
        else:
            self.cx, self.cy, self.cyaw, _, self.traj = self.get_straight_path(waypoints_x, waypoints_y, dl, min_dist)
        cyaw_unwrapped = np.unwrap(self.traj[:, 2], discont=np.pi)  # Unwrap heading
        self.traj[:, 2] = cyaw_unwrapped  # Update heading
        np.savetxt("reference.txt", self.traj, fmt="%.6f", delimiter=",")
        self.goal = [self.cx[-1], self.cy[-1]]
    @staticmethod
    def ssa(angle):
        """Wraps angle between -pi and pi."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    @staticmethod
    def pi_2_pi(angle):
        """
        Normalize angle to be within [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def generate_reference_path(self, ax, ay, dl):
        """
        Generates a reference trajectory using cubic spline interpolation.
        """
        cx, cy, cyaw, ck, _ = calc_spline_course(ax, ay, ds=dl)

        # Convert cyaw to numpy array
        cyaw = np.array(cyaw)  # Ensure cyaw is a NumPy array
        cyaw = self.ssa(cyaw)  # Apply angle wrapping

        traj = np.zeros((len(cx), self.NX + self.NU))
        traj[:, 0] = cx  
        traj[:, 1] = cy  
        traj[:, 2] = cyaw  

        return cx, cy, cyaw, ck, traj
    
    def get_straight_path(self, ax, ay, dl, min_dist):
        """
        Generates a straight-line reference trajectory with interpolated points.
        """
        cx = []
        cy = []
        cyaw = []
        ck = []

        # Process waypoints and add additional points if needed
        processed_ax = [ax[0]]
        processed_ay = [ay[0]]

        for i in range(1, len(ax)):
            x1, y1 = processed_ax[-1], processed_ay[-1]
            x2, y2 = ax[i], ay[i]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if dist > min_dist:
                # Add waypoint at min_dist after the first point
                t1 = min_dist / dist
                new_x1 = x1 + t1 * (x2 - x1)
                new_y1 = y1 + t1 * (y2 - y1)
                processed_ax.append(new_x1)
                processed_ay.append(new_y1)

                # Add waypoint at min_dist before the second point
                t2 = (dist - min_dist) / dist
                new_x2 = x1 + t2 * (x2 - x1)
                new_y2 = y1 + t2 * (y2 - y1)
                processed_ax.append(new_x2)
                processed_ay.append(new_y2)

            processed_ax.append(x2)
            processed_ay.append(y2)
        
        # Generate path based on new waypoints
        for i in range(len(processed_ax) - 1):
            x1, y1 = processed_ax[i], processed_ay[i]
            x2, y2 = processed_ax[i + 1], processed_ay[i + 1]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            local_dl = dl if dist > min_dist + 1 else dl / 4
            num_points = int(dist / local_dl)

            for j in range(num_points + 1):
                t = j / num_points
                x = x1 * (1 - t) + x2 * t
                y = y1 * (1 - t) + y2 * t

                cx.append(x)
                cy.append(y)
                cyaw.append(math.atan2(y2 - y1, x2 - x1))
                ck.append(0.0)

        traj = np.zeros((len(cx), self.NX + self.NU))
        traj[:, 0] = cx
        traj[:, 1] = cy
        traj[:, 2] = cyaw

        return cx, cy, cyaw, ck, traj

    def calc_nearest_index(self, state):
        """
        Finds the nearest index in the reference trajectory based on the current state.
        """
        global stuck_counter, goal_reached

        dx = [state[0] - icx for icx in self.cx[self.pind:(self.pind + self.N_IND_SEARCH)]]
        dy = [state[1] - icy for icy in self.cy[self.pind:(self.pind + self.N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        mind = min(d)

        ind = d.index(mind) + self.pind
        mind = math.sqrt(mind)

        dxl = self.cx[ind] - state[0]
        dyl = self.cy[ind] - state[1]

        angle = self.pi_2_pi(self.cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        # Check if the vehicle is stuck
        if state[3] < 0.2 and not goal_reached:
            stuck_counter += 1
        else:
            stuck_counter = 0

        if stuck_counter > 10:
            ind = min(ind + 1, len(self.cx) - 1)
            stuck_counter = 0

        return ind, mind

    def get_reference_trajectory(self, state):
        """
        Computes the reference trajectory starting from the nearest index.
        Uses unwrapped yaw handling and ensures smooth trajectory following.
        """
        ncourse = self.traj.shape[0]  # Total trajectory points
        cx = self.traj[:, 0]  # X-coordinates
        cy = self.traj[:, 1]  # Y-coordinates
        cyaw_unwrapped = self.traj[:, 2]  # Unwrapped yaw

        # Find nearest reference index
        nearest_index, _ = self.calc_nearest_index(state)

        # Ensure progress in the path
        if self.pind >= nearest_index:
            nearest_index = self.pind

        self.pind = nearest_index  # Update previous index

        # Initialize reference trajectory storage
        xref = np.zeros((self.HORIZON + 1, self.NX + self.NU + self.NU))

        for i in range(self.HORIZON + 1):
            if nearest_index + i < ncourse:
                xref[i, 0] = cx[nearest_index + i]  # x
                xref[i, 1] = cy[nearest_index + i]  # y
                xref[i, 2] = cyaw_unwrapped[nearest_index + i]  # yaw
            else:
                xref[i, 0] = cx[-1]
                xref[i, 1] = cy[-1]
                xref[i, 2] = cyaw_unwrapped[-1]

        return xref, nearest_index



    def check_goal(self, state):
        """
        Checks if the vehicle has reached the goal.
        """
        dx = state[0] - self.goal[0]
        dy = state[1] - self.goal[1]
        d = math.sqrt(dx ** 2 + dy ** 2)
        return d <= self.GOAL_DIS and abs(state[3]) <= self.STOP_SPEED
    def plot_trajectory(self, state, actual_x, actual_y, ref_trajectory):
        """
        Plots the reference, actual, and planned reference trajectories.
        """
        plt.cla()
        
        # Plot full reference path
        plt.plot(self.cx, self.cy, "-r", label="Reference Path")
        
        # Plot the actual trajectory
        plt.plot(actual_x, actual_y, "-b", label="Actual Path")
        
        # Plot the planned reference trajectory
        plt.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], "og", markersize=4, label="Reference Trajectory (MPC Horizon)")
        
        # Mark current vehicle position
        plt.scatter(state[0], state[1], color="black", marker="*", label="Vehicle Position")
        
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.title(f"Time: {round(time.time(), 2)} sec, Speed: {state[3] * 3.6:.2f} km/h")
        plt.pause(0.001)


# Main function
def main():
    """
    Simulates vehicle following a reference trajectory.
    """
    # Define waypoints
    ax = [0.0, 30.0, 60.0, 25.0, 40.0, 15.0, -5.0]
    ay = [0.0, 0.0, 25.0, 35.0, 15.0, 25.0, -10.0]

    # Initialize path planner
    planner = PathPlanner(ax, ay, use_spline=True)

    # Initialize vehicle state as a NumPy array [x, y, yaw, u, v, r]
    state = np.array([ax[0], ay[0], planner.cyaw[0], 1, 0, 0])

    # Storage for visualization
    actual_x, actual_y = [], []
    pind = 0  # Initial index for nearest point search

    # Simulation loop
    for _ in range(500):
        if planner.check_goal(state):
            print("✅ Goal Reached!")
            break

        # Update position (basic motion model)
        state[0] += math.cos(state[2]) * state[3] * 0.1  # x
        state[1] += math.sin(state[2]) * state[3] * 0.1  # y

        # Save actual trajectory
        actual_x.append(state[0])
        actual_y.append(state[1])

        # Get reference trajectory
        ref_trajectory, nearest_index = planner.get_reference_trajectory(state)
        pind = nearest_index  # Update the previous index

        # Update yaw based on reference
        state[2] = ref_trajectory[0, 2]  # Next heading
        # Plot
        # Plot the reference trajectory along with the actual trajectory
        planner.plot_trajectory(state, actual_x, actual_y, ref_trajectory)

        time.sleep(0.05)

    plt.show()

if __name__ == "__main__":
    main()


    

