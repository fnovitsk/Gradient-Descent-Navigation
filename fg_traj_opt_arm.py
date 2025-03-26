import gtsam
import numpy as np
import argparse
import matplotlib.pyplot as plt
from gtsam import NonlinearFactorGraph, Values, LevenbergMarquardtOptimizer, Point2
from gtsam import noiseModel

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='2-Link Robot Arm Trajectory Optimization')
    parser.add_argument('--start', nargs=2, type=float, required=True, help='Start configuration (θ0 θ1)')
    parser.add_argument('--goal', nargs=2, type=float, required=True, help='Goal configuration (θ0 θ1)')
    parser.add_argument('--T', type=int, required=True, help='Number of timesteps')
    return parser.parse_args()

# Define the main function for the trajectory optimization
def main():
    args = parse_args()
    start = np.array(args.start)
    goal = np.array(args.goal)
    T = args.T

    # Noise model for constraints
    constraint_noise = noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01]))

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add start and goal priors
    graph.add(gtsam.PriorFactorPoint2(0, Point2(start[0], start[1]), constraint_noise))
    graph.add(gtsam.PriorFactorPoint2(T-1, Point2(goal[0], goal[1]), constraint_noise))

    # Add smoothness constraints for each time step
    for t in range(T - 1):
        graph.add(gtsam.BetweenFactorPoint2(t, t + 1, Point2(0, 0), constraint_noise))

    # Initial estimate for the optimization
    initial = Values()
    for t in range(T):
        theta0 = start[0] + (goal[0] - start[0]) * t / (T - 1)
        theta1 = start[1] + (goal[1] - start[1]) * t / (T - 1)
        initial.insert(t, Point2(theta0, theta1))

    # Optimize using Levenberg-Marquardt
    optimizer = LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    # Extract the results
    theta0_vals = []
    theta1_vals = []
    for t in range(T):
        theta = result.atPoint2(t)
        theta0_vals.append(theta[0])
        theta1_vals.append(theta[1])

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(theta0_vals, theta1_vals, marker='o')
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.title('2-Link Robot Arm Trajectory')
    plt.grid(True)

    # Visualization of the factor graph structure
    plt.subplot(1, 2, 2)
    plt.title('Factor Graph Visualization')
    for t in range(T - 1):
        plt.plot([t, t + 1], [0, 0], 'bo-', label='Smoothness Factor' if t == 0 else '')
    plt.scatter([0, T-1], [0, 0], color='r', label='Start/Goal Prior')
    plt.xlabel('Timestep')
    plt.ylabel('Factor Value')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('images/fg_traj_opt_arm/robot_with_factor_graph.png')

if __name__ == '__main__':
    main()
