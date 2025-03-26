import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple
import matplotlib.animation as animation
import argparse
from generating_environment import Environment, Obstacle, scene_from_file, visualize_scene

# Gradient of the attractive potential function
def gradient_attractive_potential(q: np.ndarray, goal: np.ndarray, zeta: float = 1.0) -> np.ndarray:
    return zeta * (q - goal)

# Gradient of the repulsive potential function
def gradient_repulsive_potential(q: np.ndarray, obstacles: list, eta: float = 100.0, d0: float = 2.0) -> np.ndarray:
    grad_U_r = np.zeros_like(q)
    for obstacle in obstacles:
        d = np.linalg.norm(q - np.array(obstacle.center))
        if d < d0:
            grad_U_r -= eta * ((1.0 / d) - (1.0 / d0)) * (1.0 / (d ** 2)) * (q - np.array(obstacle.center)) / d
    return grad_U_r

# Gradient of the combined potential function
def gradient_potential(q: np.ndarray, goal: np.ndarray, obstacles: list, zeta: float = 1.0, eta: float = 100.0, d0: float = 2.0) -> np.ndarray:
    grad_U_a = gradient_attractive_potential(q, goal, zeta)
    grad_U_r = gradient_repulsive_potential(q, obstacles, eta, d0)
    return grad_U_a + grad_U_r

# Gradient descent algorithm to move towards the goal
def gradient_descent(start: Tuple[float, float], goal: Tuple[float, float], environment: Environment, alpha: float = 0.05, tol: float = 1e-2, max_iters: int = 10000):
    q = np.array(start, dtype=float)
    i = 0
    path = [q.copy()]
    
    while i < max_iters:
        grad_U = gradient_potential(q, goal, environment.obstacles)
        print(f"Iteration {i}: Position {q}, Gradient {grad_U}")
        # Stop if the gradient is small enough (i.e., we've reached the goal region)
        if np.linalg.norm(grad_U) < tol:
            break
        
        # Update position using gradient descent
        q += -alpha * grad_U
        
        # Store the new position
        path.append(q.copy())
        i += 1
    
    return path

# Visualization of the path taken by the robot
def visualize_path(path: list, environment: Environment, filename: str = None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.set_title('Path Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Plot obstacles
    for obstacle in environment.obstacles:
        rect = patches.Rectangle(
            (obstacle.center[0] - obstacle.width / 2, obstacle.center[1] - obstacle.height / 2),
            obstacle.width,
            obstacle.height,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        t = patches.Affine2D().rotate_around(obstacle.center[0], obstacle.center[1], obstacle.pose)
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)
    
    # Plot path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'b.-', label='Path')
    ax.plot(path[0, 0], path[0, 1], 'go', label='Start')
    ax.plot(path[-1, 0], path[-1, 1], 'ro', label='Goal')
    
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# Animation of the robot moving through the environment
def animate_path(path: list, environment: Environment, filename: str = None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.set_title('Path Animation')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Plot obstacles
    for obstacle in environment.obstacles:
        rect = patches.Rectangle(
            (obstacle.center[0] - obstacle.width / 2, obstacle.center[1] - obstacle.height / 2),
            obstacle.width,
            obstacle.height,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        t = patches.Affine2D().rotate_around(obstacle.center[0], obstacle.center[1], obstacle.pose)
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)
    
    # Initialization function for animation
    def init():
        line.set_data([], [])
        return line,

    # Update function for animation
    def update(frame):
        line.set_data(path[:frame, 0], path[:frame, 1])
        return line,
    
    # Plot path
    path = np.array(path)
    line, = ax.plot([], [], 'b-', label='Path')
    ax.plot(path[0, 0], path[0, 1], 'go', label='Start')
    ax.plot(path[-1, 0], path[-1, 1], 'ro', label='Goal')
    plt.legend()
    plt.grid(True)

    ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, repeat=False)
    ani.save(filename, writer='imagemagick')
    plt.close()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Gradient Descent Path Planning")
    parser.add_argument('--start', type=float, nargs=2, required=True, help="Start coordinates (e.g., --start 0.0 0.0)")
    parser.add_argument('--goal', type=float, nargs=2, required=True, help="Goal coordinates (e.g., --goal 5.0 5.0)")
    args = parser.parse_args()

    # Load environment from file
    env = scene_from_file('environments/testEnvironment.txt')
    
    # Define start and goal positions from input arguments
    start = tuple(args.start)
    goal = tuple(args.goal)
    
    # Perform gradient descent to find path
    path = gradient_descent(start, goal, env)
    
    # Visualize the resulting path
    visualize_path(path, env, 'images/potentialFunctions/testPathVisualization.png')
    animate_path(path, env, 'images/potentialFunctions/testPathAnimation.gif')
