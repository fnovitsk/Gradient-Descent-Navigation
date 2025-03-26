import random
from typing import List, Tuple
import ast
import re
import math
import matplotlib.pyplot as plt #conda install conda-forge::matplotlib
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

#region generate environment

class Obstacle:
    def __init__(self, width: float, height: float, center: Tuple[float, float], pose: float):
        self.width = width
        self.height = height
        self.center = center
        self.pose = pose

    def __repr__(self):
        return f"Obstacle(width={self.width}, height={self.height}, center={self.center}, pose={self.pose})"

class Environment:
    def __init__(self, obstacles: List[Obstacle]):
        self.obstacles = obstacles

    def __repr__(self):
        return f"Environment(obstacles={self.obstacles})"

def generate_environment(number_of_obstacles: int) -> Environment:
    obstacles = []
    for _ in range(number_of_obstacles):
        # Randomly generate width and height in the range [0.5, 2]
        width = random.uniform(0.5, 2.0)
        height = random.uniform(0.5, 2.0)
        
        # Randomly generate the center coordinates (x, y) within the environment (20x20 area)
        x = random.uniform(0, 20)
        y = random.uniform(0, 20)
        center = (x, y)

        # Randomly generate the pose of the obstacle (0 to 360 degrees)
        pose = random.uniform(0, 2 * math.pi)

        # Create an obstacle with these properties
        obstacle = Obstacle(width, height, center, pose)
        obstacles.append(obstacle)
    
    # Create an environment instance with the generated obstacles
    return Environment(obstacles=obstacles)

#endregion

#region scene_to_file & scene_from_file & visualize scene

def scene_to_file(environment: Environment, filename: str):
    with open(filename, 'w') as file:
        file.write(repr(environment))

def scene_from_file(filename: str) -> Environment:
    with open(filename, 'r') as file:
        content = file.read()
        obstacles_data = re.findall(r"Obstacle\(width=(.*?), height=(.*?), center=\((.*?), (.*?)\), pose=(.*?)\)", content)
        obstacles = [
            Obstacle(
                width=float(width),
                height=float(height),
                center=(float(center_x), float(center_y)),
                pose=float(pose)
            ) for width, height, center_x, center_y, pose in obstacles_data
        ]
        return Environment(obstacles=obstacles)
    
def visualize_scene(environment: Environment, filename: str = None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.set_title('Environment Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    for obstacle in environment.obstacles:
        # Create the rectangle, centered around its center before rotation
        rect = patches.Rectangle(
            (obstacle.center[0] - obstacle.width / 2, obstacle.center[1] - obstacle.height / 2),  # bottom-left corner
            obstacle.width,
            obstacle.height,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        
        # Create a rotation transformation that rotates around the obstacle's center
        t = Affine2D().rotate_around(obstacle.center[0], obstacle.center[1], obstacle.pose)
        
        # Apply the transformation to the rectangle and add it to the plot
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)

    plt.grid(True)
    
    if filename:
        plt.savefig(filename)
    else:
        plt.savefig('environments/testEnvironmentVisualization.png')


#endregion
# testEnv = generate_environment(10)
# scene_to_file(testEnv, 'environments/testEnvironment.txt')
# visualize_scene(testEnv)
#visualize_scene(scene_from_file('environments/testEnvironment.txt'))

# env1 = generate_environment(1)
# scene_to_file(env1, 'environments/environment1.txt')
# visualize_scene(env1, 'environments/environment1.png')

# env2 = generate_environment(3)
# scene_to_file(env2, 'environments/environment2.txt')
# visualize_scene(env2, 'environments/environment2.png')

# env3 = generate_environment(5)
# scene_to_file(env3, 'environments/environment3.txt')
# visualize_scene(env3, 'environments/environment3.png')

# env4 = generate_environment(7)
# scene_to_file(env4, 'environments/environment4.txt')
# visualize_scene(env4, 'environments/environment4.png')

# env5 = generate_environment(10)
# scene_to_file(env5, 'environments/environment5.txt')
# visualize_scene(env5, 'environments/environment5.png')