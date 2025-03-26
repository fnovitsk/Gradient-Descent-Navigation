import numpy as np
import gtsam
from functools import partial
import argparse
import matplotlib.pyplot as plt
import networkx as nx

# Function for the first-order point system
def point_system(q_t, q_dot_t, delta_t=0.1):
    return q_t + q_dot_t * delta_t

def error_func(y: float, x: float, this: gtsam.CustomFactor, v: gtsam.Values, H):
    key_q_t = this.keys()[0]
    key_q_dot = this.keys()[1]
    q_t = v.atDouble(key_q_t)
    q_dot = v.atDouble(key_q_dot)
    predicted = point_system(q_t, q_dot)
    error = np.array([predicted - y])
    if H is not None:
        H[0] = np.array([[1.0]])  # Partial derivative with respect to q_t
        H[1] = np.array([[0.1]])  # Partial derivative with respect to q_dot
    return error

def constraint_error(target: float, this: gtsam.CustomFactor, v: gtsam.Values, H):
    key_q_t = this.keys()[0]
    q_t = v.atDouble(key_q_t)
    error = np.array([q_t - target])
    if H is not None:
        H[0] = np.array([[1.0]])  # Partial derivative with respect to q_t
    return error

def visualize_factor_graph(graph, constraints, save_path=None):
    G = nx.DiGraph()
    for i in range(graph.size()):
        factor = graph.at(i)
        keys = factor.keys()
        for key in keys:
            label = gtsam.DefaultKeyFormatter(key)
            G.add_node(label, type='state')
        if len(keys) > 1:
            for j in range(len(keys) - 1):
                G.add_edge(gtsam.DefaultKeyFormatter(keys[j]), gtsam.DefaultKeyFormatter(keys[j + 1]), color='gray', weight=1.0)

    # Add constraint nodes and edges
    for idx, (key, constraint_value) in enumerate(constraints.items()):
        constraint_label = f"constraint_{idx}"
        G.add_node(constraint_label, type='constraint')
        G.add_edge(constraint_label, gtsam.DefaultKeyFormatter(key), color='red', weight=2.0)

    plt.figure(figsize=(15, 10))  # Increased figure size for better readability
    pos = nx.shell_layout(G)  # Use shell layout for better spacing and clarity
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    node_colors = ['lightblue' if G.nodes[n].get('type') == 'state' else 'darkred' for n in G.nodes()]
    node_shapes = ['o' if G.nodes[n].get('type') == 'state' else 's' for n in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_size=800, node_color=node_colors, font_size=10, font_weight='bold', edge_color=edge_colors, width=edge_weights)

    # Annotate constraint nodes for clarity
    for idx, (key, _) in enumerate(constraints.items()):
        constraint_label = f"constraint_{idx}"
        x, y = pos[constraint_label]
        plt.text(x, y + 0.1, f"Constraint {idx}", fontsize=12, color='red', ha='center', fontweight='bold')

    plt.title("Factor Graph with Emphasized Constraints")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_factor_graph_structure(graph):
    print("Factor Graph Structure:")
    for i in range(graph.size()):
        factor = graph.at(i)
        keys = factor.keys()
        print(f"Factor {i}: Connected keys -> {[gtsam.DefaultKeyFormatter(k) for k in keys]}")

def main(start, goal, T, x0_in, x1_in):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # Initialize the initial point and noise model
    q_dot = 0.5  # Initial guess for q_dot
    kq_dot = gtsam.symbol('q', 0)
    values.insert(kq_dot, q_dot)
    sigma = 1
    noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
    very_strong_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, 0.01)  # Much stronger noise model for constraints

    # Generate ground truth and noisy observations
    GT = []
    Z = []
    q_t = start[0]
    kq_t_prev = None
    for t in range(T):
        # Create a new symbol for each time step's state
        kq_t = gtsam.symbol('x', t)
        values.insert(kq_t, q_t)

        # Update ground truth to gradually move from start to goal
        q_goal = start[0] + (goal[0] - start[0]) * (t / (T - 1))
        GT.append(q_goal)
        Z.append(q_goal + np.random.normal(0.0, sigma))

        # Add a factor between the current state and q_dot
        keys = gtsam.KeyVector([kq_t, kq_dot])
        gf = gtsam.CustomFactor(noise_model, keys, partial(error_func, Z[t], q_t))
        graph.add(gf)

        # Connect previous state to the current state to enforce dynamics
        if kq_t_prev is not None:
            graph.add(gtsam.BetweenFactorDouble(kq_t_prev, kq_t, q_dot * 0.1, noise_model))
        kq_t_prev = kq_t

        q_t = point_system(q_t, q_dot)

    # Add extra constraints at T/3 and 2T/3
    kq_t_1 = gtsam.symbol('x', T // 3)
    kq_t_2 = gtsam.symbol('x', 2 * T // 3)
    constraint_1 = gtsam.CustomFactor(very_strong_noise_model, gtsam.KeyVector([kq_t_1]), partial(constraint_error, x0_in[0]))
    constraint_2 = gtsam.CustomFactor(very_strong_noise_model, gtsam.KeyVector([kq_t_2]), partial(constraint_error, x1_in[0]))
    graph.add(constraint_1)
    graph.add(constraint_2)

    # Store constraint information for visualization purposes
    constraints = {
        kq_t_1: x0_in[0],
        kq_t_2: x1_in[0]
    }

    # Visualize factor graph structure
    visualize_factor_graph_structure(graph)
    visualize_factor_graph(graph, constraints, save_path="images/fg_traj_opt_2/factor_graph_with_constraints.png")

    # Optimizing
    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    q_dot = result.atDouble(kq_dot)
    print("Optimized q_dot: ", q_dot)

    # Plotting ground truth and noisy data
    plt.figure(figsize=(6, 4))
    plt.plot(range(T), GT, label='Ground Truth', color='blue')
    plt.scatter(range(T), Z, label='Noisy Observations', color='red', s=10)
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.title('Ground Truth and Observations')
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/fg_traj_opt_2/trajectory_with_constraints.png")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory Optimization via Factor Graphs with Extra Constraints')
    parser.add_argument('--start', type=float, nargs=2, required=True, help='Start state (x, y)')
    parser.add_argument('--goal', type=float, nargs=2, required=True, help='Goal state (x, y)')
    parser.add_argument('--T', type=int, required=True, help='Number of states')
    parser.add_argument('--x0', type=float, nargs=2, required=True, help='Intermediate state x0_in (x, y)')
    parser.add_argument('--x1', type=float, nargs=2, required=True, help='Intermediate state x1_in (x, y)')
    args = parser.parse_args()

    main(args.start, args.goal, args.T, args.x0, args.x1)
