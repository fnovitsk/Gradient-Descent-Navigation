import argparse
from functools import partial
import numpy as np
import gtsam
from typing import List, Optional
import matplotlib.pyplot as plt

# "True" function with its respective parameters for linear function
def f_linear(x, m=0.6, b=1.5):
    return m * x + b

# "True" function with its respective parameters for cubic function
def f_cubic(x, a=0.045, b=0.2, c=0.7, d=4.86):
    return a * x**3 + b * x**2 + c * x + d

def error_func_linear(y: np.ndarray, x: np.ndarray, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    """
    Error function for linear model
    """
    key_m = this.keys()[0]
    key_b = this.keys()[1]
    m = v.atDouble(key_m)
    b = v.atDouble(key_b)
    yp = m * x + b
    error = yp - y
    if H is not None:
        H[0] = np.eye(1) * x # derr / dm
        H[1] = np.eye(1) # derr / db
    return error

def error_func_cubic(y: np.ndarray, x: np.ndarray, this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray]):
    """
    Error function for cubic model
    """
    key_a = this.keys()[0]
    key_b = this.keys()[1]
    key_c = this.keys()[2]
    key_d = this.keys()[3]
    a = v.atDouble(key_a)
    b = v.atDouble(key_b)
    c = v.atDouble(key_c)
    d = v.atDouble(key_d)
    yp = a * x**3 + b * x**2 + c * x + d
    error = yp - y
    if H is not None:
        H[0] = np.eye(1) * x**3 # derr / da
        H[1] = np.eye(1) * x**2 # derr / db
        H[2] = np.eye(1) * x # derr / dc
        H[3] = np.eye(1) # derr / dd
    return error

if __name__ == '__main__':
    # Setting up argument parsing
    parser = argparse.ArgumentParser(description='Run polynomial optimization.')
    parser.add_argument('--initial', nargs=4, type=float, required=True, help='Initial values for a, b, c, d')

    args = parser.parse_args()
    a, b, c, d = args.initial

    graph_linear = gtsam.NonlinearFactorGraph()
    graph_cubic = gtsam.NonlinearFactorGraph()
    v_linear = gtsam.Values()
    v_cubic = gtsam.Values()
    T = 100

    # Linear model initialization
    GT_linear = [] # The ground truth, for comparison
    Z_linear = [] # GT + Normal(0, Sigma)
    m = 1
    b = -1
    km = gtsam.symbol('m', 0)
    kb = gtsam.symbol('b', 0)
    v_linear.insert(km, m)
    v_linear.insert(kb, b)
    sigma = 1 #NOISE
    noise_model_linear = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
    for i in range(T):
        GT_linear.append(f_linear(i))
        Z_linear.append(f_linear(i) + np.random.normal(0.0, sigma))
        keys_linear = gtsam.KeyVector([km, kb])
        gf_linear = gtsam.CustomFactor(noise_model_linear, keys_linear, partial(error_func_linear, np.array([Z_linear[i]]), np.array([i])))
        graph_linear.add(gf_linear)

    # Cubic model initialization
    GT_cubic = [] # The ground truth, for comparison
    Z_cubic = [] # GT + Normal(0, Sigma)
    ka = gtsam.symbol('a', 0)
    kb = gtsam.symbol('b', 0)
    kc = gtsam.symbol('c', 0)
    kd = gtsam.symbol('d', 0)
    v_cubic.insert(ka, a)
    v_cubic.insert(kb, b)
    v_cubic.insert(kc, c)
    v_cubic.insert(kd, d)
    noise_model_cubic = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
    for i in range(T):
        GT_cubic.append(f_cubic(i))
        Z_cubic.append(f_cubic(i) + np.random.normal(0.0, sigma))
        keys_cubic = gtsam.KeyVector([ka, kb, kc, kd])
        gf_cubic = gtsam.CustomFactor(noise_model_cubic, keys_cubic, partial(error_func_cubic, np.array([Z_cubic[i]]), np.array([i])))
        graph_cubic.add(gf_cubic)

    # Optimizing linear model
    result_linear = gtsam.LevenbergMarquardtOptimizer(graph_linear, v_linear).optimize()
    m = result_linear.atDouble(km)
    b = result_linear.atDouble(kb)
    print("Linear Model - m: ", m, " b: ", b)
    for i in range(T):
        print("Linear", i, GT_linear[i], Z_linear[i])

    # Optimizing cubic model
    result_cubic = gtsam.LevenbergMarquardtOptimizer(graph_cubic, v_cubic).optimize()
    a = result_cubic.atDouble(ka)
    b = result_cubic.atDouble(kb)
    c = result_cubic.atDouble(kc)
    d = result_cubic.atDouble(kd)
    print("Cubic Model - a: ", a, " b: ", b, " c: ", c, " d: ", d)
    for i in range(T):
        print("Cubic", i, GT_cubic[i], Z_cubic[i])

    # Plotting ground truth, noisy data, and optimized models for linear and cubic models
    plt.figure(figsize=(12, 6))

    # Plot linear model
    plt.subplot(1, 2, 1)
    plt.plot(range(T), GT_linear, label='Ground Truth (Linear)', color='blue')
    plt.scatter(range(T), Z_linear, label='Noisy Observations (Linear)', color='red', s=10)
    plt.plot(range(T), [m * x + b for x in range(T)], label='Optimized Model (Linear)', color='purple')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Model Ground Truth, Observations, and Optimized Model')
    plt.legend()

    # Plot cubic model
    plt.subplot(1, 2, 2)
    plt.plot(range(T), GT_cubic, label='Ground Truth (Cubic)', color='green')
    plt.scatter(range(T), Z_cubic, label='Noisy Observations (Cubic)', color='orange', s=10)
    plt.plot(range(T), [a * x**3 + b * x**2 + c * x + d for x in range(T)], label='Optimized Model (Cubic)', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Model Ground Truth, Observations, and Optimized Model')
    plt.legend()

    plt.tight_layout()
    plt.savefig("images/fg_polynomial/fg_polynomial.png")
    plt.show()
