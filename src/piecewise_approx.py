import numpy as np
import matplotlib.pyplot as plt

from interpolation import lagrange_interp
from error import max_norm_error


def piecewise_lagrange_interp(outer_nodes, f, n=4, m=100):
    """n = nodes per subinterval (degree n-1), m = evaluation points per subinterval."""

    if len(np.unique(outer_nodes)) != len(outer_nodes):
        raise ValueError("outer_nodes must be distinct")

    x_parts, y_parts = [], []

    for i in range(len(outer_nodes) - 1):
        nodes = np.linspace(outer_nodes[i], outer_nodes[i + 1], n)
        eval_x = np.linspace(outer_nodes[i], outer_nodes[i + 1], m)

        x_parts.append(eval_x)
        y_parts.append(lagrange_interp(nodes, f(nodes), eval_x))

    return np.concatenate(x_parts), np.concatenate(y_parts)


def plot_piecewise_equidistant(outer_nodes, f):
    x_vals, y_vals = piecewise_lagrange_interp(outer_nodes, f)

    plt.plot(x_vals, f(x_vals), label="f(x), Runge's function", color="black")
    plt.plot(x_vals, y_vals, label="Equidistant", color="red")
    plt.title("Piecewise Lagrange interpolation with equidistant nodes")
    plt.xlabel("x"); plt.ylabel("y")
    plt.grid(); plt.legend()
    plt.show()


def f(x):
    return 1 / (1 + x**2)


if __name__ == "__main__":
    a, b = -5, 5

    K_vals = np.unique(np.logspace(1, 3, 10).round().astype(int))
    error = []

    for K in K_vals:
        outer_nodes = np.linspace(a, b, K)
        x_vals, y_vals = piecewise_lagrange_interp(outer_nodes, f)
        error.append(max_norm_error(f, x_vals, y_vals))

    plt.semilogy(K_vals, error)
    plt.show()