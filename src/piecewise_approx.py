import numpy as np
import matplotlib.pyplot as plt

from interpolation import lagrange_interp, chebyshev_nodes
from error import max_norm_error


def piecewise_lagrange_interp(outer_nodes, f, n, x):
    """Evaluating the piecewise interpolation polynomial in x."""
    y = np.empty_like(x, dtype=float)

    for i in range(len(outer_nodes) - 1):

        if i == len(outer_nodes) - 2:
            mask = (x >= outer_nodes[i]) & (x <= outer_nodes[i+1]) # last includes b
        else:
            mask = (x >= outer_nodes[i]) & (x < outer_nodes[i+1])

        if not mask.any():
            continue

        nodes = np.linspace(outer_nodes[i], outer_nodes[i+1], n+1)
        y[mask] = lagrange_interp(nodes, f(nodes), x[mask])

    return y


def plot_piecewise_equidistant(outer_nodes, f, n, x):
    x_vals, y_vals = piecewise_lagrange_interp(outer_nodes, f, n, x)

    plt.plot(x_vals, f(x_vals), label="f(x), Runge's function", color="black")
    plt.plot(x_vals, y_vals, label="Equidistant", color="red")
    plt.title("Piecewise Lagrange interpolation with equidistant nodes")
    plt.xlabel("x"); plt.ylabel("y")
    plt.grid(); plt.legend()
    plt.show()


# one of the test functions from b)
def f(x):
    return np.cos(2*np.pi*x)


if __name__ == "__main__":
    a, b = -5, 5; n_max = 10

    K_vals = np.unique(np.logspace(0.5, 3, 15).round().astype(int))
    x_eval = np.linspace(a, b, 100*n_max)

    for n in range(1, n_max+1):
        error = []

        for K in K_vals:
            outer_nodes = np.linspace(a, b, K+1)
            y_vals = piecewise_lagrange_interp(outer_nodes, f, n, x_eval)
            error.append(max_norm_error(f, x_eval, y_vals))

        plt.title("Interpolation error in the max norm for piecewise interpolation")
        plt.loglog(K_vals, error, label=f"n = {n}")

    plt.xlabel("K")
    plt.ylabel(r"$||f-p_n||_{\infty}$")

    plt.legend()
    plt.show()


    for n in [1, 3]:
        total_nodes = []
        error = []

        for K in K_vals:
            outer_nodes = np.linspace(a, b, K+1)
            y_vals = piecewise_lagrange_interp(outer_nodes, f, n, x_eval)

            total_nodes.append(K*n + 1)
            error.append(max_norm_error(f, x_eval, y_vals))

        
        plt.loglog(total_nodes, error, label=f"piecewise, n = {n}")


    total_nodes_global = []
    error_equidistant = []
    error_chebyshev = []

    for n_global in range(2, 41, 2):
        nodes_equidistant = np.linspace(a, b, n_global)
        nodes_chebyshev = chebyshev_nodes(a, b, n_global+1)

        y_equidistant = lagrange_interp(nodes_equidistant, f(nodes_equidistant), x_eval)
        y_chebyshev = lagrange_interp(nodes_chebyshev, f(nodes_chebyshev), x_eval)

        total_nodes_global.append(n_global + 1)
        error_equidistant.append(max_norm_error(f, x_eval, y_equidistant))
        error_chebyshev.append(max_norm_error(f, x_eval, y_chebyshev))


    plt.loglog(total_nodes_global, error_equidistant, label="max error, equidistant nodes")
    plt.loglog(total_nodes_global, error_chebyshev, label="max error, chebyshev nodes")

    plt.title("Interpolation error in the max norm for piecewise vs. global interpolation")
    plt.xlabel("total nodes, N")
    plt.ylabel(r"$||f-p_n||_{\infty}$")

    plt.legend()
    plt.show()
    