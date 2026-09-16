import numpy as np
import matplotlib.pyplot as plt

from interpolation import lagrange_interp, chebyshev_nodes


def two_norm_error(f, x_eval, y_vals):
    """Discrete L2 error between f and the approximation y_vals sampled on nodes."""
    a, b = x_eval[0], x_eval[-1]
    N = len(x_eval)

    return np.sqrt((b - a)/N) * np.sqrt(np.sum((f(x_eval) - y_vals)**2))

def two_norm_error_square(f, x_eval, y_vals):
    """Discrete L2 error squared between f and the approximation y_vals sampled on nodes."""
    a, b = x_eval[0], x_eval[-1]
    N = len(x_eval)

    return (b - a)/N * np.sum((f(x_eval) - y_vals)**2)

def max_norm_error(f, x_eval, y_vals):
    """Discrete max error between f and the approximation y_vals sampled on nodes."""
    return np.max(np.abs(f(x_eval) - y_vals))


def lagrange_error(f, n, a, b, N, chebyshev=True):
    """Sample the degree-n Lagrange interpolant of f on a grid of N points."""
    if chebyshev:
        interp_nodes = chebyshev_nodes(a, b, n+1)
    else:
        interp_nodes = np.linspace(a, b, n+1)

    x_vals = np.linspace(a, b, N)
    y_vals = lagrange_interp(interp_nodes, f(interp_nodes), x_vals)

    return x_vals, y_vals


if __name__ == "__main__":

    a = 0; b = 1; n = 50

    n_vals = np.arange(1, n)

    def f(x):
        return np.cos(2*np.pi*x)

    def g(x):
        return np.exp(3*x) * np.sin(2*x)

    equidistant = [lagrange_error(f, k, a, b, 100*k, chebyshev=False) for k in n_vals]
    chebyshev = [lagrange_error(f, k, a, b, 100*k, chebyshev=True) for k in n_vals]

    error_2_norm_equidistant = np.array([two_norm_error(f, x, y) for x, y in equidistant])
    error_2_norm_chebyshev = np.array([two_norm_error(f, x, y) for x, y in chebyshev])
    error_max_norm_equidistant = np.array([max_norm_error(f, x, y) for x, y in equidistant])
    error_max_norm_chebyshev = np.array([max_norm_error(f, x, y) for x, y in chebyshev])

    plt.semilogy(n_vals, error_2_norm_equidistant, label="2 norm, equidistant nodes")
    plt.semilogy(n_vals, error_2_norm_chebyshev, label="2 norm, Chebyshev nodes")
    plt.semilogy(n_vals, error_max_norm_equidistant, label="max norm, equidistant nodes")
    plt.semilogy(n_vals, error_max_norm_chebyshev, label="max norm, Chebyshev nodes")

    plt.xlabel("n")
    plt.ylabel("error")
    plt.title("Interpolation error vs. polynomial degree")
    plt.grid()
    plt.legend()
    plt.show()
