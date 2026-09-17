import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

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


# one of the test functions from b)
def f(x):
    return np.cos(2*np.pi*x)


def piecewise_perf(f, a, b, n, K, x_eval, runs=10):
    """Measures the performance of piecewise interpolation of f on [a, b], given n and K"""

    best_time = np.inf

    # find the best time over multiple runs given same params 
    for _ in range(runs):
        start_time = perf_counter()
        outer_nodes = np.linspace(a, b, K+1)
        y_vals = piecewise_lagrange_interp(outer_nodes, f, n, x_eval)
        end_time = perf_counter()

        best_time = min(best_time, end_time - start_time)

    return best_time


def global_perf(f, a, b, n, x_eval, runs=10):
    """Measures the performance of global interpolation of f on [a, b], given n"""

    c_nodes = chebyshev_nodes(a, b, n+1)
    best_time = np.inf

    # find the best time over multiple runs given same params 
    for _ in range(runs):
        start_time = perf_counter()
        y_vals = lagrange_interp(c_nodes, f(c_nodes), x_eval)
        end_time = perf_counter()

        best_time = min(best_time, end_time - start_time)

    return best_time


if __name__ == "__main__":
    a, b = 0, 1
    n_max_pw = 10; n_max_global = 100; # pw = piecewise and global = not piecewise

    K_vals = np.unique(np.logspace(0, 3, 30).round().astype(int))

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for n in range(1, n_max_pw+1):
        error = []

        for K in K_vals:
            outer_nodes = np.linspace(a, b, K+1)
            x_eval_pw = np.linspace(a, b, 100*K + 1) # 100 points per subinterval
            y_vals = piecewise_lagrange_interp(outer_nodes, f, n, x_eval_pw)
            error.append(max_norm_error(f, x_eval_pw, y_vals))

        ax.set_title("Interpolation error in the max norm for piecewise interpolation")
        ax.loglog(K_vals, error, label=f"n = {n}")

    ax.set_xlabel("K")
    ax.set_ylabel(r"$||f-p_n||_{\infty}$")

    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for n in range(1, n_max_pw+1):
        total_nodes_piecewise, error = [], []

        for K in K_vals:
            outer_nodes = np.linspace(a, b, K+1)
            x_eval_pw = np.linspace(a, b, 100*K + 1) # 100 points per subinterval
            y_vals = piecewise_lagrange_interp(outer_nodes, f, n, x_eval_pw)

            e = max_norm_error(f, x_eval_pw, y_vals)
            total_nodes_piecewise.append(K*n + 1); error.append(e)

        ax.loglog(total_nodes_piecewise, error, label=f"n = {n}")


    total_nodes_global = []
    error_equidistant = []
    error_chebyshev = []

    x_eval_global = np.linspace(a, b, 100*n_max_global + 1)

    for n_global in range(2, 100):
        nodes_equidistant = np.linspace(a, b, n_global+1)
        nodes_chebyshev = chebyshev_nodes(a, b, n_global+1)

        y_equidistant = lagrange_interp(nodes_equidistant, f(nodes_equidistant), x_eval_global)
        y_chebyshev = lagrange_interp(nodes_chebyshev, f(nodes_chebyshev), x_eval_global)

        total_nodes_global.append(n_global + 1)
        error_equidistant.append(max_norm_error(f, x_eval_global, y_equidistant))
        error_chebyshev.append(max_norm_error(f, x_eval_global, y_chebyshev))


    ax.loglog(total_nodes_global, error_equidistant, label="global, equidistant nodes (n = N-1)", color="k", linestyle="--")
    ax.loglog(total_nodes_global, error_chebyshev, label="global, Chebyshev nodes (n = N-1)", color="r", linestyle="--")

    ax.set_xlabel(r"total nodes, $N = K*n + 1$")
    ax.set_ylabel(r"$||f-p_n||_{\infty}$")

    fig.subplots_adjust(top=0.80)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.93),
            ncol=4, fontsize="small", frameon=False)
    fig.suptitle("Interpolation error in the max norm for piecewise vs. global interpolation",
                y=0.975)
    plt.show()

    # measure performance
    N = 10_000
    x_eval = np.linspace(a, b, N+1)

    for n, K in [(1, 100), (1, 1000), (3, 20), (3, 100), (10, 10)]:
        outer_nodes = np.linspace(a, b, K+1)
        error = max_norm_error(f, x_eval, piecewise_lagrange_interp(outer_nodes, f, n, x_eval))
        print(f"piecewise (n={n}, K={K}). N={K*n+1}, error={error:.2e}, time={piecewise_perf(f, a, b, n, K, x_eval)*1e3:.2f} ms")

    for n in [10, 20, 40]:
        nodes = chebyshev_nodes(a, b, n+1)
        error = max_norm_error(f, x_eval, lagrange_interp(nodes, f(nodes), x_eval))
        print(f"global (n={n}). N={n+1}, error={error:.2e}, time={global_perf(f, a, b, n, x_eval)*1e3:.2f} ms")
    