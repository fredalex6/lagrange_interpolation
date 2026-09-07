import numpy as np
import matplotlib.pyplot as plt


def lagrange_interp(nodes, vals, x_vals=[]):
    if len(np.unique(nodes)) != len(nodes):
        raise ValueError("nodes must be distinct")

    if len(x_vals) == 0:
        x_vals = nodes

    y_vals = np.zeros(len(x_vals))

    for j in range(len(nodes)):
        p_j = vals[j]

        for k in range(len(nodes)):
            if k == j: continue
            p_j *= (x_vals - nodes[k]) / (nodes[j] - nodes[k])

        y_vals += p_j

    return y_vals

def f(x):
    return 1 / (1 + x**2)


def chebyshev_nodes(a, b, n):
    x_vals = np.array([np.cos(np.pi/n * (k + 1/2)) for k in range(n)])
    x_thilde_vals = 1/2 * (a + b) + 1/2 * (b - a) * x_vals

    return x_thilde_vals

def plot_equidistant_chebyshev(f, x_equidistant, x_chebyshev, n, N = 1000):
    x_vals = np.linspace(x_equidistant[0], x_equidistant[-1], N)

    y_equidistant = lagrange_interp(x_equidistant, f(x_equidistant), x_vals)
    y_chebyshev = lagrange_interp(x_chebyshev, f(x_chebyshev), x_vals)

    plt.plot(x_vals, f(x_vals), label="f(x), Runges function", color="black")
    plt.plot(x_vals, y_equidistant, label="Equidistant", color="green")
    plt.plot(x_vals, y_chebyshev, label="Chebyshev", color="red")

    plt.title(f"Lagrange interpolation with Chebyshev and equidistant nodes, n = {n}")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    a = -5; b = 5; n = 10

    x_equidistant = np.linspace(a, b, n)
    x_chebyshev = chebyshev_nodes(a, b, n)

    plot_equidistant_chebyshev(f, x_equidistant, x_chebyshev, n)

    for n_i in [15, 20, 30]:
        x_equidistant = np.linspace(a, b, n_i)
        x_chebyshev = chebyshev_nodes(a, b, n_i)

        plot_equidistant_chebyshev(f, x_equidistant, x_chebyshev, n_i)


