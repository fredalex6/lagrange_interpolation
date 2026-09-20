import numpy as np
import matplotlib.pyplot as plt
import math

from interpolation import lagrange_interp, chebyshev_nodes


def factorial(k):
    return np.array([math.factorial(int(j)) for j in np.atleast_1d(k)], dtype=float)

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


def error_bound_f(a, b, n, chebyshev=False, norm="max"):
    max_bound = 1/(4*(n+1)) * ((2*np.pi*(b-a))/n)**(n+1) # equidistant bound

    if chebyshev == True:
        max_bound = 2/factorial(n+1) * (np.pi/2)**(n+1)

    if norm == "two":
        max_bound *= np.sqrt(b-a)

    return max_bound


def error_bound_g(a, b, n, chebyshev=False, norm="max"):
    max_bound = np.exp(3*np.pi/4)/(4*(n+1)) * (np.sqrt(13)*(b-a)/n)**(n+1) # equidistant bound

    if chebyshev == True:
        max_bound = 2*np.exp(3*np.pi/4)/factorial(n+1) * (np.sqrt(13)*(b-a)/4)**(n+1)

    if norm == "two":
        max_bound *= np.sqrt(b-a)

    return max_bound


if __name__ == "__main__":

    # parameters for func f
    a_f = 0; b_f = 1; n_max = 50

    # parameters for func g
    a_g = 0; b_g = np.pi/4; n_max = 50

    n_vals = np.arange(1, n_max)

    def f(x):
        return np.cos(2*np.pi*x)

    def g(x):
        return np.exp(3*x) * np.sin(2*x)

    # nodes for interpolation of f
    e_nodes_f = [lagrange_error(f, k, a_f, b_f, 100*n_max, chebyshev=False) for k in n_vals]
    c_nodes_f = [lagrange_error(f, k, a_f, b_f, 100*n_max, chebyshev=True) for k in n_vals]

    # nodes for interpolation of g
    e_nodes_g = [lagrange_error(g, k, a_g, b_g, 100*n_max, chebyshev=False) for k in n_vals]
    c_nodes_g = [lagrange_error(g, k, a_g, b_g, 100*n_max, chebyshev=True) for k in n_vals]

    # 2 norm and max norm error for interpolation of f
    error_2_norm_equidistant_f = np.array([two_norm_error(f, x, y) for x, y in e_nodes_f])
    error_2_norm_chebyshev_f = np.array([two_norm_error(f, x, y) for x, y in c_nodes_f])
    error_max_norm_equidistant_f = np.array([max_norm_error(f, x, y) for x, y in e_nodes_f])
    error_max_norm_chebyshev_f = np.array([max_norm_error(f, x, y) for x, y in c_nodes_f])

    # 2 norm and max norm error for interpolation of g
    error_2_norm_equidistant_g = np.array([two_norm_error(g, x, y) for x, y in e_nodes_g])
    error_2_norm_chebyshev_g = np.array([two_norm_error(g, x, y) for x, y in c_nodes_g])
    error_max_norm_equidistant_g = np.array([max_norm_error(g, x, y) for x, y in e_nodes_g])
    error_max_norm_chebyshev_g = np.array([max_norm_error(g, x, y) for x, y in c_nodes_g])

    # plotting for interpolation of f
    plt.semilogy(n_vals, error_2_norm_equidistant_f, label="2 norm, equidistant nodes")
    plt.semilogy(n_vals, error_2_norm_chebyshev_f, label="2 norm, Chebyshev nodes")
    plt.semilogy(n_vals, error_max_norm_equidistant_f, label="max norm, equidistant nodes")
    plt.semilogy(n_vals, error_max_norm_chebyshev_f, label="max norm, Chebyshev nodes")

    plt.semilogy(n_vals, error_bound_f(a_f, b_f, n_vals, chebyshev=False), "C2--", label="bound on max norm, equidistant")
    plt.semilogy(n_vals, error_bound_f(a_f, b_f, n_vals, chebyshev=True), "C3--", label="bound on max norm, Chebyshev")

    plt.xlabel("n")
    plt.ylabel("error")
    plt.title(r"Interpolation error vs. polynomial degree, $f(x) = \cos(2\pi x)$")
    plt.legend()
    plt.show()

    # plotting for interpolation of g
    plt.semilogy(n_vals, error_2_norm_equidistant_g, label="2 norm, equidistant nodes")
    plt.semilogy(n_vals, error_2_norm_chebyshev_g, label="2 norm, Chebyshev nodes")
    plt.semilogy(n_vals, error_max_norm_equidistant_g, label="max norm, equidistant nodes")
    plt.semilogy(n_vals, error_max_norm_chebyshev_g, label="max norm, Chebyshev nodes")

    plt.semilogy(n_vals, error_bound_g(a_g, b_g, n_vals, chebyshev=False), "C2--", label="bound on max norm, equidistant")
    plt.semilogy(n_vals, error_bound_g(a_g, b_g, n_vals, chebyshev=True), "C3--", label="bound on max norm, Chebyshev")

    plt.xlabel("n")
    plt.ylabel("error")
    plt.title(r"Interpolation error vs. polynomial degree, $g(x) = e^{3x} \sin(2x)$")
    plt.legend()
    plt.show()


    # tightness numerical approximation for func f
    def pi_max(nodes, x):
        """max_x |prod_i (x - x_i)|"""
        return np.max(np.abs(np.prod(np.expand_dims(x, axis=1) - np.expand_dims(nodes, axis=0), axis=1)))

    x_eval = np.linspace(a_f, b_f, 100*n_max)

    for cheb, name in [(False, "Equidistant"), (True, "Chebyshev")]:
        print(f"\n{name} nodes, f(x) = cos(2*pi*x)")
        print(f"{'n':>3} {'error':>10} {'max|PI|':>10} {'est|PI|':>10} {'est/max':>8} {'bound':>10} {'bound/err':>10}")

        for n in [4, 6, 8, 10, 12, 16]:
            nodes = chebyshev_nodes(a_f, b_f, n+1) if cheb else np.linspace(a_f, b_f, n+1)

            error_est = max_norm_error(f, *lagrange_error(f, n, a_f, b_f, 100*n_max, chebyshev=cheb))
            deriv_term = (2*np.pi)**(n+1) / factorial(n+1)[0] # M_{n+1}/(n+1)!
            PI = pi_max(nodes, x_eval) # measured product
            PI_est = 2*((b_f-a_f)/4)**(n+1) if cheb else factorial(n)[0]*((b_f-a_f)/n)**(n+1)/4 # analytical product
            bound = deriv_term*PI_est

            print(f"{n:>3} {error_est:10.2e} {PI:10.2e} {PI_est:10.2e} {PI_est/PI:8.2f} {bound:10.2e} {bound/error_est:10.1f}")