# using autograd here, so the functions can be used in 1e)
import autograd.numpy as np 
import matplotlib.pyplot as plt

from error import max_norm_error


def phi_square(r2, eps):
    """phi as a function of r^2 = (x - x_i)^2"""
    return np.exp(-eps**2 * r2)


def interpolation_matrix(x, nodes, eps):
    """M[i, j] = phi(|x_i - nodes_j|), such that f_thilde(x) approximates M @ w."""
    d = np.expand_dims(x, axis=1) - np.expand_dims(nodes, axis=0)
    return phi_square(d**2, eps)


def f_thilde(w, nodes, eps, x):
    """f_thilde(x) = M @ w"""
    return interpolation_matrix(np.atleast_1d(x), nodes, eps) @ w


def M(nodes, eps):
    """calculates the interpolation matrix M[i, j] = phi(|nodes_i - nodes_j|)"""
    return interpolation_matrix(nodes, nodes, eps)


def RBF_interpolate(nodes, y_vals, x_eval, eps=1.39):
    """interpolate on nodes using a RBF with given epsilon, given y_vals, and evalue on x_eval"""
    M_ = M(nodes, eps)
    w = np.linalg.solve(M_, y_vals)

    return f_thilde(w, nodes, eps, x_eval)

def cond_and_error(func, a, b, n, eps_vals):
    """cond(M) and max norm error for RBF interpolation of func, for every eps."""
    nodes = np.linspace(a, b, n+1)
    y_vals = func(nodes)
    x_eval = np.linspace(a, b, 100*(n+1))

    conds, errors = [], []
    for eps in eps_vals:
        M_eps = M(nodes, eps)
        conds.append(np.linalg.cond(M_eps))
        try:
            w = np.linalg.solve(M_eps, y_vals)
            errors.append(max_norm_error(func, x_eval, f_thilde(w, nodes, eps, x_eval)))
        except np.linalg.LinAlgError:
            errors.append(np.nan) # 0 as one of the pivots
    return np.array(conds), np.array(errors)


def plot_rbf(func, a, b, n, eps_list, title):
    """plot func and its RBF interpolant for every eps in eps_list"""
    nodes = np.linspace(a, b, n+1)
    y_vals = func(nodes)
    x_eval = np.linspace(a, b, 100*(n+1))

    fig, ax = plt.subplots()

    ax.plot(x_eval, func(x_eval), label="f", color="black")
    ax.plot(nodes, y_vals, "o", color="black")
    for eps in eps_list:
        ax.plot(x_eval, RBF_interpolate(nodes, y_vals, x_eval, eps), label=rf"$\epsilon$ = {eps}")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    a = -5; b = 5; n = 10

    def f(x):
        return 1 / (1 + x**2)

    eps_vals_f = [0.1, 0.5, 1.3, 5]
    plot_rbf(f, a, b, n, eps_vals_f, "RBF interpolation, Runge's function")

    # test function parameters
    a = 0; b = 1; n = 10

    # one of the test functions
    def g(x):
        return np.cos(2*np.pi*x)

    eps_vals_g = [0.3, 1.4, 20]
    plot_rbf(g, a, b, n, eps_vals_g, r"RBF interpolation, $\cos(2\pi x)$")

    eps_vals = np.logspace(-3, 1, num = 100)
    M_cond, error = cond_and_error(g, a, b, n, eps_vals)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.loglog(eps_vals, M_cond)
    ax1.set_title("Condition number of M")
    ax1.set_ylabel(r"cond($M$)")

    ax2.loglog(eps_vals, error)
    ax2.set_title(r"Max norm error RBF approximation")
    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel(r"$||f - \tilde{f}||_{\infty}$")

    plt.show()