import autograd.numpy as np
import matplotlib.pyplot as plt


def phi_sq(r2, eps):
    """phi as a function of r^2 = (x - x_i)^2"""
    return np.exp(-eps**2 * r2)


def interpolation_matrix(x, nodes, eps):
    """A[i, j] = phi(|x_i - nodes_j|), slik at f_thilde(x) = A @ w."""
    d = x[:, None] - nodes[None, :]
    return phi_sq(d**2, eps)


def f_thilde(w, nodes, eps, x):
    return interpolation_matrix(np.atleast_1d(x), nodes, eps) @ w


def M(nodes, eps):
    return interpolation_matrix(nodes, nodes, eps)


def RBF_interpolate(x_vals, y_vals, x_eval, eps=1):
    M_ = M(x_vals, eps)
    w = np.linalg.solve(M_, y_vals)

    return f_thilde(w, x_vals, eps, x_eval)



if __name__ == "__main__":
    a = -5; b = 5; n = 10

    def f(x):
        return 1 / (1 + x**2)

    nodes = np.linspace(a, b, n+1)
    y_vals = f(nodes)
    x_eval = np.linspace(a, b, 100*n)

    plt.plot(x_eval, f(x_eval), label="Runge's function", color="black")

    for eps in [0.1, 0.5, 1, 2, 5]:

        f_thilde_ = RBF_interpolate(nodes, y_vals, x_eval, eps)
        plt.plot(x_eval, f_thilde_, label=rf"$\epsilon$ = {eps}")


    plt.title("RBF interpolation")
    plt.legend()
    plt.show()


    eps_vals = np.logspace(-4, 1, num = 100)

    M_ = np.array([M(nodes, eps) for eps in eps_vals])
    M_cond = np.linalg.cond(M_)

    plt.semilogy(eps_vals, M_cond)
    plt.show()




    