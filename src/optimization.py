import autograd.numpy as np
from autograd import value_and_grad
import matplotlib.pyplot as plt

from RBF import RBF_interpolate
from error import two_norm_error_square
from interpolation import chebyshev_nodes

# Runge's function
def f(x):
    return 1 / (1 + x**2)

# task parameters
a = -5; b = 5; N = 1000

def make_loss(f, a, b, N):
    x_eval = np.linspace(a, b, N+1)
    f_eval = f(x_eval)

    def loss_fn(nodes, theta):
        eps = np.exp(theta)
        f_tilde = RBF_interpolate(nodes, f(nodes), x_eval, eps)
        return (b - a) / N * np.sum((f_eval - f_tilde)**2)

    return loss_fn, x_eval

loss_fn, x_eval = make_loss(f, a, b, N)
loss_and_grad = value_and_grad(loss_fn, (0, 1))


# merges an array with a scalar
def pack(v, t):
    return np.concatenate([v, np.atleast_1d(t)])


def optimal_nodes(a, b, n, eps0, L):
    rho = 0.5; rho_bar = 1.5; TOL = 1e-7; theta = np.log(eps0); max_epochs = 2000

    nodes = np.linspace(a, b, n+1)
    loss_history = []
    epoch = 0

    while epoch < max_epochs:
        nodes_prev = nodes.copy()
        theta_prev = theta

        loss, (grad_x, grad_theta) = loss_and_grad(nodes_prev, theta_prev)
        grad = pack(grad_x, grad_theta)
        x = pack(nodes_prev, theta_prev)

        if np.isnan(grad).any():
            print("Gradient has atleast one NaN-element")
            break

        if np.linalg.norm(grad) <= TOL:
            break

        max_inner_its = 100
        inner_it = 0

        # backtracking for gradient descent
        while inner_it < max_inner_its:
            x_thilde = x - (1/L) * grad
            loss_thilde = loss_fn(x_thilde[:-1], x_thilde[-1])

            if loss_thilde <= loss + np.dot(grad, x_thilde - x) + L/2 * np.linalg.norm(x_thilde - x)**2:
                break

            L *= rho_bar
            inner_it += 1

        nodes = x_thilde[:-1]
        theta = x_thilde[-1]
        L *= rho

        # print(f"L = {L}")
        loss_history.append(loss)
        epoch += 1

    if nodes.min() < a or nodes.max() > b:
        print("One or more points are outside the interval")
        print(f"Min: {nodes.min():.4f}, Max: {nodes.max():.4f}")

    final_loss = loss_fn(nodes, theta)

    return pack(np.sort(nodes), np.exp(theta)), pack(loss_history, final_loss)


if __name__ == "__main__":
    a = -5; b = 5; n = 10; eps0 = 1.39;

    # finding optimal nodes for multiple startvalues of L
    fig, ax = plt.subplots(1, 1)

    L_vals = [1, 10, 100]

    for i in range(len(L_vals)):
        params, loss_history = optimal_nodes(a, b, n, eps0, L_vals[i]) 
        epochs = np.arange(1, len(loss_history) + 1) # first point is lost

        ax.loglog(epochs, loss_history, label=f"n = {n}, L = {L_vals[i]}")

    ax.set_title(r"Loss history for gradient descent on $[x,\epsilon]$ for Runge's function, $x\in [-5,5]$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$||f - \tilde{f}||_{2}^2$")
    ax.legend()

    plt.show()

    n_vals = np.arange(10, 51, 5)

    optimal_nodes_error = []
    error_equidistant = []
    error_chebyshev = []

    optimal_params = []

    for n in n_vals:
        # approximate 2 norm error for optimal nodes
        params, loss_history = optimal_nodes(a, b, n, eps0, L=100)

        optimal_params.append(params)
        optimal_nodes_error.append(loss_history[-1])

        e_nodes = np.linspace(a, b, n+1)
        c_nodes = chebyshev_nodes(a, b, n+1)

        eps = params[-1]

        # use same epsilon value for equidistant and Chebyshev RBF interpolation
        f_thilde_equidistant = RBF_interpolate(e_nodes, f(e_nodes), x_eval, eps)
        f_thilde_chebyshev = RBF_interpolate(c_nodes, f(c_nodes), x_eval, eps)

        error_equidistant.append(two_norm_error_square(f, x_eval, f_thilde_equidistant))
        error_chebyshev.append(two_norm_error_square(f, x_eval, f_thilde_chebyshev))

    # table with squared 2-norm
    print(f"{'n':>4} {'Optimal':>12} {'Equidistant':>12} {'Chebyshev':>12}")
    for n, e_opt, e_eq, e_ch in zip(n_vals, optimal_nodes_error, error_equidistant, error_chebyshev):
        print(f"{n:>4} {e_opt:>12.2e} {e_eq:>12.2e} {e_ch:>12.2e}")

    fig, ax = plt.subplots(1, 1)

    ax.semilogy(n_vals, optimal_nodes_error, label="Optimal nodes")
    ax.semilogy(n_vals, error_equidistant, label="Equidistant nodes")
    ax.semilogy(n_vals, error_chebyshev, label="Chebyshev nodes")

    ax.set_title("2 norm sqaured error with RBF interpolation")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$||f - \tilde{f}||_{2}^2$")
    ax.legend()

    plt.show()

    # compare with exact function for n = 10
    nodes, eps = optimal_params[0][:-1], optimal_params[0][-1]
    f_thilde_optimal = RBF_interpolate(nodes, f(nodes), x_eval, eps)

    fig, ax = plt.subplots(1, 1)

    ax.plot(x_eval, f(x_eval), label="f")
    ax.plot(x_eval, f_thilde_optimal, "--", label=r"$\tilde{f}$ with optimal nodes")
    ax.plot(nodes, f(nodes), "o", label="Optimal nodes")

    ax.set_title(rf"RBF interpolation with optimal nodes, n = {n_vals[0]}, $\epsilon$ = {eps:.3f}")
    ax.set_xlabel("x")
    ax.legend()

    plt.show()

    # pointwise error with optimal nodes for all n
    fig, ax = plt.subplots(1, 1)

    for n, params in zip(n_vals, optimal_params):
        nodes, eps = params[:-1], params[-1]
        f_thilde_optimal = RBF_interpolate(nodes, f(nodes), x_eval, eps)
        ax.semilogy(x_eval, np.abs(f(x_eval) - f_thilde_optimal), label=rf"n = {n}, $\epsilon$ = {eps:.3f}")

    ax.set_title("Pointwise error with optimal nodes")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|f - \tilde{f}|$")
    ax.set_ylim(bottom=1e-10) # the error is approx. 0 at the nodes, which squashes the plot
    ax.legend()

    plt.show()