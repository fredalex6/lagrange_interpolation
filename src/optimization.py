import autograd.numpy as np
from autograd import value_and_grad
import matplotlib.pyplot as plt

from RBF import RBF_interpolate
from error import two_norm_error
from interpolation import chebyshev_nodes

# Runge's function
def f(x):
    return 1 / (1 + x**2)

# task parameters
a = -5; b = 5; N = 1000
x_eval = np.linspace(a, b, N)

def loss_fn(interior_nodes, theta):
    eps = np.exp(theta)

    nodes = np.concatenate([[a], interior_nodes, [b]])
    f_thilde = RBF_interpolate(nodes, f(nodes), x_eval, eps)

    return two_norm_error(f, x_eval, f_thilde)

loss_and_grad = value_and_grad(loss_fn, (0, 1))

def descent_step(x_vals, theta, learning_rate):
    loss, (grad_x, grad_theta) = loss_and_grad(x_vals, theta)

    new_x_vals = x_vals - learning_rate * grad_x
    new_theta = theta - learning_rate * grad_theta

    return new_x_vals, new_theta, loss, (grad_x, grad_theta)

# merges an array with a scalar
def pack(v, t):
    return np.concatenate([v, np.atleast_1d(t)])


def optimal_interior_nodes(f, a, b, n, eps0):
    L = 100; rho = 0.5; rho_bar = 1.5; TOL = 1e-10
    theta_prev = 0; theta_next = np.log(eps0)
    max_epochs = 2000

    interior_prev = np.zeros(n)
    interior_next = np.linspace(a, b, n+1)[1:-1]

    loss_history = []
    epoch = 0

    while (np.linalg.norm(interior_next - interior_prev) > TOL or np.abs(theta_next - theta_prev) > TOL) and epoch < max_epochs:
        interior_prev = interior_next.copy()
        theta_prev = theta_next

        loss, (grad_x, grad_theta) = loss_and_grad(interior_prev, theta_prev)
        grad = pack(grad_x, grad_theta)
        x = pack(interior_prev, theta_prev)

        if np.isnan(grad).any():
            print("Gradient has atleast one NaN-element")
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

        interior_next = x_thilde[:-1]
        theta_next = x_thilde[-1]
        L *= rho

        # print(f"L = {L}")
        loss_history.append(loss)
        epoch += 1

    return loss_history



if __name__ == "__main__":
    a = -5; b = 5; n = 10; eps0 = 2;

    loss_history = optimal_interior_nodes(f, a, b, n, eps0) 
    epochs = np.arange(len(loss_history))

    fig, ax = plt.subplots(1, 1)
    ax.loglog(epochs, loss_history, label=f"n = {n}")

    ax.set_title(r"Loss history for gradient descent on $[x,\epsilon]$ for Runge's function, $x\in [-5,5]$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss in approximate 2 norm")

    ax.legend()
    plt.show()


    n_vals = np.arange(10, 50, 10)

    optimal_nodes_error = []
    error_equidistant = []
    error_chebyshev = []
    
    for n in n_vals:
        # 2 norm error for optimal nodes
        optimal_nodes_error.append(optimal_interior_nodes(f, a, b, n, eps0)[-1])

        e_nodes = np.linspace(a, b, n+1)
        c_nodes = chebyshev_nodes(a, b, n+1)

        f_thilde_equidistant = RBF_interpolate(e_nodes, f(e_nodes), x_eval)
        f_thilde_chebyshev = RBF_interpolate(c_nodes, f(c_nodes), x_eval)

        error_equidistant.append(two_norm_error(f, x_eval, f_thilde_equidistant))
        error_chebyshev.append(two_norm_error(f, x_eval, f_thilde_chebyshev))


    fig, ax = plt.subplots(1, 1)

    ax.semilogy(n_vals, optimal_nodes_error, label="Optimal nodes")
    ax.semilogy(n_vals, error_equidistant, label="Equidistant nodes")
    ax.semilogy(n_vals, error_chebyshev, label="Chebyshev nodes")

    ax.set_title("2 norm error RBF interpolation")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$||f - \tilde{f}||_{2}$")
    ax.legend()

    plt.show()