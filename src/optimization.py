import autograd.numpy as np
from autograd import value_and_grad
import matplotlib.pyplot as plt

from RBF import RBF_interpolate


def f(x):
    return 1 / (1 + x**2)

a = -5; b = 5; N = 1000
x_eval = np.linspace(a, b, N)
y_vals = f(x_eval)

def loss_fn(interior_nodes, theta):
    eps = np.exp(theta)

    nodes = np.concatenate([[a], interior_nodes, [b]])
    f_thilde = RBF_interpolate(nodes, f(nodes), x_eval, eps)

    return (b - a)/N * np.sum((y_vals - f_thilde)**2)

loss_and_grad = value_and_grad(loss_fn, (0, 1))

def descent_step(x_vals, theta, learning_rate):
    loss, (grad_x, grad_theta) = loss_and_grad(x_vals, theta)

    new_x_vals = x_vals - learning_rate * grad_x
    new_theta = theta - learning_rate * grad_theta

    return new_x_vals, new_theta, loss, (grad_x, grad_theta)


def pack(v, t):
    return np.concatenate([v, np.atleast_1d(t)])


if __name__ == "__main__":

    n = 20; L = 100; rho = 0.5; rho_bar = 1.5; TOL = 1e-10
    eps0 = 2; theta_prev = 0; theta_next = np.log(eps0)
    max_epochs = 2000

    interior_prev = np.zeros(n)
    interior_next = np.linspace(a, b, n + 2)[1:-1]

    loss_history = []
    epoch = 0

    while (np.linalg.norm(interior_next - interior_prev) > TOL or np.abs(theta_next - theta_prev) > TOL) and epoch < max_epochs:
        interior_prev = interior_next.copy()
        theta_prev = theta_next

        loss, (grad_x, grad_theta) = loss_and_grad(interior_prev, theta_prev)
        grad = pack(grad_x, grad_theta)
        x = pack(interior_prev, theta_prev)

        max_inner_its = 100
        inner_it = 0

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

        print(f"L = {L}")
        loss_history.append(loss)
        epoch += 1


    plt.plot(np.arange(epoch), loss_history, label="Loss function")

    plt.legend()
    plt.show()


