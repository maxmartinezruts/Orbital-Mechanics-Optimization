#-------------------------------#
# Author:   Max Martinez Ruts
# Creation: 2019
#-------------------------------#

import numpy as np
import matplotlib.pyplot as plt

def fe(ui, f, dfdu, dt):
    """
    Single step of forward-Euler.

    Args:
        ui (np.array): State at time-step i
        f (function): Fn f defining the ODE
        dfdu (function): Fn returning the derivative of f (not used in FE, just
                         for consistency with other methods)
        dt (float): Time-step
    Return:
        u (np.array): State at timestep i+1
    """

    return ui + dt * f(ui)
    ### TODO: implement forward-Euler


def predict(u_0, f, dfdu, T_final, N, step_fn):
    """
    Solve the ODE defined by f and dfdu in time, from initial condition `u_0`
    to final time `T_final` with `N` time-steps, using `step_fn()` to obtain
    $u_{i+1}$ from $u_i$ at each timestep.  Return array shaped (N+1, M)
    containing the solution u at each time, including t=0.
    """
    M = u_0.shape[0]
    dt = T_final / N
    u = np.zeros((int(N) + 1, M))
    u[0, :] = u_0
    for i in range(int(N)):
        u[i + 1, :] = step_fn(u[i, :], f, dfdu, dt)

    return u


def plot(u, T_final):
    """Plot the output of `predict()`."""
    N = u.shape[0] - 1

    plt.plot(np.linspace(0, T_final, N + 1), u[:, 0] * 180. / np.pi, '-+')
    plt.xlabel('t')
    plt.ylabel('deg')


def plot_stability_region(stabfn):
    """
    Given a stability fn for a particular scheme, plot the stability region.
    Green is unstable, white is stable.

    Args:
      stabfn (function): Takes one complex argument, returns complex value.
    """
    x = np.linspace(-4, 4, 100)
    X = np.meshgrid(x, x)
    z = X[0] + 1j * X[1]
    Rlevel = np.abs(stabfn(z))
    plt.figure(figsize=(8, 8))
    plt.contourf(x, x, Rlevel, [1, 1000])
    plt.contour(x, x, Rlevel, [1, 1000])
    plt.xlabel(r'Re');
    plt.ylabel(r'Im')
    plt.plot([0, 0], [-4, 4], '-k')
    plt.plot([-4, 4], [0, 0], '-k')
    plt.axes().set_aspect('equal')


def plot_eigvals(u, dfdu, dt):
    """
    Plot the eigenvalues of $df/du \cdot \Delta t$, for all values of the solution u.

    Args:
      u (array (N, M)): Solution array (real-valued) with N timesteps, M values per step.
      dt (float): Timestep.
    Return:
      eigs (array (N, M)): Eigenvalue array (complex-valued) for each solution time.
    """
    N, M = u.shape
    eigs = np.zeros((N, M), dtype=np.complex)
    for n in range(N):
        eigs[n, :] = np.linalg.eigvals(dt * dfdu(u[n]))
    # TODO: Compute and plot eigenvalues
    return eigs


def fe_stability(z): return 1 + z


def newton(f, dfdx, x_0, iter_max=10, min_error=1e-14):
    """
    Newton method for rootfinding.  It guarantees quadratic convergence
    given f'(root) != 0 and abs(f'(xi)) < 1 over the domain explored.

    Args:
        f (function): function
        dfdx (function): derivative of f
        x_0 (array M): starting guess
        iter_max (int): max number of iterations
        min_error (float): min allowed error

    Returns:
        x[-1] (float) = root of f
        x (np.array) = history of convergence
    """
    x = [x_0]
    for i in range(1,iter_max):
        xp1 = x[-1] - np.linalg.inv(dfdx(x[-1])).dot(f(x[-1]))
        x.append(xp1)
        if np.max(np.abs(f(x[-1]))) < min_error:
            break
    return x[-1], np.array(x)


def be(ui, f, dfdu, dt):
    """
    Single step of backward-Euler with Newton solution method.

    Args:
        ui (np.array): State at time-step i
        f (function): Fn f defining the ODE
        dfdu (function): Fn returning the derivative of f (not used in FE, just
                         for consistency with other methods)
        dt (float): Time-step
    Return:
        u (np.array): State at timestep i+1
    """

    # TODO: Implement backward-Euler using `newton()`
    def F(x):
        return x - ui - dt * f(x)

    def dF(x):
        jaccobian = np.eye(len(x)) - dt * dfdu(x)

        return jaccobian

    return newton(F, dF, ui)[0]


def be_stability(z):
    return 1 / (z - 1)


def rk4(u, f, dfdu, dt):
    """
    Runge-Kutta explicit 4-stage scheme - single step.

    Args:
        z (np.array): State at time-step i
        f (function): Fn f defining the ODE
        dt (float): Time-step
    Return:
        z (np.array): State at timestep i+1
    """
    # TODO: Implement RK4
    k1 = dt * f(u)
    k2 = dt * f(u + k1 / 2)
    k3 = dt * f(u + k2 / 2)
    k4 = dt * f(u + k3)
    ui_1 = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return ui_1


def rk4_stability(z):
    return 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24

