import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
from scipy import optimize

l   = 1.            # (m) Length of pendulum
g   = 9.81          # (m/s^2)
m   = 0.001         # (kg) Mass of the bob
rho = 1.225         # (kg/m^3) Density of air
r   = 0.05          # (m) Radius of sphere
A   = np.pi * r**2  # (m^2) Cross-secional area
c_d = 0.47          # (.) Drag coefficient of a sphere

alpha = c_d * rho * A / 2.   # Initially assume air-resitance is zero


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


n_body = 3
M_multibody = 4 * n_body
masses = [5.972e24,  # Mass of earth [kg]
          7.34767309e22,  # Mass of moon [kg]
          70.0]  # ~ mass of you [kg]
G = 6.67384e-11  # Universal graviational constant


def f_multibody(u):
    x = u[0:n_body]
    y = u[n_body:2 * n_body]
    xdot = u[2 * n_body:3 * n_body]
    ydot = u[3 * n_body:]
    f = np.zeros((n_body, 4))
    f[:, 0] = xdot
    f[:, 1] = ydot
    for i in range(n_body):
        for j in range(n_body):
            if i != j:
                r = np.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
                f[i, 2] += G * masses[j] * (x[j] - x[i]) / r ** 3
                f[i, 3] += G * masses[j] * (y[j] - y[i]) / r ** 3
    return f.T.flatten()


def dfdu_multibody(u):
    x = u[0:n_body]
    y = u[n_body:2 * n_body]
    xdot = u[2 * n_body:3 * n_body]
    ydot = u[3 * n_body:]
    jac = np.zeros((M_multibody, M_multibody))

    jac[0:2 * n_body, 2 * n_body:] = np.eye(2 * n_body)
    for i in range(n_body):
        for j in range(n_body):
            if i != j:
                r = np.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
                dFdxj = G * masses[j] * (np.eye(2) / r ** 3 -
                                         3. / r ** 5 * np.array([[(x[j] - x[i]) ** 2, (x[j] - x[i]) * (y[j] - y[i])],
                                                                 [(x[j] - x[i]) * (y[j] - y[i]), (y[j] - y[i]) ** 2]]))
                dFdxi = G * masses[j] * (-np.eye(2) / r ** 3 -
                                         3. / r ** 5 * np.array([[(x[j] - x[i]) ** 2, (x[j] - x[i]) * (y[j] - y[i])],
                                                                 [(x[j] - x[i]) * (y[j] - y[i]), (y[j] - y[i]) ** 2]]))
                jac[2 * n_body + i, j] += dFdxj[0, 0]
                jac[2 * n_body + i, j] += dFdxj[0, 1]
                jac[2 * n_body + i, n_body + j] += dFdxj[0, 1]
                jac[3 * n_body + i, j] += dFdxj[1, 0]
                jac[3 * n_body + i, n_body + j] += dFdxj[1, 1]
    return jac


x_moon = 384400.0e3  # [m]

# Calculate velocity of earth/moon resulting in a circular orbit
# about the Earth-moon CoM.
r_moon = masses[0] * x_moon / (masses[0] + masses[1])
r_earth = masses[1] * x_moon / (masses[0] + masses[1])
T = np.sqrt(4. * np.pi ** 2 * x_moon ** 3 / (G * (masses[0] + masses[1])))  # Orbital period
vel_moon = 2. * np.pi * r_moon / T
vel_earth = -2. * np.pi * r_earth / T



def init_multibody(v):
    """Initial conditions for Earth-Moon system... and you!"""
    theta = v[1] * np.pi / 180
    #        Earth      Moon          You
    return [0, x_moon, x_moon / 20 * np.cos(theta),  # x-position (Earth at origin)
            0, 0, x_moon / 20 * np.sin(theta),  # y-position
            0, 0, v[0] * -np.sin(theta),  # x-vel (center-of-mass stationary)
            vel_earth, vel_moon, v[0] * np.cos(theta)]  # y-velocity


T_final = 24 * 7 * 3600 *1                                  # Duration simulation
dt = 60 * 5                                                 # Delta time [s]
N = T_final / dt                                            # Number of steps [-]
perigee = 1/550                                             # Perigee of desired orbit/xmoon [-]
v_center = np.array([6270,-100])                            # Initial velocity and angle guess


# Plot initial orbit
u_0 = np.array(init_multibody(v_center))
u = predict(u_0, f_multibody, dfdu_multibody, T_final, N, rk4)  # Results


plt.ion()
plt.figure()
def min_distance(v_0):
    u_0 = np.array(init_multibody(v_0))
    u = predict(u_0, f_multibody, dfdu_multibody, T_final, N, rk4)  # Results
    p_moon = np.array([u[:, 1], u[:, 4]])                           # Positions of moon
    p_you = np.array([u[:, 2], u[:, 5]])                            # Positions of you
    p_dif = p_moon - p_you                                          # Difference moon-you (vectorial)
    dists = np.linalg.norm(p_dif, axis=0)                           # Difference moon-you (scalar)
    min_dist = np.min(dists)                                        # Minimum distance moon-you (within 1 week)
    cost =np.e**((min_dist/x_moon-1/600)**2)                        # A peak is desired at the perigee distance, therefore a peak function (gaussian) is applied
    print(v_0,'Distance/distance moon: ',x_moon/min_dist,'Cost:', cost)
    plt.clf()
    plt.plot(u[:, 0], u[:, 0 + n_body], '-or', label='Earth')
    plt.plot(u[:, 1], u[:, 1 + n_body], '-k', label='Moon')
    plt.plot(u[:, 2], u[:, 2 + n_body], '-g', label='You')
    plt.title('Initial conditions')
    plt.xlim(-2.1 * x_moon, 2.1 * x_moon)
    plt.ylim(-2.1 * x_moon, 2.1 * x_moon)
    plt.legend()
    plt.pause(0.01)
    return np.e**((min_dist/x_moon-perigee)**2)

v_center = optimize.fmin(min_distance,v_center,maxiter=30)

T_final = 24 * 7 * 3600 *3                                  # Duration simulation
dt = 60 * 5                                                 # Delta time [s]
N = T_final / dt
# Plot initial orbit
u_0 = np.array(init_multibody(v_center))
u = predict(u_0, f_multibody, dfdu_multibody, T_final, N, rk4)  # Results

print(v_center)

# Cost function that maximizes the time oribiting around moon
def max_time(v_0):
    u_0 = np.array(init_multibody(v_0))
    u = predict(u_0, f_multibody, dfdu_multibody, T_final, N, rk4)  # Results
    p_moon = np.array([u[:, 1], u[:, 4]])                           # Positions of moon
    p_you = np.array([u[:, 2], u[:, 5]])                            # Positions of you
    p_dif = p_moon - p_you                                          # Difference moon-you (vectorial)
    dists = np.linalg.norm(p_dif, axis=0)                           # Difference moon-you (scalar)
    kf = np.argmin(dists[:2000])                                    # Determine index where distance to moon is closest
    t_orbiting = 0
    distance = dists[kf]
    while distance < x_moon / 5:  # While orbiting moon
        kf += 1
        t_orbiting += dt  # 5 more minutes orbiting (dt)
        if kf<len(dists)-1:
            distance = dists[kf]
        else:print('object is still in orbit after X weeks with:', v_0)
    cost = T_final/(t_orbiting+1)                                   # Trying to minimize cost, therefore cost should be constant/maximizing_parameter
    print(v_0, 'time orbiting:',t_orbiting, 'cost: ',T_final/(t_orbiting+1))
    plt.clf()
    plt.plot(u[:, 0], u[:, 0 + n_body], '-or', label='Earth')
    plt.plot(u[:, 1], u[:, 1 + n_body], '-k', label='Moon')
    plt.plot(u[:, 2], u[:, 2 + n_body], '-g', label='You')
    plt.title('Initial conditions')
    plt.xlim(-2.1 * x_moon, 2.1 * x_moon)
    plt.ylim(-2.1 * x_moon, 2.1 * x_moon)
    plt.legend()
    plt.pause(0.01)
    return cost
v_center = optimize.fmin(max_time,v_center, maxiter=100)

# Plot initial orbit
u_0 = np.array(init_multibody(v_center))
u = predict(u_0, f_multibody, dfdu_multibody, T_final, N, rk4)  # Results
plt.figure(figsize=(10,10))
plt.plot(u[:,0], u[:,0+n_body], '-or', label='Earth')
plt.plot(u[:,1], u[:,1+n_body], '-k', label='Moon')
plt.plot(u[:,2], u[:,2+n_body], '-g', label='You')
plt.title('Orbit after 2nd optimization problem')
plt.xlim(-2.1*x_moon, 2.1*x_moon)
plt.ylim(-2.1*x_moon, 2.1*x_moon)
plt.legend()
plt.show()
print(v_center)
