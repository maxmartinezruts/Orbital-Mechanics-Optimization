import numpy as np

n_body = 3
M_multibody = 4 * n_body
masses = [5.972e24,         # Mass of earth [kg]
          7.34767309e22,    # Mass of moon [kg]
          70.0]             # ~ mass of you [kg]
G = 6.67384e-11             # Universal graviational constant

x_moon = 384400.0e3  # [m]

# Calculate velocity of earth/moon resulting in a circular orbit
# about the Earth-moon CoM.
r_moon = masses[0] * x_moon / (masses[0] + masses[1])
r_earth = masses[1] * x_moon / (masses[0] + masses[1])
T = np.sqrt(4. * np.pi ** 2 * x_moon ** 3 / (G * (masses[0] + masses[1])))  # Orbital period
vel_moon = 2. * np.pi * r_moon / T
vel_earth = -2. * np.pi * r_earth / T



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


def init_multibody(v):
    """Initial conditions for Earth-Moon system... and you!"""
    theta = v[1] * np.pi / 180
    #        Earth      Moon          You
    return [0, x_moon, x_moon / 20 * np.cos(theta),  # x-position (Earth at origin)
            0, 0, x_moon / 20 * np.sin(theta),  # y-position
            0, 0, v[0] * -np.sin(theta),  # x-vel (center-of-mass stationary)
            vel_earth, vel_moon, v[0] * np.cos(theta)]  # y-velocity