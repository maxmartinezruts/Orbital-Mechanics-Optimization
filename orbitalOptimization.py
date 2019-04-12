import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
from scipy import optimize
import numericalMethodsODE as ODE
import multibodySimulation as SIM

T_final = 24 * 7 * 3600 *1                                  # Duration simulation
dt = 60 * 5                                                 # Delta time [s]
N = T_final / dt                                            # Number of steps [-]
perigee = 1/550                                             # Perigee of desired orbit/xmoon [-]
v_center = np.array([6270,-100])                            # Initial velocity and angle guess
x_moon = SIM.x_moon                                         # Distance Earth - Moon
n_body = SIM.n_body                                         # Number of bodies

# Plot initial orbit
u_0 = np.array(SIM.init_multibody(v_center))
u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final, N, ODE.rk4)  # Results

plt.ion()
plt.figure()

def min_distance(v_0):
    u_0 = np.array(SIM.init_multibody(v_0))
    u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final, N, ODE.rk4)  # Results
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
u_0 = np.array(SIM.init_multibody(v_center))
u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final, N, ODE.rk4)  # Results

print(v_center)

# Cost function that maximizes the time oribiting around moon
def max_time(v_0):
    u_0 = np.array(SIM.init_multibody(v_0))
    u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final, N, ODE.rk4)  # Results
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
u_0 = np.array(SIM.init_multibody(v_center))
u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final, N, ODE.rk4)  # Results
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
