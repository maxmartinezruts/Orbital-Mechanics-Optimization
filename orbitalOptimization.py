#-------------------------------#
# Author:   Max Martinez Ruts
# Creation: 2019
#-------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import numericalMethodsODE as ODE
import multibodySimulation as SIM
import pygame
import time

# Convert coordinates form cartesian to screen coordinates (used to draw in pygame screen)
def cartesian_to_screen(car_pos):
    factor = 0.021
    screen_pos = np.array([center[0]*factor+car_pos[0],center[1]*factor-car_pos[1]])/factor
    screen_pos = screen_pos.astype(int)
    return screen_pos

# Screen parameters
width = 800
height = 800
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
yellow = (255,255, 0)

fpsClock = pygame.time.Clock()

T_final = 24 * 7 * 3600 *1                                 # Duration simulation
dt =  60 *5                                       # Delta time [s]
N = T_final / dt                                            # Number of steps [-]
perigee = 1/550                                             # Perigee of desired orbit/xmoon [-]
v_center = np.array([7793,270])                             # Initial velocity and angle guess
ts = 0
v_center = np.array([ 10.49080329,   2.66576622, 299.96015532, 312.24811105])
x_moon = SIM.x_moon                                         # Distance Earth - Moon
n_body = SIM.n_body                                         # Number of bodies

# Plot initial orbit
# u_0 = np.array(SIM.init_multibody(v_center))
# u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final, N, ODE.rk4)  # Results

def min_distance(v_0):
    u_0 = np.array(SIM.init_multibody())
    u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final*3, None, ODE.rk4, v_0)  # Results
    p_moon = np.array([u[:, 1], u[:, 4]])                           # Positions of Moon
    print(p_moon)
    p_sat = np.array([u[:, 2], u[:, 5]])                            # Positions of Satellite
    p_eth = np.array([u[:, 0], u[:, 3]])                            # Positions of Earth
    p_dif = p_moon - p_sat                                          # Difference moon-sat (vectorial)
    dists = np.linalg.norm(p_dif, axis=0)                           # Difference moon-sat (scalar)
    min_dist = np.min(dists)                                        # Minimum distance moon-sat (within 1 week)
    print(min_dist)

    cost =abs((min_dist-300000)/100000000)
    # A peak is desired at the perigee distance, therefore a peak function (gaussian) is applied
    print(v_0,'Distance/distance moon: ', min_dist, x_moon/min_dist,'Cost:', cost)
    i = 0
    # Game loop
    screen.fill((0,0,0))
    pygame.display.flip()

    while i < p_moon.shape[1]:
        pygame.event.get()
        pygame.draw.circle(screen, (0, 255, 0), cartesian_to_screen(p_moon[:,i]/100000000), 3)
        pygame.draw.circle(screen, (255, 255, 255), cartesian_to_screen(p_sat[:,i]/100000000), 3)
        pygame.draw.circle(screen, (0, 255, 0), cartesian_to_screen(p_eth[:,i]/100000000), 3)
        i+=10
        pygame.display.flip()
        time.sleep(0.0000025)

    return np.e**((min_dist/x_moon-perigee)**2)

v_center = optimize.fmin(min_distance,v_center,maxiter=30)

T_final = 24 * 7 * 3600 *3                                  # Duration simulation
dt = 60 * 5                                                 # Delta time [s]
N = T_final / dt
# Plot initial orbit
# u_0 = np.array(SIM.init_multibody(v_center))
# u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final, N, ODE.rk4)  # Results
# v_center = np.array([120, 4.507 * 3600 * 24, 785])

print(v_center)

# Cost function that maximizes the time oribiting around moon
def max_time(v_0):
    u_0 = np.array(SIM.init_multibody())
    u = ODE.predict(u_0, SIM.f_multibody, SIM.dfdu_multibody, T_final, 8000, ODE.rk4, v_0)  # Results
    p_eth = np.array([u[:, 1], u[:, 4]])                            # Positions of Earch
    p_moon = np.array([u[:, 1], u[:, 4]])                           # Positions of Moon
    p_sat = np.array([u[:, 2], u[:, 5]])                            # Positions of Satellite
    p_dif = p_moon - p_sat                                          # Difference moon-sat (vectorial)
    dists = np.linalg.norm(p_dif, axis=0)                           # Difference moon-sat (scalar)
    kf = np.argmin(dists[:2000])                                    # Determine index where distance to moon is closest
    t_orbiting = 0
    distance = dists[kf]
    while distance < x_moon / 5:  # While orbiting moon
        kf += 1
        t_orbiting += dt  # 5 more minutes orbiting (dt)
        if kf<len(dists)-1:
            distance = dists[kf]
            print(kf, len(dists))
        else:
            print('object is still in orbit after X weeks with:', v_0)
            i = 0
            # Game loop
            screen.fill((0, 0, 0))
            while i < p_moon.shape[1]:
                pygame.event.get()
                pygame.draw.circle(screen, (255, 0, 0), cartesian_to_screen(p_moon[:, i] / 100000000), 3)
                pygame.draw.circle(screen, (255, 255, 255), cartesian_to_screen(p_sat[:, i] / 100000000), 3)
                pygame.draw.circle(screen, (0, 255, 0), cartesian_to_screen(p_eth[:, i] / 100000000), 3)
                i += 1
                pygame.display.flip()
                fpsClock.tick(500)
    cost = T_final/(t_orbiting+1)                                   # Trying to minimize cost, therefore cost should be constant/maximizing_parameter
    print(v_0, 'time orbiting:',t_orbiting, 'cost: ',T_final/(t_orbiting+1))
    i = 0
    # Game loop
    screen.fill((0, 0, 0))
    while i < p_moon.shape[1]:
        pygame.event.get()
        pygame.draw.circle(screen, (255, 0, 0), cartesian_to_screen(p_moon[:, i] / 100000000), 3)
        pygame.draw.circle(screen, (255, 255, 255), cartesian_to_screen(p_sat[:, i] / 100000000), 3)
        pygame.draw.circle(screen, (0, 255, 0), cartesian_to_screen(p_eth[:, i] / 100000000), 3)
        i += 14
        pygame.display.flip()
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
