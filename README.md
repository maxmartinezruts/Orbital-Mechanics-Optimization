# Orbital-Mechanics-Optimization
Simulation of a 3 body motion using ODE and determination of parameters needed to achieve a lunar insertion orbit from Earth.
Animation: https://www.youtube.com/watch?v=Vs_O9MWWWiY

Optimization process to find a numerical solution for a lunar insertion orbit in a 3 body simulation (Earth, Moon, satellite). The optimization process is separated into two separate subproblems. 

The 1st part of the problem consists of searching an orbit with a specific perigee (the closest point on orbit to the moon), which happens to be the closest point on the earth-moon orbit to the moon as well. This is achieved by defining a cost function, by establishing the minimum distance to the moon within a period of one week and then redefining the cost as a peak function (gaussian) that peaks on the perigee specified.

The advantage of using this method is that if the orbit collides with the moon, the starting initial conditions of the 2nd optimization problem will be located on a domain where the cost function is very sensitive to slight variations in the initial conditions and therefore will converge to a local minimum. However, if the 1st problem leads to an orbit that passes the perigee, the second problem starts on a domain where the function is much smoother and it will be easier to find the global maximum.

The 2nd optimization problem consists of varying the initial conditions such that the last time the satellite leaves the moon orbit is maximized. This is also achieved using simplex optimization
