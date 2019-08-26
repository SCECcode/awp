import numpy as np
"""
 This file contains configurable parameters.

 xmin: minimum x-coordinate value
 xmax: maximum x-coordinate value
 tmin: start time
 tmax: end time
 dx: grid spacing
 cfl: CFL number
 omega: frequency of wave (radians)
 c: wave velocity
 st: time step to output solution
"""
xmin = 0
xmax = 1
tmin = 0
tmax = 4*np.pi
dx = 0.01
cfl = 1.0
omega = 2*np.pi
c = 1
st = 10
