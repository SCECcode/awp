import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import helper
import numpy as np

kernel = 'advection.cu'
# Load GPU compute kernels
kernelfile = open(kernel, 'r')
mod = SourceModule(kernelfile.read())

pde_l = mod.get_function("pde_l")
pde_i = mod.get_function("pde_i")
pde_r = mod.get_function("pde_r")
bc_l = mod.get_function("bc_l")
update_i = mod.get_function("update_i")

# 1D advection solver
def adv_1d():
    from params_adv import xmin, xmax, tmin, tmax, dx, cfl, omega, c, st
    from helper import RK4
    c = np.float32(c)
    dt = np.float32(cfl*dx/c)
    # set grid for x and t
    x = np.arange(xmin,xmax+dx,dx)
    t = np.arange(tmin,tmax+dt,dt)
    nx = len(x)
    nt = len(t)
    omega = np.float32(omega)
    k = omega/c

    # initial condition for wave height
    u = np.sin(k*x)
    u = u.astype(np.float32)
    u_gpu = cuda.mem_alloc(u.nbytes)
    cuda.memcpy_htod(u_gpu, u)
    du = np.zeros(nx)
    du = du.astype(np.float32)
    du_gpu = cuda.mem_alloc(du.nbytes)

    hi = np.float32(1.0/dx)
    n = np.int32(nx)

    # Number of threads in a block and number of blocks in a grid
    block = (32, 1, 1)
    grid = (int(n/block[0] + 1), 1)

    print("Number of time steps: %d"% nt)
    print("Number of grid points: %d"% n)
    print("dx = %g, dt = %g" %(dx, dt))

    frame = 0
    for step in range(0, nt+1):
        t = np.float32(step*dt)
        for k in range(len(RK4.a)):
            ak = np.float32(RK4.a[k])
            bk = np.float32(RK4.b[k])
            ck = np.float32(RK4.c[k])

            # Compute RK4 rates for PDE
            pde_l(du_gpu, u_gpu, c, ak, hi, n, block=block)
            pde_i(du_gpu, u_gpu, c, ak, hi, n, block=block, grid=grid)
            pde_r(du_gpu, u_gpu, c, ak, hi, n, block=block)
            # Compute RK4 rates for bc
            bc_l(du_gpu, u_gpu, c, ck, dt, hi, omega, t, n, block=block)
            # Compute RK4 update
            update_i(u_gpu, du_gpu, bk, dt, n, block=block, grid=grid)

        if step % st == 0:
            print("step = %d, t = %g"% (step, t + dt))
            cuda.memcpy_dtoh(u, u_gpu)
            helper.output_vtk("vtk/advection_%i.vtk" % frame, "velocity", nx, u)
            frame += 1

# call solver
adv_1d()
