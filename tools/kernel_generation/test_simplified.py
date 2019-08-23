import pycuda.autoinit
from openfd.cuda import load_kernels, init, copy_htod, copy_dtoh
import numpy as np

prec=np.float64
vel = load_kernels('simplified', 'velocity_*')
stress = load_kernels('simplified', 'stress_*')
kinetic = load_kernels('simplified', 'kinetic_energy_rate*')
strain = load_kernels('simplified', 'strain_energy_rate*')

ngsl = 8
align = 32

nx = np.int32(32)
ny = np.int32(32)
nz = np.int32(32)

ox = 2 + ngsl
oy = 2 + ngsl
oz = align

mx = np.int32(4 + 2 * ngsl + nx)
my = np.int32(4 + 2 * ngsl + ny)
mz = np.int32(nz + 2 * align)
shape = (mx, my, mz)
cpu, gpu = init('u1 s13 du1 ds13', shape=shape, precision=prec)
block = (128, 1, 1)
grid = (int(mx),int(my), int(mz / block[0]) + 1)
grids = (grid, grid, grid)
timestep = prec(1.0)
nu = prec(0.1)

shift = 10
cpu.u1[ox::-ox,oy::-oy,oz+shift::-oz] = 1

copy_htod(gpu, cpu, 'u1 s13')
for step in range(100):
    for vk, sk, kr, sr, grid in zip(vel, stress, kinetic, strain, grids):
            # Compute update
            sk(gpu.s13, gpu.u1, timestep, nu, nx, ny, nz, np.int32(0),
                    np.int32(0), nx, ny, block=block, grid=grid)
            vk(gpu.u1, gpu.s13, timestep, nu, nx, ny, nz, np.int32(0),
                    np.int32(0), nx, ny, block=block, grid=grid)
            # Compute rates
            sk(gpu.ds13, gpu.u1, np.int32(0), nu, nx, ny, nz, np.int32(0),
                    np.int32(0), nx, ny, block=block, grid=grid)
            vk(gpu.du1, gpu.s13, np.int32(0), nu, nx, ny, nz, np.int32(0),
                    np.int32(0), nx, ny, block=block, grid=grid)
            # Compute energy rates
            kr(gpu.v1, gpu.du1, gpu.u3, np.int32(0), nu, nx, ny, nz, np.int32(0),
                    np.int32(0), nx, ny, block=block, grid=grid)

__global__ void kinetic_energy_111(double *v3, const double *du3, const double *u3, const int nx, const int ny, const int nz, const int bi, const int bj, const int ei, const int ej)

cpu2, gpu2 = init('u1 s13 du1 ds13', shape=shape, precision=prec)
copy_dtoh(cpu, gpu, 'u1 s13')
print(np.max(np.abs(cpu.u1)))
print(np.max(np.abs(cpu.s13)))
