import numpy as np
import fs2
import openfd
from openfd.cuda import make_kernel, write_kernels, load_kernels, init, \
                        copy_htod, copy_dtoh

prec = np.float64

openfd.prec = prec

def new(nx, ny):

    cpu, gpu = init('s12 s22', shape=(nx, ny))

    # apply linear function that is zero at the free surface 
    # nbnd is the index of the grid point on the free surface
    nbnd = 4
    y = prec(np.arange(start=0, stop=ny, step=1))
    Y = np.reshape(np.repeat(y, nx), (nx, ny)).T
    cpu.ans_s12 = prec(Y - nbnd)
    cpu.s12 = prec(Y - nbnd)
    cpu.ans_s22 = prec(Y - nbnd - 0.5)
    cpu.s22 = prec(Y - nbnd - 0.5)

    copy_htod(gpu, cpu, 's12 s22')
    return cpu, gpu


def test_image():
    """

    Check that the anti-symmetric function (y - yb) is unmodified if yb is the
    coordinate on the boundary.

    """
    kernels = fs2.stress()
    fs2.write(kernels)
    nx = np.int32(8)
    ny = np.int32(8)

    image, = load_kernels('kernels/fs2', 'fs2_stress_*')
    cpu, gpu = new(nx, ny)

    image(gpu.s12, gpu.s22, nx, ny, block=(32, 32, 1), grid=(1, 1))
    copy_dtoh(cpu, gpu, 's12 s22')

    assert np.all(np.isclose(cpu.s12, cpu.ans_s12))
    assert np.all(np.isclose(cpu.s22, cpu.ans_s22))

def test_velocities():

    kernels = fs2.stress()
    kernels += fs2.velocity()
    fs2.write(kernels)
    nx = np.int32(8)
    ny = np.int32(8)

    image, = load_kernels('kernels/fs2', 'fs2_velocity_*')
    cpu, gpu = new(nx, ny)
    assert 0
