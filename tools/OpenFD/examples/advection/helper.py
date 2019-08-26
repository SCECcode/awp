from openfd.dev import kernelgenerator as kg
import numpy as np

def build(kg, name, regions=None):
    """
    Generates a compute kernel for each region with the name `name_region` where
    `region` is replaced by `l`, `r`, or `i` depending on if the computation
    takes place on the left, right boundaries, or in the interior of the
    computational domain.

    Arguments:
        kg : A kernel object. Call `kernelgenerator` to construct one.
        name : the name of the kernel function.
        regions(optional) : A tuple that contains `-1`, `0`, or `1` depending on
            which regions compute kernels should be generated for.

    """
    names = {-1: 'r', 0: 'l', 1: 'i'}
    kernelname = lambda x: '%s_%s' % (name, x)
    kernels = []
    if not regions:
        regions = [-1, 0, 1]
    regx = regions
    regy = regions
    for rx in regx:
            kernel = kg.kernel(kernelname(names[rx]), (rx,))
            kernels.append(kernel)
    return kernels


import numpy as np

# vtk output file
def output_vtk(filename, label, nx, array):
    file = open(filename,"w")
    file.write("# vtk DataFile Version 2.0\n%s\n" %filename)
    file.write("ASCII\n")
    file.write("DATASET STRUCTURED_POINTS\n")
    file.write("DIMENSIONS %i %i 1\n" % (nx, 1))
    file.write("ASPECT_RATIO 1 1 1\n")
    file.write("ORIGIN 0 0 0\n")
    file.write("POINT_DATA %i\n" % (nx))
    file.write("SCALARS %s float 1\n" % label)
    file.write("LOOKUP_TABLE default\n")
    for i in range(nx):
        if i % 3 == 0:
            file.write("\n")
        file.write("%f " %array[i])
    file.close()

class RK4:
    # Runge-Kutta coefficients for time stepping (low-storage)
    a = np.array([0,
                  -567301805773.0/1357537059087,
                  -2404267990393.0/2016746695238,
                  -3550918686646.0/2091501179385,
                  -1275806237668.0/842570457699])
    b = np.array([1432997174477.0/9575080441755,
                  5161836677717.0/13612068292357,
                  1720146321549.0/2090206949498,
                  3134564353537.0/4481467310338,
                  2277821191437.0/14882151754819])
    c = np.array([0,
                  1432997174477.0/9575080441755,
                  2526269341429.0/6820363962896,
                  2006345519317.0/3224310063776,
                  2802321613138.0/2924317926251])
