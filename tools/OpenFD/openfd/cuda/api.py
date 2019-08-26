"""
    OpenFD CUDA API

"""
import openfd
from openfd import CudaGenerator, write_kernels


openfd.generator = CudaGenerator


def init(variables, shape, precision=None):
    """ 
    Initialize variables to zero on the host and device.

    Arguments:
        variables : String of variables to initialize, e.g., 'p v'.
        shape : Array shape.
        precision(optional) : Precision. Pass `np.float32` for single precision
            (default) or `np.float64` for double precision.

    Returns:
        cpu : A dict containing the host variables
        gpu : A dict containing the device variables

    """
    import numpy
    import pycuda.driver as cuda
    from openfd import Struct
    import openfd

    if not precision:
        precision = openfd.prec

    zero = lambda : numpy.zeros(shape).astype(precision)

    cpu = {'%s'% var : zero() for var in variables.split(' ')}
    gpu = {'%s' % key : cuda.mem_alloc(cpu[key].nbytes) for key in cpu}
    for key in gpu:
        cuda.memcpy_htod(gpu[key], cpu[key])

    return Struct(**cpu), Struct(**gpu)

def append(cpu, gpu, variables, shape, precision=None):
    """
    Append variables to an already existing struct. New variables are
    initialized to zero on both the host and device.

    Arguments:
        cpu : A dict containing the host variables
        gpu : A dict containing the device variables
        variables : String of variables to initialize, e.g., 'p v'.
        shape : Array shape.
        precision(optional) : Precision. Pass `np.float32` for single precision
            (default) or `np.float64` for double precision.

    """
    import warnings
    import numpy
    import pycuda.driver as cuda
    from openfd import Struct
    import openfd

    if not precision:
        precision = openfd.prec

    zero = lambda : numpy.zeros(shape).astype(precision)

    new_variables = variables.split(' ')
    for var in new_variables:
        if not var in cpu:
            cpu[var] = zero()
        else:
            warnings.warn('Variable `%s` already exists in host struct.\
                         Skipping.')
            continue
        if not var in gpu:
            gpu[var] = cuda.mem_alloc(cpu[var].nbytes)
            cuda.memcpy_htod(gpu[var], cpu[var])
        else:
            warnings.warn('Variable `%s` already exists in device struct.\
                         Skipping.')

def init_h(variables, shape, precision=None):
    from openfd.dev import kernelevaluator as ke
    return ke.init_h(variables, shape, precision) 

def init_d(variables, shape, precision=None):
    """ 
    Initialize variables to zero on the device.

    Arguments:
        variables : String of variables to initialize, e.g., 'p v'
        shape : Array shape.
        precision(optional) : Precision. Pass `np.float32` for single precision
            (default) or `np.float64` for double precision.

    Returns:
        A dict of initialized device variables.

    """

    import pycuda.driver as cuda
    import numpy
    from openfd import Struct
    import openfd

    if not precision:
        precision = openfd.prec

    zero = lambda : numpy.zeros(shape).astype(precision)

    cpu = {'%s'% var : zero() for var in variables.split(' ')}

    gpu = {'%s' % key : cuda.mem_alloc(cpu[key].nbytes) for key in cpu}
    for key in gpu:
        cuda.memcpy_htod(gpu[key], cpu[key])

    return Struct(**gpu)

def make_kernel(label, 
           dout, 
           din, 
           bounds, 
           gridsize, 
           const=[], 
           extrain=[], 
           extraconst=[], 
           indices=[], 
           regions=None,
           debug=False, 
           precision=None,
           loop_order=None, 
           reduction=None,
           lhs_indices=None,
           rhs_indices=None):
    """
    Wrapper for `kernelgenerator.make_kernel`.

    """
    from openfd import make_kernel

    return make_kernel(label, 
                       dout, 
                       din, 
                       bounds, 
                       gridsize, 
                       const=const,
                       extrain=extrain, 
                       extraconst=extraconst, 
                       indices=indices, 
                       regions=regions,
                       generator=CudaGenerator,
                       debug=debug, 
                       precision=precision,
                       loop_order=loop_order, 
                       reduction=reduction, 
                       lhs_indices=lhs_indices,
                       rhs_indices=rhs_indices
                       )

def load_kernels(kernel, functions, dict=False):
    """
    Load CUDA compute kernel functions from file.
    
    Arguments:
        kernel : Name of kernel to load excluding file extension.
        functions : String of functions to load. Each function is separated by a
            space.
        dict : Return a `dict` of loaded kernel functions if `True`. Defaults to
            `False`.

    Returns:
        A list of the loaded kernel functions.

    """
    from pycuda.compiler import SourceModule
    from openfd.dev.kernelgenerator import extension
    from openfd.dev.kernelevaluator import find_kernels

    kernelname = kernel + extension('Cuda')['source']
    kernelfile = open(kernelname, 'r')
    source = kernelfile.read()
    kernelfile.close()
    mod = SourceModule(source)

    if dict:
        kernels = {'%s' % name : mod.get_function(name) for name in
                   find_kernels(functions, source)}
    else:
        kernels = [mod.get_function(name) for name in 
                   find_kernels(functions, source)]
    
    return kernels

def copy_htod(dest, source, dest_vars='', source_vars=''):
    """
    Copy variables stored in a `dict` from the host to the device.

    Arguments:
        dest(`dict`) : Destination reference  (GPU)
        source(`dict`) : Source reference (CPU)
        dest_vars(Optional) : String of variables to copy to the GPU. 
            Each variable is separated by a space. If not specified, all
            variables are copied.
        source_vars(Optional) : String of variables to copy from the CPU. 
            Each field is separated by a space. If not specified, all variables
            are copied.

    """
    import pycuda.driver as cuda
    from openfd.dev.kernelevaluator import get_dest_source_keys

    dest_keys, source_keys = get_dest_source_keys(dest, source, 
                                              dest_vars, source_vars)

    for dest_key, source_key in zip(dest_keys, source_keys):
        cuda.memcpy_htod(dest[dest_key], source[source_key])

def copy_dtoh(dest, source, dest_vars='', source_vars=''):
    """
    Copy variables from the device to the host.

    Arguments:
        dest(`dict`) : Destination reference (CPU)
        source(`dict`) : Source reference (GPU)
        dest_vars(Optional) : String of variables to copy to the CPU. 
            Each variable is separated by a space. If not specified, all
            variables are copied.
        source_vars(Optional) : String of variables to copy from the GPU. 
            Each field is separated by a space. If not specified, all variables
            are copied.

    """
    import pycuda.driver as cuda
    from openfd.dev.kernelevaluator import get_dest_source_keys

    dest_keys, source_keys = get_dest_source_keys(dest, source, 
                                                  dest_vars, source_vars)

    for dest_key, source_key in zip(dest_keys, source_keys):
        cuda.memcpy_dtoh(dest[dest_key], source[source_key])

