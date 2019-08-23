from openfd import GridFunction, Bounds
from sympy import symbols
from openfd import CudaGenerator, make_kernel, write_kernels
import numpy as np

def write_cuda_kernel(name='test', functions=['test'], ranges=[range(3)]):
    n = symbols('n')
    shape = (n,)
    dout = GridFunction('out', shape)
    din = GridFunction('in', shape)
    
    kl = []
    for func, reg in zip(functions, ranges):
        kl += make_kernel(func, dout, din, Bounds(n), shape, 
                    generator=CudaGenerator, regions=reg)
    write_kernels(name, kl)

