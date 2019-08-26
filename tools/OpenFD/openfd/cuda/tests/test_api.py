import os
import numpy as np
from openfd.cuda import init, init_h, init_d, copy_dtoh, copy_htod,\
                        load_kernels, append
from . conftest import write_cuda_kernel
import pycuda.autoinit
import pytest

def test_init():
    shape = (10, 10)
    cpu, gpu = init('in out', shape=shape, precision=np.float32)
    assert cpu['in'].shape == shape

def test_append():
    shape = (10, 10)
    cpu, gpu = init('in out', shape=shape, precision=np.float32)
    new_shape = (12, 12)
    append(cpu, gpu, 'new', shape=new_shape, precision=np.int32)
    assert cpu['new'].shape == new_shape
    assert cpu['new'].dtype == np.int32
    with pytest.warns(UserWarning) : append(cpu, gpu, 'new', 
                                             shape=new_shape, 
                                             precision=np.int32)


def test_init_h():

    shape = (10, 10)
    cpu = init_h('in', shape=shape)
    assert cpu['in'].shape == shape

    cpu = init_h('p v', shape=shape)
    assert 'p' in cpu

def test_init_d():

    shape = (10, 10)
    gpu = init_d('in', shape=shape)
    assert 'in' in gpu

def test_load_cuda():
    write_cuda_kernel()
    kernels = load_kernels('test', 'test_0', dict=True)
    assert 'test_0' in kernels

    kernels = load_kernels('test', 'test_0')
    assert kernels[0]
    kernels = load_kernels('test', 'test_*')
    assert kernels[2]
    os.remove('test.cu')

def test_copy():
    shape = (10, )
    cpu, gpu = init('in out', shape=shape)
    cpu['in'] += 1.0
    # Copy `in` from cpu to 'in' on gpu 
    # Source string can be left out when the dest and source variable is the
    # same
    copy_htod(gpu, cpu, 'in')
    # Copy `in` from gpu to 'out' on cpu
    copy_dtoh(cpu, gpu, 'out', 'in')
    assert np.isclose(cpu['in'][0], cpu['out'][0])
    
