from . helper import write_cuda_kernel
from .. import kernelevaluator as ke
import os 
import pytest

@pytest.fixture(scope="module", params=[1])
def cases(request):
    if request.param == 0:
        # query, answer, functions, regions
        yield ('test_*', ['test_0', 'test_1'], ['test'], [range(3)])
    if request.param == 1:
        yield ('test', ['test'], ['test', 'test'], [range(3), None])

def test_fetch_kernels():
    write_cuda_kernel('test')
    source = open('test.cu', 'r').read()
    kernels = ke.fetch_kernels(source)
    assert 'test_0' in kernels[0]
    os.remove('test.cu')

def test_find_kernels(cases):
    query = cases[0]
    answer = cases[1]
    write_cuda_kernel('test', cases[2], cases[3])
    source = open('test.cu', 'r').read()
    os.remove('test.cu')

    result = ke.find_kernels(query, source)
    for kernel in answer:
        assert kernel in result

    with pytest.raises(ValueError) : ke.find_kernels('does not exist', source)

