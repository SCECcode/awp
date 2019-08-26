from . helper import kernel1d, kernel2d
from  .. cudaevaluator import CudaEvaluator
from  .. kernelgenerator import CudaGenerator

def test_cuda():
    kernel1d(CudaGenerator, CudaEvaluator)
    kernel2d(CudaGenerator, CudaEvaluator)
