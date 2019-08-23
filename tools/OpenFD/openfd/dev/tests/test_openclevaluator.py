from . helper import kernel1d, kernel2d
from  .. openclevaluator import OpenclEvaluator
from  .. kernelgenerator import OpenclGenerator

def test_opencl():
    kernel1d(OpenclGenerator, OpenclEvaluator)
    kernel2d(OpenclGenerator, OpenclEvaluator)
