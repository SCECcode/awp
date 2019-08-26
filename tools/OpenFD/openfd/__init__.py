from .base import *
from .base import kernel, utils
from .sbp import staggered as sbp_staggered, traditional as sbp_traditional
from .sbp import *
from .numerics import *
from .dev.memory import Memory
from .dev.macro import Macro
from .dev.variable import Variable
from .dev.kernelgenerator import CGenerator, CudaGenerator, OpenclGenerator
from .dev.kernelgenerator import make_kernel, write_kernels
from .dev.array import CArray
from .dev.equations import Equation, equations
from .dev.symbols import Constant
from numpy import float32 as __float32

"""
Default configurable options
    prec : Floating-point precision to use by default for kernel generation and
          evaluation
    codegen : Code generator to use.
    debug : Debug level.
"""

prec = __float32
codegen = CGenerator
debug = 0

