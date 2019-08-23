from sympy import symbols
from .. rungekutta import LSRK4
import numpy as np

def test_lsrk_init():
    rk = LSRK4()
    assert rk.dt == symbols('dt')

def test_lsrk_coef():
    rk = LSRK4()
    rk.a[0]
    rk.a[rk.nstages-1]
    rk.b[0]
    rk.b[rk.nstages-1]

def test_rates():
    rk = LSRK4()
    u, du, ak = symbols('u du ak')
    lhs, rhs = rk.rates(du, u)
    assert lhs == du
    assert str(rhs) == str(ak*du + u)

def test_update():
    rk = LSRK4()
    u, du, bk, dt = symbols('u du bk dt')
    lhs, rhs = rk.update(u, du)
    assert lhs == u
    assert str(rhs) == str(dt*bk*du + u)

def test_eval():
    from openfd.dev import cudaevaluator as ce
    from openfd.dev import kernelgenerator as kg
    from openfd import Bounds
    from sympy import symbols
    from openfd import GridFunction
    import numpy as np
    from openfd.sbp import traditional as sbp
    from openfd import GridFunctionExpression as GFE

    evaluator = ce.CudaEvaluator
    generator = kg.CudaGenerator

    n = symbols('n')
    u = GridFunction('u', shape=(n,))
    du = GridFunction('du', shape=(n,))
    s = GridFunction('s', shape=(n,))

    rhs = -s*u
    
    rk = LSRK4()

    nmem = np.int32(32)
    gpu_u = np.ones((nmem,)).astype(np.float32)
    gpu_du = np.ones((nmem,)).astype(np.float32)
    gpu_s = np.ones((nmem,)).astype(np.float32)

    krate = []
    kupd = []
    kernel = generator((n,), Bounds(n), *rk.rates(du, rhs))
    krate.append(kernel.kernel('rates', 1))

    #TODO: Fix kernel evaluation so that inputs can be updated and also execute
    # multiple, different kernels
    kernel = generator((n,), Bounds(n), *rk.update(u, du))
    kupd.append(kernel.kernel('update', 1))

    dt = np.float32(0.1)
    erate = evaluator(krate + kupd, 
            inputs={n : nmem, s : gpu_s, u : gpu_u, du : gpu_du,
                   rk.ak : rk.a[1],
                   rk.bk : rk.b[1], 
                   rk.dt : dt }, 
            outputs={du : gpu_du, u : gpu_u})

    erate.eval()
    erate.get_outputs()

def test_precision():

    rk = LSRK4(prec=np.float64)
    assert isinstance(rk.a[0], np.float64)

