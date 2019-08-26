import numpy as np
from .. import types

def test_C():
    assert types.C._int == types.C.get_type(np.int32) 
    assert types.C._uint == types.C.get_type(np.uint32) 
    assert types.C._float == types.C.get_type(np.float32) 
    assert types.C._double == types.C.get_type(np.float64) 
    assert '*' in types.C.get_ptr(np.int32) 

