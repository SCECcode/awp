from .. import index
from .. import GridFunction
def test_init():
    idx = index.IndexTarget('x', (0, 0))
    idx = index.IndexTarget('x', (0, 0, 0))
    idx = index.IndexTarget('x', [0]*3)

def test_add():
    idx = index.IndexTarget('x', (1, 2))
    idx.add(1) == (2, 0)
    idx = index.IndexTarget('y', (1, 2))
    idx.add(1) == (0, 3)
    u = GridFunction('u', shape=(10, 10))
    u[idx.add(1)] + u[idx.add(-1)]

def test_mul():
    idx = index.IndexTarget('x', (1, 2))
    idx.mul(1) == (1, 0)
    idx = index.IndexTarget('y', (1, 2))
    idx.mul(1) == (0, 2)

def test_set():
    idx = index.IndexTarget('x', (1, 2))
    idx.set(2) == (2, 0)
    idx = index.IndexTarget('y', (1, 2))
    idx.set(3) == (0, 3)

def test_get_component_id():
    assert index.get_component_id(0) == 0
    assert index.get_component_id('x') == 0
    assert index.get_component_id('y') == 1
    assert index.get_component_id('z') == 2
    assert index.get_component_id('t') == 3
    assert index.get_component_id('u') == 4
    assert index.get_component_id('v') == 5
    assert index.get_component_id('w') == 6
    assert index.get_component_id('xx', default_components={'xx' : 0}) == 0
