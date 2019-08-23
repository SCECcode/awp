from openfd import Axis

def test_new_axis():
    a = Axis("x", shape=(1,1,1))
    a = Axis("y", shape=(1,1,1))
    a = Axis("z", shape=(1,1,1))

def test_id():
    a = Axis("x", shape=(1,1,1))
    assert a.id == 0
    a = Axis("y", shape=(1,1,1))
    assert a.id == 1
    a = Axis("z", shape=(1,1,1))
    assert a.id == 2
    a = Axis("z", shape=(1,), dims=(2,))
    assert a.id == 2

def test_local_id():
    a = Axis("z", shape=(1,), dims=(2,))
    assert a.local_id == 0

def test_len():
    a = Axis("x", shape=(1,))
    assert a.len == 1
    a = Axis("y", shape=(1,2))
    assert a.len == 2
    a = Axis("z", shape=(1,2,3))
    assert a.len == 3
    a = Axis("z", shape=(3,), dims=(2,))
    assert a.len == 3

def test_val():
    a = Axis("x", shape=(1,))
    assert a.val(1) == 1
    a = Axis("y", shape=(1,1))
    assert a.val((1,2)) == 2
    a = Axis("z", shape=(1,1,1))
    assert a.val((1,2,3)) == 3
    a = Axis("z", shape=(1,), dims=(2,))
    assert a.val((1,2,3)) == 3

    a = Axis("z", shape=(1,1), dims=(0,2))
    assert a.val((1,2,3)) == 3

def test_add():
    a = Axis("x", shape=(1,))
    assert a.add(1, 1) == 2
    a = Axis("x", shape=(1,1))
    assert a.add((1, 1), 1) == (2, 1)
    assert a.add((1, 1), 1, overwrite=True) == (1, 1)

