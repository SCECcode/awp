from .. import bounds, index
import pytest

def test_init():
    b = bounds.Bounds(0, 1)

def test_getitem():
    n, m = index.indices('n m')
    # Test wrap-around
    b = bounds.Bounds(0, n, 
            lower_out_of_bounds='wrap-around',
            upper_out_of_bounds='wrap-around')

    assert b[-2] == n - 2
    assert b[-1] == n - 1
    assert b[n+1] == 1
    assert b[n+2] == 2

    # Test raise exception
    b = bounds.Bounds(0, n, 
            lower_out_of_bounds='raise exception',
            upper_out_of_bounds='raise exception')
    with pytest.raises(bounds.LowerOutOfBoundsException) : b[-2]
    with pytest.raises(bounds.LowerOutOfBoundsException) : b[-1]
    with pytest.raises(bounds.UpperOutOfBoundsException) : b[n+1]
    with pytest.raises(bounds.UpperOutOfBoundsException) : b[n+2]
    with pytest.raises(IndexError) : b[m]

    # Test no action
    b = bounds.Bounds(0, n, 
            lower_out_of_bounds='no action',
            upper_out_of_bounds='no action')
    assert b[-2] == -2
    assert b[-1] == -1

def test_is_in_bounds():
    a, b = index.indices('a b')
    B = bounds.Bounds(a, b)
    assert B.is_in_bounds(a)
    assert B.is_in_bounds(b-1)
    assert not B.is_in_bounds(a-1)
    with pytest.raises(IndexError) : B.is_in_bounds(a+b)


