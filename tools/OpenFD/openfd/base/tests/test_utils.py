import pytest
from .. import utils

def test_to_tuple():
    assert utils.to_tuple(1) == (1,)
    assert utils.to_tuple((1,)) == (1,)
    assert utils.to_tuple('test') == ('test',)

def test_to_atom():
    assert utils.to_atom((1,)) == 1
    assert utils.to_atom((1, 2)) == (1, 2)

def test_is_seq():
    assert not utils.is_seq(1)
    assert utils.is_seq((1,))
    assert not utils.is_seq('test')
