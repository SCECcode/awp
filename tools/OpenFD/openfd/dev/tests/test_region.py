import pytest
from .. import region 

def test_autoconvert():
    assert isinstance(region.autoconvert(0), region.Coordinate)
    assert isinstance(region.autoconvert(0, 1), region.Index)
    assert isinstance(region.autoconvert('left'), region.Label)
    with pytest.raises(ValueError) : region.autoconvert('unknown')
    with pytest.raises(ValueError) : region.autoconvert(-2)
    with pytest.raises(ValueError) : region.autoconvert(3)

def test_iscoordfmt():
    fmt = 'c'
    assert region.iscoordfmt(fmt)
    fmt = 'py'
    assert region.iscoordfmt(fmt)
    fmt = 'unknown'
    assert not region.iscoordfmt(fmt)

def test_iscoord():
    fmt = 'c'
    assert region.iscoord(0, fmt)
    assert region.iscoord(1, fmt)
    assert region.iscoord(2, fmt)
    assert region.iscoord((0, 0), fmt)
    assert region.iscoord((1, 0), fmt)
    assert region.iscoord((2, 1), fmt)
    assert region.iscoord((0, 1, 0), fmt)
    assert region.iscoord((1, 0, 1), fmt)
    assert region.iscoord((2, 1, 0), fmt)
    assert not region.iscoord(3, fmt)
    assert not region.iscoord(-1, fmt)
    fmt = 'py'
    assert region.iscoord(-1, fmt)
    assert not region.iscoord(2, fmt)
    assert not region.iscoord((0,0,0,0), fmt)

def test_isindex():
    assert region.isindex(0, dim=1)
    assert region.isindex(3, dim=2)
    assert region.isindex(26, dim=3)
    assert not region.isindex(3, dim=1)
    assert not region.isindex(-1, dim=1)
    assert not region.isindex('left', dim=1)

def test_islabel():
    assert region.islabel('left')
    assert region.islabel('left bottom')
    assert region.islabel('left bottom center')
    assert not region.islabel('unknown')


def test_isdim():
    assert region.isdim(1)
    assert region.isdim(2)
    assert region.isdim(3)
    assert not region.isdim(4)
    assert not region.isdim(0)

def test_coordtoindex():
    fmt = 'c'
    onebased = region._onebased
    assert region.coordtoindex(0, fmt)         == 0 + onebased
    assert region.coordtoindex(1, fmt)         == 1 + onebased
    assert region.coordtoindex(2, fmt)         == 2 + onebased
    assert region.coordtoindex((0, 0), fmt)    == 0 + onebased
    assert region.coordtoindex((1, 0), fmt)    == 1 + onebased
    assert region.coordtoindex((2, 0), fmt)    == 2 + onebased
    assert region.coordtoindex((0, 1), fmt)    == 3 + onebased  
    assert region.coordtoindex((1, 1), fmt)    == 4 + onebased
    assert region.coordtoindex((2, 1), fmt)    == 5 + onebased
    assert region.coordtoindex((2, 2), fmt)    == 8 + onebased
    assert region.coordtoindex((0, 0, 0), fmt) == 0 + onebased
    assert region.coordtoindex((2, 2, 2), fmt) == 26 + onebased
    fmt = 'py'
    assert region.coordtoindex(0, fmt)            == 0 + onebased
    assert region.coordtoindex(1, fmt)            == 1 + onebased
    assert region.coordtoindex(-1, fmt)           == 2 + onebased
    assert region.coordtoindex((-1, -1), fmt)     == 8 + onebased
    assert region.coordtoindex((-1, -1, -1), fmt) == 26 + onebased
    with pytest.raises(IndexError) : region.coordtoindex((0,0,0,0))

def test_coordtolabel():
    fmt = 'c'
    assert region.coordtolabel(0, fmt) == 'left'
    assert region.coordtolabel(1, fmt) == 'center'
    assert region.coordtolabel(2, fmt) == 'right'
    assert region.coordtolabel((0, 0), fmt) == 'left bottom'
    assert region.coordtolabel((1, 0), fmt) == 'center bottom'
    assert region.coordtolabel((2, 0), fmt) == 'right bottom'
    with pytest.raises(IndexError) : region.coordtolabel((0,0,0,0))

def test_coordtodeffmt():
    fmt = 'c'
    region._defcoordfmt = fmt
    assert region.coordtodeffmt(0) == (0,)
    assert region.coordtodeffmt(1) == (1,)
    assert region.coordtodeffmt(-1) == (2,)
    assert region.coordtodeffmt((0, 0)) == (0, 0)
    assert region.coordtodeffmt((1, 1)) == (1, 1)
    assert region.coordtodeffmt((-1, -1)) == (2, 2)
    fmt = 'py'
    region._defcoordfmt = fmt
    assert region.coordtodeffmt(0) == (0,)
    assert region.coordtodeffmt(1) == (1,)
    assert region.coordtodeffmt(2) == (-1,)
    with pytest.raises(IndexError) : region.coordtodeffmt((0,0,0,0))

def test_coordtoc():
    assert region.coordtocfmt(0) == (0,)
    assert region.coordtocfmt(1) == (1,)
    assert region.coordtocfmt(2) == (2,)
    assert region.coordtocfmt(-1) == (2,)

def test_indextocoord():
    onebased = region._onebased
    assert region.indextocoord(0  + onebased, dim=1) == (0,)
    assert region.indextocoord(1  + onebased, dim=1) == (1,)
    assert region.indextocoord(2  + onebased, dim=1) == (2,)
    assert region.indextocoord(4  + onebased, dim=2) == (1, 1)
    assert region.indextocoord(13 + onebased, dim=3) == (1, 1, 1)
    with pytest.raises(IndexError) : region.indextocoord(14, dim=4)

def test_indextolabel():
    onebased = region._onebased
    assert region.indextolabel(0  + onebased, dim=1) == 'left'
    assert region.indextolabel(1  + onebased, dim=1) == 'center'
    assert region.indextolabel(2  + onebased, dim=1) == 'right'
    assert region.indextolabel(4  + onebased, dim=2) == 'center center'
    assert region.indextolabel(13 + onebased, dim=3) == 'center center center'
    with pytest.raises(IndexError) : region.indextolabel(13 + onebased, dim=4)

def test_labeltoindex():
    onebased = region._onebased
    assert region.labeltoindex('left')        == onebased + 0
    assert region.labeltoindex('center')      == onebased + 1
    assert region.labeltoindex('right')       == onebased + 2
    assert region.labeltoindex('left bottom') == onebased + 0
    assert region.labeltoindex('left center') == onebased + 3
    assert region.labeltoindex('right top')   == onebased + 8
    with pytest.raises(ValueError) : region.labeltoindex('unknown')

def test_labeltocoord():
    fmt = 'c'
    region._defcoordfmt = fmt
    assert region.labeltocoord('left', fmt) == (0, )
    assert region.labeltocoord('center', fmt) == (1, )
    assert region.labeltocoord('right', fmt) == (2, )
    assert region.labeltocoord('left bottom', fmt) == (0, 0)
    assert region.labeltocoord('left center', fmt) == (0, 1)
    assert region.labeltocoord('right top', fmt) == (2, 2)
    with pytest.raises(ValueError) : region.labeltocoord('unknown')

def test_coordinate():
    C = region.Coordinate(0)
    # Invalid coordinate format
    with pytest.raises(ValueError) : region.Coordinate((0, 1, 2), fmt='unknown') 

    assert region.Coordinate(0).label().label == 'left'
    assert region.Coordinate(0).index().index ==  0 + region._onebased
    assert region.Coordinate(-1, fmt='py').toc().coord ==  (2,)
    assert region.Coordinate(2).topy().coord ==  (-1,)

def test_index():
    assert region.Index(0, dim=1).coord().coord == (0,)
    assert region.Index(1, dim=1).coord().coord == (1,)
    assert region.Index(2, dim=1).coord(fmt='c').coord == (2,)
    assert region.Index(2, dim=2).coord(fmt='c').coord == (2, 0)
    assert region.Index(2, dim=2).label().label == 'right bottom'

    # Invalid dimension
    with pytest.raises(IndexError) : region.Index(3, dim=4)
    # Invalid index
    with pytest.raises(IndexError) : region.Index(27, dim=3)

def test_label():
    onebased = region._onebased
    assert region.Label('left').coord(fmt='c').coord == (0,)
    assert region.Label('center').coord(fmt='c').coord == (1,)
    assert region.Label('right').coord(fmt='c').coord == (2,)
    assert region.Label('right bottom').coord(fmt='c').coord == (2, 0)
    assert region.Label('right bottom').index().index == 2 + onebased
    assert region.Label('right top').index().index == 8 + onebased

    #Invalid label
    with pytest.raises(ValueError) : region.Label('unknown')

def test_labels():
    assert len(region._labels1d) == 3
    assert len(region._labels2d) == 9
    assert len(region._labels3d) == 27

