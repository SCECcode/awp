import numpy

def operator(op, expr, axis, order=2, hat=False, gpu=True, shape=None, 
        ops='cidsg'):
    """
    Construct a staggered grid operator.

    Arguments:
        op : Operator to construct. `'D'` for derivative. `'H'` for norm
        expr : Expression to apply the derivative to
        order(optional) : Order of accuracy
        hat(optional) : Set to `True` if the derivatives are computed on the
            hat-grid (cell-centered grid).
        gpu : Generate code for the GPU. Defaults to `True`.

    """
    from openfd import sbp_traditional as sbp
    #FIXME: Update this statement once proper support for the uniform grid
    # staggered operators have been implemented
    hatstr = ['', 'hat']
    fmt = '%s/%s%s' % (ops, op, hatstr[hat]) + '%s.json'
    coef = '%s%s' % (op.lower(), hatstr[hat])
    return sbp.Derivative(expr, axis, order=order, fmt=fmt,
                          gpu=gpu, coef=coef, shape=shape)

def norm_coef(order=2, hat=False, ops='cidsg'):
    """
    Return the norm coefficient needed in the penalty terms. The coefficient
    returned is `1/H[0,0]`.
        
    Arguments:
        hat(optional) : Set to `True` if the derivatives are computed on the
            hat-grid (cell-centered grid).
        order(optional) : Order of accuracy

    """
    H = operator('H', '', 'x', order=order, hat=hat, ops=ops)
    h00 = H.coef(0).data[0,0]
    return 1.0/h00

def grids(nv, precision=None):
    """
    Construct the `p` grid and `v` grid given `nv` grid points in the `v` grid
    (regular grid).

    The grids are constructed to be of the same size. An additional zero point
    is included at the end of the `v` grid.

    Arguments:
        nv : Number of grid points in the regular grid
        prec : Floating-point precision to use. Defaults to `np.float64`.

    Returns:
        xp, xv, h
        xp : Cell-centered grid
        xv : Regular grid
        h : Grid spacing
        hi : Reciprocal grid spacing
    """
    import openfd

    if not precision:
        prec = openfd.prec
    else:
        prec = precision
    
    h = prec(1/(nv-1))
    hi = prec(nv-1)

    xv = numpy.array([prec(j*h) for j in range(nv)] + [prec(0)])
    xp = prec(xv - 0.5*h)
    xp[0] = 0.0
    xp[-1]= 1.0

    return xp, xv, h, hi

def init_block2d(nx, ny, threads=32):
    """
    Initialize block size and grid size for 2d computation.

    Arguments:
        npts : Number of grid points (nx*ny)

    """
    t = threads
    block = (t, t, 1)

    gx = int((nx+t-1)/t)
    gy = int((ny+t-1)/t)

    c = 1
    grids = ((c, c), (c, gy), (c, c), 
             (gx, c), (gx, gy), (gx, c), 
             (c, c), (c, gy), (c, c))

    return block, grids

def init_block_sponge(nx, ny, threads=32, width=64):
    """
    Initialize block and grid size for the sponge layer.

    Arguments:
    nx, ny : Grid dimensions
    threads : Number of threads in each block direction
    width : Width of the sponge layer. Must match the value used in the kernel.

    """

    t = threads
    gx = int((nx+t-1)/t)
    gy = int((ny+t-1)/t)
    cx = int((width+t-1)/t)
    cy = int((width+t-1)/t)

    block = (t, t, 1)

    grids = ((cx, cy), (cx, gy), (cx, cy), 
             (gx, cy), (gx, gy), (gx, cy), 
             (cx, cy), (cx, gy), (cx, cy))

    return block, grids


class Writer(object):

    def __init__(self, filename, writer, extension='.vtk'):
        self.frame = 0
        self.filename = filename
        self.writer = writer
        self.extension = extension

    def write(self, *args):
        fh = open('%s_%d%s' %(self.filename, self.frame, self.extension), 'w')
        self.writer(fh, *args)
        fh.close()
        self.frame += 1

def output_vtk(filename, label, x, y, fields):
    """
    Write vtk output file.

    Arguments:
        filename : Name of file to write to including `'.vtk'` extension. 
            If the file already exists, it will be overwritten, and otherwise
            created.
        label : Label to give the dataset
        grid : Coordinates to write to vtk file.

    """
    assert x.shape == y.shape
    nx = x.shape[0]
    ny = x.shape[1]
    fh = open(filename,"w")
    fh.write("# vtk DataFile Version 4.2\n%s\n" %filename)
    fh.write("ASCII\n")
    fh.write("DATASET STRUCTURED_GRID\n")
    fh.write("DIMENSIONS %i %i 1\n" % (nx, ny))
    fh.write("POINTS %i float\n" % (nx*ny))
    for j in range(ny):
        for i in range(nx):
            fh.write("%f %f %f \n" %(x[i, j], y[i, j], 1))

    numpts = nx*ny
    fh.write("POINT_DATA %ld \n\n"% numpts)
    
    labels = label.split(" ")
    for label, field in zip(labels, fields):
        fh.write("FIELD scalar 1\n")
        fh.write("%s 1 %ld float\n"%( label, numpts))
        for j in range(ny):
            for i in range(nx):
                fh.write("%f \n" %field[i, j])
    fh.close()

def vtk_2d(fh, label, x, y, fields):
    assert x.shape == y.shape
    nx = x.shape[0]
    ny = x.shape[1]
    fh.write("# vtk DataFile Version 4.2\n\n")
    fh.write("ASCII\n")
    fh.write("DATASET STRUCTURED_GRID\n")
    fh.write("DIMENSIONS %i %i 1\n" % (nx, ny))
    fh.write("POINTS %i float\n" % (nx*ny))
    for j in range(ny):
        for i in range(nx):
            fh.write("%f %f %f \n" %(x[i, j], y[i, j], 1))

    numpts = nx*ny
    fh.write("POINT_DATA %ld \n\n"% numpts)
    
    labels = label.split(" ")
    for label, field in zip(labels, fields):
        fh.write("FIELD scalar 1\n")
        fh.write("%s 1 %ld float\n"%( label, numpts))
        for j in range(ny):
            for i in range(nx):
                fh.write("%f \n" %field[i, j])

def shifts():
    """
     Flags that indicate if the grid is shifted in the particular direction or
     not
     Example v = (1, 0) 
     implies that v = v(x_i+1/2, y_j)
    """
    from openfd import Struct
    G = Struct()
    G.v1 = (0, 1)
    G.v2 = (1, 0)
    G.snn = (1, 1)
    G.s12 = (0, 0)
    return G
