import numpy as np

def gridspacing(n, a=-1.0, b=1.0):
    h = float(b-a)/n
    return h

def grid(n, hat=False, a=-1, b=1):
    h = gridspacing(n, a, b)
    if hat:
        x = np.array([a] + [a + (i + 0.5)*h for i in range(n)] + [b])
    else:
        x = np.linspace(a, b, n+1)
    return x

def derivative(n, h, order, hat = False):
    from . import staggered as sbp
    from scipy.sparse import diags
    from scipy import sparse

    p = order
    d = sbp.readderivative(order, hat) 

    D = d['op_left']
    Dint = d['op_interior']
    Dint_idx = d['idx_interior']

    if hat:
        m = n + 2
    else:
        m = n + 1

    if D.shape[0]*2 > m:
        raise ValueError('Not enough grid points')

    diagonals = []
    hi = 1.0/h
    if hat:
        for i in range(len(Dint_idx)):
            diagonals.append([hi*Dint[i]]*(n+2-np.abs(Dint_idx[i])))
        Dmat = diags(diagonals, Dint_idx, shape=(n+2, n+1)).tolil()
    else:
        for i in range(len(Dint_idx)):
            diagonals.append([hi*Dint[i]]*(n+1-np.abs(Dint_idx[i])))

        # FIXME There is a bug here somewhere.
        # The operators do not pass the accuracy test.
        Dmat = sparse.lil_matrix((n+1, n+2))
        for i in range(n+1):
                for j in range(len(Dint_idx)):
                    k = i+Dint_idx[j]
                    if k < n + 2:
                        Dmat[i,k] = hi*Dint[j]

        #old code:
        #Dmat = diags(diagonals, Dint_idx, shape=(n+1, n+2)).tolil()
    Dmat[:D.shape[0],:D.shape[1]] = D*hi
    Dmat[-1:-D.shape[0]-1:-1,-1:-D.shape[1]-1:-1] = -D*hi
    print Dmat
    Dmat = Dmat.tocsr()
    return Dmat

def quadrature(n, h, order, hat = False, invert = True):
    """ 
    SBP staggered grid quadrature rule defined as a sparse matrix. 
    
    # Input arguments:
    * n: Number of grid points. 
    * h: Grid spacing. 
    * order: Order of accuracy.

    # Optional arguments:
    * hat: Specifies the type of grid to use. Staggered grid (True), Default grid (False)
    * invert: Determines if the quadrature rule should be inverted or not. 

    # Examples:
    >>> quadrature(8, 1.0, 2).diagonal()
    array([ 2.56057441,  0.86473379,  1.0492768 ,  1.        ,  1.        ,  1.        ,  1.0492768 ,  0.86473379,  2.56057441])

    # See also:
    * @grid
    * @gridspacing
    """
    from . import staggered as sbp
    from scipy.sparse import csr_matrix, lil_matrix

    p = order

    d = sbp.readquadrature(order, hat, invert=True)
    Hi = d['op_left']
    Hleft_idx = d['idx_left']
    Hright_idx = d['idx_right']
    
    if hat:
        m = n + 2
    else:
        m = n + 1
    
    if Hi.shape[0]*2 > m:
        raise ValueError('Not enough grid points')

    if invert:
        g = 1.0/h
    else:
        g = h

    A = lil_matrix( (m, m))
    for i in range(Hi.shape[0]):
        if invert:
            c = Hi[i]*g
        else:
            c = g/Hi[i]
        A[i, i]       = c
        A[-1-i, -1-i] = c
    for i in range(Hi.shape[0], m - Hi.shape[0]):
        A[i, i]       = g

    return A.tocsr()

def restrict(n, bnd, hat = False):
    from scipy.sparse import csr_matrix, lil_matrix

    if hat:
        m = n + 2
    else:
        m = n + 1

    if bnd == 0:
        k = 0
    else:
        k = m-1
    
    x = lil_matrix( (m, 1))
    x[k] = 1.0
    e = csr_matrix(x)
    return e
