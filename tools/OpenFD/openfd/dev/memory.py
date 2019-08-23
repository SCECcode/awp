class Memory(object):
    """
    Memory class. 

    Represents the physical memory allocation of a grid. To account for memory
    alignment, or for other reasons, the allocated memory may be larger than the
    dimensions of the numerical grid. The access of the memory layout may also
    have a different permutation ordering of the slow and fast indices. For
    example, `(i, j, k)` where `k` is the fast index (consecutive entries `k+1`,
    `k+1` are contiguous in memory) may be reversed to `(k, j,
    i)`.

    Provides functions for accessing the data according to 
    * C
    * Fortran (#TODO)

    It is the user's responsibility to allocate memory that are consistent with
    the size defined by the `Memory` object.


         Memory (mx x my)

         mx >= align[0] + nx 
         my >= align[1] + ny 
         -------------------------------------------------------
         |           |                             | ^         |
         |<--------->|                             | | align[1]|
         | align[0]  |                             | |         |
         |           |                             | |         |
         |-----------|-----------------------------|-----------|
         |           |                             |           |
         |           |                             |           |
         |           |                             |           |
         |           |       Numerical grid        |           |
         |           |       (nx x ny)             |           |
         |           |                             |           |
         |           |                             |           |
         |           |                             |           |
         |-----------|-----------------------------|-----------|
         |           |                             |           |
         |           |                             |           |
         |           |                             |           |
         -----------------------------------------------------

    Attributes:
        shape(tuple): Size of memory allocation in each grid direction.
        align(tuple, optional) : Alignment in terms of an offset for each grid
            dimension. Defaults to `None`.
        perm(tuple, optional) : A permuted set of indices. Defaults to `None`,
            which is the same as `{0, 1, 2}`. Use `{2, 1, 0}` to reverse order.

    """

    def __init__(self, shape, align=None, perm=None):
        default_align = [0]*len(shape)
        default_perm = range(len(shape))

        if not align:
            align = default_align

        if not perm:
            perm = default_perm

        assert(len(shape) == len(align))
        assert(len(shape) == len(perm))
        assert(max(perm) == len(shape)-1)
        assert(min(perm) == 0)
        
        sorted_perm = sorted(perm)
        for def_pi, sorted_pi in zip(default_perm, sorted_perm):
            assert def_pi == sorted_pi

        self.shape = shape
        self.align = align
        self.perm = perm

    def get_c(self, *indices):
        """
        Returns a symbolic expression that converts a tuple of indices into a
        single, one-dimensional index. The returned index is consistent with the
        memory layout specification defined by this objects attributes.

        Arguments:
            indices : Tuple of indices to convert into a one-dimensional
                representation.

        Raises:
            IndexError: Raised if length of `indices` exceeds the dimensionality
                of the memory object (its shape).
        
        """
        from sympy import sympify
        from sympy.core.compatibility import is_sequence

        if not is_sequence(indices):
            indices = (indices,)

        #if len(indices) > len(self.shape):
        #    raise IndexError("Too many indices. Expected len(indices) < %d." 
        #                     " Got len(indices) == %d." % (len(indices), 
        #                     len(self.shape)))

        offset = {}
        offset[0] = 1

        shape = [self.shape[idx] for idx in self.perm]
        indices = [indices[idx] for idx in self.perm]
        for i in range(1,len(shape)):
            offset[i] = offset[i-1]*shape[i-1]
        indices = [(a + idx)*offset[o] for idx, a, o in  zip(indices, self.align, offset)]
        index   = sympify(sum(indices))

        return index

