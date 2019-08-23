from sympy import Expr, Tuple, Matrix, sympify
from sympy.core.compatibility import NotIterable, is_sequence
from . import gridfunction as gridfunc
from . gridfunction import GridFunctionExpression, right, GridFunction
from . bounds import Bounds
from . axis import Axis
from numpy import zeros
from ..dev.array import CArray

class OperatorException(Exception):
    pass

class Operator(GridFunctionExpression):
    """

    The Operator class is used to build tensor like operators
    such as finite difference derivatives, and interpolators. Operators act on user-specified expressions and transform
    the expressions by applying stencils. Operators are symbolic objects that only perform any
    computations when they are evaluated (i.e., they use lazy evaluation). 
    
    ### Example: One dimensional use
    As an example, suppose we have an expression of the form  \\(e = a_i + b_i + 2\\) and we wish to transform this expression
    by applying a difference approximation to it. Note here that a_i and b_i are GridFunction objects.
    The grid functions support indexing via a[0], a[1], and a[j], where j is symbolic. In the
    interior of a regular Cartesian grid, the difference approximation we wish to apply is written as e_{i+1} - e{i-1}. 
    This operation can easily be carried out using operators. 

    ```python
    >>> from sympy import symbols
    >>> from openfd import GridFunction, sbp_traditional as sbp 
    >>> nx,j = symbols('nx j')
    >>> a  = GridFunction("a",shape=(nx+1,))
    >>> b  = GridFunction("b",shape=(nx+1,))
    >>> e  = a + b
    >>> e_x  = sbp.Derivative(e,"x")
    >>> e_x[j]
    -0.5*(a[j - 1] + b[j - 1]) + 0.5*(a[j + 1] + b[j + 1])
    
    ```

    ### Example: Boundary access
    The operators understand the concept of a boundary and interior. For example, a different calculation can be formed
    at the boundary than in the interior. The example above shows how to invoke the interior stencil. To invoke the
    left boundary stencil, we simply type

    ```python
    >>> e_x[0]
    -1.0*(a[0] + b[0]) + 1.0*(a[1] + b[1])

    ```

    ### Example: Two dimensional use
    Although an operator is restricted to only act in one direction, multiple directions are supported by nesting
    operators. This enables one to construct quite complex expressions. For example, we can use this technique to
    construct a mixed derivative

    ```python
    >>> nx,ny,i,j = symbols('nx ny i j')
    >>> a  = GridFunction("a", shape=(nx+1, ny+1))
    >>> a_x  = sbp.Derivative(a, "x", order=2)
    >>> a_xy = sbp.Derivative(a_x, "y", order=2)
    >>> a_xy[i,j]
    -0.5*(-0.5*a[i - 1, j - 1] + 0.5*a[i + 1, j - 1]) + 0.5*(-0.5*a[i - 1, j + 1] + 0.5*a[i + 1, j + 1])

    ```
    ### Customization

    The stencil that has been applied can be modified and used to build new operators,  or by invoking
    other pre-existing operators that derive from this class. Examples include high order difference operators and
    interpolators 
    
    Take a look at the sbp_traditional module to see more examples.
    
    """

    def __new__(cls, expr, axis, label=None, shape=None, dims=None, op_left=None, idx_left=None, op_right=None, idx_right=None,
                op_interior=None, idx_interior=None, operator_data=None, periodic=False, coef=None, gpu=False, **kw_arg):
# Attempt to infer shape from expression
        if shape is None:
            shape = gridfunc.infer_shape(expr)

        # Attempt to infer dimensions from expression
        if dims is None:
            dims = gridfunc.infer_dims(expr)

        # Use matrix when there is no expression
        # that is instead of D(u), use D*u
        if expr is None or expr == '' or expr == "":
            is_matrix = True
            expr = GridFunction('', shape = shape)
        else:
            is_matrix = False

        # Scalar object
        if shape is None:
            shape = (1,)

        if is_sequence(shape):
            shape = Tuple(*shape)
        else:
            shape = Tuple(shape)



        obj =  GridFunctionExpression.__new__(cls, expr, axis, label, **kw_arg)

        obj._cls = cls
        obj._label = label
        obj._transpose = False
        obj._shape     = shape
        obj._axis      = Axis(axis, shape, dims)
        obj._is_matrix = is_matrix
        obj._periodic  = periodic
        obj._coef      = coef
        obj._gpu       = gpu
        obj.diagonal  = False

        # enums specifying if an index maps to the boundary or the interior
        obj._idx_left     = 0
        obj._idx_right    = -1
        obj._idx_interior = 1

        if operator_data is None:
            obj._operator_data = operatordata(op_left, idx_left, op_right, idx_right, op_interior, idx_interior)
        else:
            obj._operator_data = operator_data

        # Set default coefficient name when `GPU` is enabled.
        if gpu:
            if not coef:
                coef = 'd'
            obj._coef = coef

        if coef is not None:
            op = obj._operator_data
            obj._coef = {}
            # Reverse ordering of indices for C compatibility (shape[1], shape[0])
            obj._coef['left']     = CArray(coef + 'l', op['op_left'])
            obj._coef['right']    = CArray(coef + 'r', op['op_right'])
            obj._coef['interior'] = CArray(coef,       op['op_interior'])
        else:
            obj._coef = None

        obj._bounds = Bounds(size=obj._shape[obj._axis.local_id],left=obj.num_bnd_pts_left,right=obj.num_bnd_pts_right)

        return obj


    def __getitem__(self, indices, **kw_args):
        """ 
         Get item is used to access and evaluate an operator
         at a particular index. Certain rules are setup to determine
         if the requested value of the operator is on the left or right boundary,
         or if it is in the interior.

         # Examples

        
        ```python
         >>> from sympy import symbols
         >>> from openfd import GridFunction, sbp_traditional as sbp 
         >>> nx,j = symbols('nx j')
         >>> u    = GridFunction("u", shape=(nx+1,))
         >>> u_x  = sbp.Derivative(u,"x", order=2)
         >>> u_x[0]
         -1.0*u[0] + 1.0*u[1]
         >>> u_x[nx]
         1.0*u[nx] - 1.0*u[nx - 1]
         >>> u_x[j]
         -0.5*u[j - 1] + 0.5*u[j + 1]

        ``` 
        """

        index = self._axis.val(indices)
        m     = self._axis.len
        n  = gridfunc.infer_shape(self.expr)[self._axis.local_id]
        expr = 0

        if self.gpu:
            from sympy import symbols
            i, j, k = symbols('i j k')
            gpu_indices = {0 : i, 1 : j , 2 : k}
        else:
            gpu_indices = {0 : index, 1 : index, 2 : index}


        if self.periodic:
            # Output symbolic coef 
            if self._coef:
                for i,idx in enumerate(self._operator_data['idx_interior']):
                    expr = expr + (self._coef['interior'][i]
                                  *gridfunc.eval(self.expr, self._axis.add(indices, int(idx)), **kw_args))
            # Output coef value
            else:
                for w,idx in zip(self._operator_data['op_interior'], self._operator_data['idx_interior']):
                    expr = expr + w*gridfunc.eval(self.expr, self._axis.add(indices, int(idx)), **kw_args)
            return expr

        if self._bounds[index] == self._bounds.tag_left:
            # Output symbolic coef 
            if self._coef:
                gpuidx = gpu_indices[self._axis.id]
                for i,idx in enumerate(self._operator_data['idx_left'][0,:]):
                    if self.diagonal:
                        expr = expr + (self._coef['left'][gpuidx, i]
                                      *gridfunc.eval(self.expr, 
                                                     self._axis.add(indices, 
                                                     0), 
                                                     **kw_args))
                        pass
                    else:
                        expr = expr + (self._coef['left'][gpuidx, i]
                                      *gridfunc.eval(self.expr, 
                                                     self._axis.add(indices, 
                                                     int(idx)-index), 
                                                     **kw_args))
            # Output coef value
            else:
                for w,idx in zip(self._operator_data['op_left'][index,:], self._operator_data['idx_left'][index,:]):
                    expr = expr + w*gridfunc.eval(self.expr, self._axis.add(indices, int(idx)-index), **kw_args)
        elif self._bounds[index] == self._bounds.tag_right:
            index = right(index, m)
            index_right = m-index-1
            newindices = self._axis.add(indices, n - 1, overwrite=True)
            # Output symbolic coef 
            if self._coef:
                if self.gpu:
                    gpuidx = gpu_indices[self._axis.id]
                else:
                    gpuidx = index_right
                for i,idx in enumerate(self._operator_data['idx_right'][0,:]):
                    if self.diagonal:
                        expr = expr + (self._coef['right'][gpuidx, i]
                                       *gridfunc.eval(self.expr, 
                                                      self._axis.add(indices,
                                                          -int(idx)), 
                                                      **kw_args))
                    else:
                        expr = expr + (self._coef['right'][gpuidx, i]
                                       *gridfunc.eval(self.expr, 
                                                      self._axis.add(newindices, -int(idx)), 
                                                      **kw_args))
# Output coef value
            else:
                for w,idx in zip(self._operator_data['op_right'][index_right,:], 
                                 self._operator_data['idx_right'][index_right,:]):
                    expr = expr + w*gridfunc.eval(self.expr, self._axis.add(newindices, -int(idx)), **kw_args)
        # Interior
        else:
            # Output symbolic coef 
            if self._coef:
                for i,idx in enumerate(self._operator_data['idx_interior']):
                    expr = expr + (self._coef['interior'][i]
                                  *gridfunc.eval(self.expr, self._axis.add(indices, int(idx)), **kw_args))
            # Output coef value
            else:
                for w,idx in zip(self._operator_data['op_interior'], self._operator_data['idx_interior']):
                    expr = expr + w*gridfunc.eval(self.expr, self._axis.add(indices, int(idx)), **kw_args)
        return expr

    def right(self, indices):
        """ 
        Returns the index of the right boundary by treating it like the left boundary index. For example,
        if there are \\(nx + 1\\) grid points then right(0) returns \\(nx\\). 

        # Examples

        ```python
        >>> from sympy import symbols
        >>> from openfd import GridFunction, sbp_traditional as sbp 
        >>> nx,j = symbols('nx j')
        >>> u    = GridFunction("u", shape=(nx+1,))
        >>> u_x  = sbp.Derivative(u, "x", order=2)
        >>> u_x.right(0)
        nx

        ```

        ```python
        >>> nx,ny,i,j = symbols('nx ny i j')
        >>> u    = GridFunction("u", shape=(nx+1, ny+1))
        >>> u_x  = sbp.Derivative(u, "x", order=2)
        >>> u_x.right((0, 1))
        (nx, 1)
        >>> u_y  = sbp.Derivative(u, "y", order=2)
        >>> u_y.right((0, 1))
        (0, ny - 1)
        >>> u_y[u_y.right((0, 1))]
        0.5*u[0, ny] - 0.5*u[0, ny - 2]

        ```

        """
        # Check bounds
        if is_sequence(indices):
            new_indices = list(indices)
            # Do not do anything if the index format is symbolic
            index       = sympify(indices[self._axis.id])
            index_dict  = index.as_coefficients_dict()
            if len(index_dict) > 1 or (len(index_dict) == 1 and index_dict[1] == 0 and index != 0):
                return indices

            if indices[self._axis.id] < 0:
                new_indices[self._axis.id] = self._shape[self._axis.id] + indices[self._axis.id]
            else:
                new_indices[self._axis.id] = self._shape[self._axis.id] - indices[self._axis.id] - 1
            if self._bounds.inbounds(new_indices[self._axis.id]):
                return tuple(new_indices)
            else:
                raise IndexError("Index out of bounds.")
        else:
            # Do not do anything if the index format is symbolic
            index       = sympify(indices)
            index_dict  = index.as_coefficients_dict()
            if len(index_dict) > 1 or (len(index_dict) == 1 and index_dict[1] == 0 and index != 0):
                return indices
            if indices < 0:
                new_indices = self._shape[self._axis.id] + indices
            else:
                new_indices = self._shape[self._axis.id] - indices - 1
            if self._bounds.inbounds(new_indices):
                return new_indices
            else:
                raise IndexError("Index out of bounds.")

    # Determines which boundary an index maps to, or interior. 
    def index_mapping(self,indices):
        if is_sequence(indices):
            return self._bounds[indices[self._axis.id]]
        else:
            return self._bounds[indices]

    def inbounds(self,indices):
        if is_sequence(indices):
            return self._bounds.inbounds(indices[self._axis.id])
        else:
            return self._bounds.inbounds(indices)

    def bounds(self, region=None):
        """
        Gives the bounds for a specific region of the operator.
        The bounds is defined as a tuple of the first and last index + 1 belonging to the 
        the particular region. 

        Parameters

        region : int, optional
                 `region = 0` : selects the left boundary region,
                 `region = -1` : selects the right boundary region,
                 `region = 1` : selects the interior region.

        Returns

        out : tuple,
              is the bounds for the selected region.

        out : Bounds,
              If `region = None` (default), then a bounds object is returned.

        """
        if region is None:
            return self._bounds
        if region == "left" or region == 0:
            return self._bounds.left_range
        if region == "right" or region == -1:
            return self._bounds.right_range
        if region == "interior" or region == 1:
            return self._bounds.interior_range
        raise ValueError("Invalid option.")

    def range(self, region):
        """
        Gives the range for a specific region of the operator.
        This function can only be used when the shape of the operator 
        can be determined.

        Parameters

        region : int, optional
                 `region = 0` : selects the left boundary region,
                 `region = -1` : selects the right boundary region,
                 `region = 1` : selects the interior region.

        Returns

        out : list,
              the range for the selected region

        """
        if region != 0 and region != 1 and region != -1:
            raise ValueError('Invalid region selected. Got `region = %s`. Expected `region = -1, 0, or 1` ')
        b = self.bounds(region)
        return range(b[0], b[1])

    def coef(self, region):
        """ 
        Returns the data used by the operator

        Parameters

        region : int,
                 selects the data region (0 : left, 1 : interior, -1 : right).

        Returns

        label : string,
                the name of data array
        data : numpy array,
               the data itself
        """

        if self.periodic:
            region = 1

        regions = [0, 1, -1, 2]
        region_labels = ['left', 'interior', 'right', 'right']

        if region not in regions:
            raise ValueError('Undefined region. Expected %s. Got: %s' 
                             %(', '.join(map(str, regions)), region))

        label = None
        data = None
        for reg, lbl in zip(regions, region_labels):
            if region == reg:
                return self._coef[lbl]

    def copy(self):
        obj = self.__new__(self._cls, self._expr, self._axis.axis, shape=self._shape)
        obj._is_matrix = self._is_matrix
        obj._operator_data = self._operator_data
        return obj

    @property
    def expr(self):
        return self._expr

    @property
    def axis(self):
        return self._args[1]

    @property
    def axis_id(self):
        return self._axis.id

    @property
    def shape(self):
        return self._shape
    
    @property
    def num_bnd_pts_left(self):
        return self._operator_data['op_left'].shape[0]
    
    @property
    def num_bnd_pts_right(self):
        return self._operator_data['op_right'].shape[0]

    @property
    def free_symbols(self):
        return self.expr.free_symbols

    @property
    def T(self):
        obj = self.copy()
        obj._transpose = not self._transpose 
        return obj

    @property
    def is_matrix(self):
        return self._is_matrix
    
    @property
    def is_transpose(self):
        return self._transpose

    @property
    def periodic(self):
        return self._periodic

    @property
    def symbol(self):
        return ''

    @property
    def gpu(self):
        return self._gpu

    def _sympystr(self, p):
        from sympy.printing import sstr
        out = str(self.symbol) + self._axis.axis
        if not self.is_matrix:
            out += '(' + str(self._expr) + ')'
        if self.is_transpose:
            out += '^T'
        return out

def operatordata(op_left, idx_left, op_right, idx_right, op_interior, idx_interior):
    if op_left is None:
        raise ValueError('No left operator data')
    if idx_left is None:
        raise ValueError('No left operator index data')
    if op_right is None:
        raise ValueError('No right operator data')
    if idx_right is None:
        raise ValueError('No right operator index data')
    if op_interior is None:
        raise ValueError('No interior operator data')
    if idx_interior is None:
        raise ValueError('No interior operator index data')

    d = {}
    d['op_left']      = op_left
    d['idx_left']     = idx_left
    d['op_right']     = op_right
    d['idx_right']    = idx_right
    d['op_interior']  = op_interior
    d['idx_interior'] = idx_interior

    return d

