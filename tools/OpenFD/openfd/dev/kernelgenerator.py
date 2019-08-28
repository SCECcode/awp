from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
from openfd.base import GridFunction, Constant
from sympy import symbols, sympify, expand, preorder_traversal
from sympy import ccode
from openfd.dev.variable import Variable
import numpy as np
import openfd


def make_kernel(label, 
                dout, 
                din, 
                bounds, 
                gridsize, 
                const=[], 
                extrain=[], 
                extraout=[], 
                extraconst=[], 
                indices=[], 
                regions=None,
                generator=None,
                debug=False, 
                precision=None,
                loop_order=None, 
                loop=False,
                index_bounds=None,
                reduction=None,
                lhs_indices=None,
                rhs_indices=None,
                grid_order=None, 
                launch_bounds=None
                ):
    """
    Generate code for compute kernels that can execute on either the CPU or GPU. 

    This function takes an expression of the form `dout = din` and generates
    code for executing this expression over a grid of different compute regions.
    For each compute region, a new kernel object is constructed. Each kernel
    object contains the generated code that should be executed for the
    particular compute region. The generated code is a function that contains
    both the function header and body.

    When one or more regions are specified, the kernel functions are labelled
    according to following conventions
    
    For 1D, 
    `label_x` : x = 0, 1, 2 which refers to the first, interior, and last
    compute regions.

    For 2D,
    `label_xy` : x = 0, 1, 2 (first, interior, last in x-direction), 
                 y = 0, 1, 2 (first, interior, last in y-direction),

    and so forth.
    
    If only one compute kernel is specified, the kernel function is simply
    labelled as `label`.

    Arguments:
        label : The name of kernel function to generate 
        dout : A tuple of output expressions
        din : A tuple of input expressions
        gridsize : Symbols that define the grid dimensions
        bounds : Kernel bounds
        regions(optional) : Compute regions to generate kernels for. Defaults to
            interior only.
        generator(optional) : `KernelGenerator` object. Defaults to `CGenerator`.
        debug(optional) : Set to `True` to enable kernel debugging.
        precision(optional) : Floating-point precision to use by generators.
            Defaults to `np.float32`. Pass `np.float64` to use double precision.
        loop(optional) : Enable loops. Default to False.
        loop_order(optional) : The order in which to execute loops. 
            Default loop order is `(0, 1, 2)`. Requires `loop = True` to enable
                loops for non C-code.
        index_bounds(optional) : Pass index bounds as input arguments to the
            kernel. Use this option to only compute for a subset of a region.
        reduction(optional) : Perform a reduction operation instead of the
            default assignment operation. Pass type type of reduction to perform
            to active `+`, `-`, `*`.
        lhs_indices(optional) : Remap the indices on the left-hand side using a
            function.
        rhs_indices(optional) : Remap the indices on the right-hand side using a
            function.
        grid_order: An array that contains the order of the CUDA thread indices.
            e.g., ['x', 'y', 'z'].
        launch_bounds(optional) : Add CUDA launch bounds to kernel. The launch
        bounds are specified by an array of size 2, one for each argument. The
        second argument can be left empty '' as it is optional. Defaults to
        `None`.



    Returns:
        A list of `Kernel` objects that contains the generated code for each
        compute region.
    """
    from openfd import GridFunctionExpression as Expr, utils

    if not generator:
        generator = openfd.codegen

    if not precision:
        precision = openfd.prec
    else:
        raise DeprecationWarning(
              "Precision option will be removed in a future"\
              " release. Currently it has not effect. Please"\
              " set precision directly via openfd.prec = np.float32"\
              ", or specify it for each object you use")

    if not debug:
        debug = openfd.debug

    
    if reduction:
        #TODO: Implement me
        raise NotImplementedError("Reduction is not yet implemented")

    bounds = utils.to_tuple(bounds)
    if regions is None:
        regions = utils.to_tuple([1]*len(bounds))
        use_region = False
    else:
        use_region = True

    dout = utils.to_tuple(dout)
    din  = utils.to_tuple(din)

    ka = generator(gridsize, bounds, 
                   [Expr(o) for o in dout], [Expr(i) for i in din],
                   debug=debug, 
                   loop=loop,
                   loop_order=loop_order,
                   index_bounds=index_bounds,
                   const=const,
                   extrain=extrain,
                   extraout=extraout,
                   extraconst=extraconst,
                   indices=indices, 
                   precision=precision,
                   grid_order=grid_order,
                   launch_bounds=launch_bounds
                   )

    # Append region ID to kernelfunction name unless only one region is used
    ids = {-1: '2', 0: '0', 1: '1'}
    kernels = []

    dim = len(bounds)

    kernel_fcn = lambda reg : ka.kernel(_kernelname(label, reg, use_region), 
                              reg,
                              indices=indices, left_index_mapping=lhs_indices,
                              right_index_mapping=rhs_indices)

    if dim == 1:
        for reg in utils.to_tuple(regions):
            kernel = kernel_fcn((reg,))
            kernels.append(kernel)
    elif dim == 2:
        for reg_x in utils.to_tuple(regions[0]):
            for reg_y in utils.to_tuple(regions[1]):
                reg = (reg_x, reg_y)
                kernel = kernel_fcn(reg)
                kernels.append(kernel)
    elif dim == 3:
        for reg_x in utils.to_tuple(regions[0]):
            for reg_y in utils.to_tuple(regions[1]):
                for reg_z in utils.to_tuple(regions[2]):
                    reg = (reg_x, reg_y, reg_z)
                    kernel = kernel_fcn(reg)
                    kernels.append(kernel)
    else:
        raise NotImplementedError('Kernel generator for more than three'\
                                  'dimensions is currently not supported')
    
    return kernels

def write_kernels(filename, kernels, header=False, 
                  kernelname=None,
                  source_includes=[], header_includes=[], include_header=False):
    """
    Write the compute kernels to source and header files.

    Arguments:
        filename : File to write to.
        kernels : List of kernels to write.
        header(optional) : Write header file. Defaults to `False`.
        kernelname(optional) : Label for C kernel functions.
        source_includes(optional) : Extra include statements placed in the
            source.
        header_includes(optional) : Extra include statements placed in the
            header.
        include_header(optional) : Include the header file automatically.
    """
    from openfd import utils

    language = kernels[0].language

    # Only generate code for kernels that use the same language
    filtered_kernels = _filter_kernels(language, kernels)

    sourcefile = filename + extension(language)['source']
    headerfile = filename + extension(language)['header']

    
    # source
    f = open(sourcefile, 'w')
    for include in utils.to_tuple(source_includes):
        f.write('%s\n' % include)
    if include_header:
        f.write('#include "%s"\n'%headerfile)
    f.write('\n')
    for kernel in filtered_kernels:
        f.write(kernel.code)
    f.close()

    if not header:
        return

    if not kernelname:
        kernelname = filename

    # header
    f = open(headerfile, 'w')
    f.write('#ifndef %s_H\n' % kernelname.upper())
    f.write('#define %s_H\n' % kernelname.upper())
    f.write('#include <math.h>\n')
    for include in utils.to_tuple(header_includes):
        f.write('%s\n' % include)
    f.write('\n')
    for kernel in kernels:
        f.write(kernel.header)
    f.write('#endif')
    f.close()

def extension(language):
    """
    Return a dictionary that contains the source and header file extensions for
    a kernel generated for a given language. The key `source` contains the
    source extension (`cu` for Cuda) and the key `header` contains the header
    extension.

    Arguments:
        language: Use 'Cuda' or 'C'.
    """

    if language == 'Cuda':
        return {'source' : '.cu', 'header' : '.cuh'}
    if language == 'C':
        return {'source' : '.c', 'header' : '.h'}


def kernelgenerator(language, gridsize, bounds, dout, din, const=[], extrain=[], 
                    extraout=[],
                    extraconst=[], indices=[], debug=False, loop=False,
                    loop_order=None, index_bounds=None):
    """
    Interface to generate a kernel function for a specific target language. 
    See `KernelGenerator` for details. 

    :param language : A string that specifies the language to generate code for. 
                      The available options are: `C`, `Cuda`, or `Opencl`.

    """
    from warnings import warn
    warn('This function will be deprecated in the near future.\n' 
         ' Please use `CGenerator`, `CudaGenerator`, or `OpenclGenerator`' \
         ' instead.', PendingDeprecationWarning)
    if language == 'C':
        return CGenerator(gridsize, bounds, dout, din, const, extrain, extraout,
                          extraconst, indices, debug, True, loop_order,
                          index_bounds) 
    elif language == 'Cuda': 
        return CudaGenerator(gridsize, bounds, dout, din, const, extrain,
                             extraout,
                             extraconst, indices, debug, loop, loop_order, 
                             index_bounds) 
    elif language == 'Opencl': 
        return OpenclGenerator(gridsize, bounds, dout, din, const, extrain,
                               extraout,
                               extraconst, indices, debug, loop, loop_order,
                               index_bounds) 
    else:
        raise NotImplementedError('No kernel generator for language: `%s`' 
                                  % language)

class Kernel(object):
    """
    A class containing the generated code of a KernelGenerator, along input
    arguments. A Kernel instance can be evaluated by the appropriate
    KernelEvaluator of the target language.
    """
    def __init__(self, name, code, header, code_args, gridsize, language,
                 precision):
        self.name = name
        self.code = code
        self._header = header
        self.code_args = code_args
        self.gridsize = tuple(gridsize)
        self.language = language
        self.precision = precision

    @property
    def header(self):
        return self._header + ';\n'


class KernelGenerator(object):
    """
    Base class to generate a kernel function by detecting the variables that
    should make up the function's argument list. Once initialized, use the
    method kernel() to generate a kernel object containing the code and the
    argument list. The kernel object that can be then be evaluated with
    KernelEvaluator implementations.

    Should be overridden to a specific target language.
    """

    language = ''

    def __init__(self, gridsize, bounds, dout, din, const=[], extrain=[],
                 extraout=[],
               extraconst=[], indices=None, debug=False, loop=False,
               loop_order=None, index_bounds=None, precision=None,
               grid_order=None, launch_bounds=None):
        """
        The base class init defines class members required for all child classes
        The sequence of functions called to generate the kernel code
        is defined here.

        Arguments:
            gridsize : A tuple that contains the symbols specifying grid size
                       (nx, ny, ...)
            bounds : A Bounds object that contains the ranges for the
                     different compute regions
            dout : A GridFunction to assign expression din to. This parameter
                   is needed to generate the argument list for the function.
            din : An Expr, the symbolic expression to evaluate.
                  This parameter is needed to generate the argument list for
                  the function.
            const : A list, optional, that contains any constants that should
                   be included in the argument list.
            extrain : A list, optional, containing any extra array input
                     arguments.
            extraout : A list, optional, containing any extra array output
                     arguments.
            extraconst : A list, optional, containing any extra constant
                         input arguments.
            indices: A str, optional, defining the labels of the indices. The
                     order of the indices defines the ordering of the loops.
            debug : A bool, optional. If `debug = True`, `rhs` is replaced by
                     a debugcode (int).
            loop_order : A tuple, optional, controlling the order of nested
                            for loops. Default to reversed ordering, 
                            `(2, 1, 0)`, that places the increment of the
                            fastest index in the inner-most loop.
            precision : Floating-point precision. Defaults to `np.float32`. 
                    Pass `np.float64` to use double precision.
            
        """
        from sympy import Symbol
        import numpy as np

        'Check inputs for right type'
        if type(gridsize) not in (list, tuple):
            gridsize = [gridsize]
        for gs in gridsize:
            if not isinstance(gs, Symbol):
                raise TypeError('Grid size parameter is of type %s.'\
                        ' Expected type %s.' % (type(gs), Symbol))
        self.gridsize = gridsize

        if type(bounds) not in (list, tuple):
            bounds = [bounds]
        self.bounds = bounds

        if type(dout) not in (list, tuple):
            dout = [dout]
        self.dout = dout

        if type(din) not in (list, tuple):
            din = [din]
        self.din = din

        if indices:
           self.indices = symbols(indices)
        else:
            self.indices = list(self.define_indices())

        self.loop = loop

        if not loop_order:
            # Default to reversed order so that fastest index is incremented in
            # the innermost loop.
            self.loop_order = range(len(bounds)-1, -1, -1)
        else:
            self.loop_order = loop_order

        if not precision:
            self.precision = np.float32
        else:
            self.precision = precision

        if type(index_bounds) not in (list, tuple):
            index_bounds = (index_bounds,)
        if index_bounds:
            self.index_bounds = self.define_index_bounds(index_bounds)
        else:
            self.index_bounds = [None]*len(bounds)

        if not grid_order:
            self.grid_order = ['x', 'y', 'z']
        else:
            self.grid_order = grid_order

        self.const = const
        self.extrain = extrain
        self.extraout = extraout
        self.extraconst = extraconst
        self.maxdim = 3
        self.launch_bounds = launch_bounds

        self.debug = debug
        self.dec_str()  #Should be overidden for a specific target language

    def kernel(self, name, region, indices=None, 
               left_index_mapping=None, right_index_mapping=None,
               ):
        """
        Generate a kernel for the specified region.

        Arguments:
            name: A str, kernel name.
            region: A tuple selecting the region for which the kernel is
                    outputted.
            left_index_mapping: A tuple containing functions that define how
                                each the indices are mapped for each
                                dimension when evaluated on the left-hand
                                side in a kernel function. 
            right_index_mapping: A tuple containing functions that define how
                                each the indices are mapped for each
                                dimension when evaluated on the left-hand
                                side in a kernel function. 
        Returns:
            A kernel object that can be evaluated by a KernelEvaluator.
        """

        if type(region) not in (list, tuple):
            region = [region]
        if len(region) != len(self.bounds):
            raise ValueError('region and bounds must have the same length')

        if not left_index_mapping:
           left_index_mapping = self.left_index_mapping
        
        if not right_index_mapping:
           right_index_mapping = self.right_index_mapping

        code = ''
        code += self.body(region, left_index_mapping, right_index_mapping)
        code = self.define_coefficients(region, code, left_index_mapping,
                        right_index_mapping)

        (header, args) = self.define_function(name, left_index_mapping,
                        right_index_mapping)
        
        code = self.append_body(name, header, code)

        gridsize = [None]*len(region)
        for ii in range(len(region)):
            gridsize[ii] = self.define_gridsize(self.bounds[ii], region[ii])

        return Kernel(name, code, header, args, gridsize, self.language, 
                      self.precision)


    def left_index_mapping(self, x):
        """
        Defines a mapping for the left hand-side indices.
        The default mapping 
        Override this function to change the default mapping. 

        :return ids: a tuple containing index labels for each dimension. 
        :return mapping: a tuple containing mappings for each dimension. 
        """

        return x
    
    def right_index_mapping(self, x):
        """
        Specifies how the right hand-side indices are defined.
        See `left_indices` for details.

        :return ids: a tuple containing index labels for each dimension. 
        :return mapping: a tuple containing mappings for each dimension. 
        """

        return x


    def body(self, region, left_index_mapping, right_index_mapping
             ):
        """
        Generate the body of the kernel's function for the selected region and
        returns the a string of the code

        Arguments:
            region: A tuple selecting the region for which the kernel is
                     outputted
            left_index_mapping: A tuple containing functions that define how
                                each the indices are mapped for each
                                dimension when evaluated on the left-hand
                                side in a kernel function. 
            right_index_mapping: A tuple containing functions that define how
                                each the indices are mapped for each
                                dimension when evaluated on the left-hand
                                side in a kernel function. 
        Returns;
            A str containing the body of the code.
        """

        if type(region) not in (list, tuple):
            region = [region]
        if len(region) != len(self.bounds):
            raise ValueError('region and bounds must have the same length')

        lmap = left_index_mapping(self.indices)
        rmap = right_index_mapping(self.indices)

        lindices = tuple(self._indices(lmap[ii], self.bounds[ii], region[ii])
                        for ii in range(0, len(self.bounds)))
        rindices = tuple(self._indices(rmap[ii], self.bounds[ii], region[ii])
                        for ii in range(0, len(self.bounds)))

        code = ''


        for ii in range(0, len(self.bounds)):
           # if not self.loop or (self.loop_order and ii not in
           #         self.loop_order):
                code += self.idgrid(ii, index_bounds=self.index_bounds[ii])
                code += self.ifguard(self.indices[ii], self.bounds[ii],
                        region[ii], self.index_bounds[ii])


        macros = self.get_macros()

        for macro in macros:
            code += macro.define() + '\n'


        if self.debug:
            dbgcode = self._debugcode(region)
        else:
            dbgcode = 0

        code_block = ''
        for l, r in zip(self.dout, self.din):
            code_block += self._expr2c(l, r, lindices, rindices, dbgcode)

        if self.loop and self.loop_order:
            code += self.addloop(region, code_block)
        else:
            code += code_block

        for macro in macros:
            code += macro.undefine() + '\n'

        return code

    def addloop(self, region, body):
        """
        Decorate the body of the function with a for loop.

        :param region: A tuple selecting the region for which the kernel is
                        outputted
        :param body: A string containing the body of the code to which the loop
                     should be added
        :return: The string body decorated with the loop string.
        """

        if type(region) not in (list, tuple):
            region = [region]
        if len(region) != len(self.bounds):
            raise ValueError('region and bounds must have the same length')

        ids = [self.indices[idx] for idx in self.loop_order]
        bnds = [self.bounds[idx] for idx in self.loop_order]
        region = [region[idx] for idx in self.loop_order]
        loop = Loop(bnds, region, ids)

        code = ''
        for ii in range(0, len(region)):
            code += loop.header(ii)

        tab = loop.indent(len(region))

        code += ''.join(['%s %s\n' % (tab, b) for b in body.split('\n')])

        for ii in range(len(region), 0, -1):
            code += loop.footer(ii)

        return code

    def define_coefficients(self, region, body, left_index_mapping,
                    right_index_mapping):
        """
        Look for required coefficients definitions and prepend the code with the
        definitons.

        :param region: A tuple selecting the region for which the kernel is
                        outputted
        :param body: A string containing the body of the code to which the array
                    definitions should be prepended
        :param left_index_mapping: A tuple containing functions that define how
                                   each the indices are mapped for each
                                   dimension when evaluated on the left-hand
                                   side in a kernel function. 
        :param right_index_mapping: A tuple containing functions that define how
                                   each the indices are mapped for each
                                   dimension when evaluated on the left-hand
                                   side in a kernel function. 
        :return: A string containing the body prepended with array definitions
        """
        from warnings import warn
        from . array import Array

        axis_dim = {'x': 0, 'y': 1, 'z': 2}
        code = ''
        coefs = {}

        def insert(coef): 
            code = ''
            # Only display the array once.
            if coef.label not in coefs:
                coefs[coef.label] = coef
                # Set array precision to match generator settings
                coef.dtype = self.cconst
                code = ccode(coef) + '\n'  
            # If two different arrays have been given the same label,
            # issue a warning.
            elif coef is not coefs[coef.label]: 
                warn('Coefficient array `%s` is already defined. ' 
                     % coef.label)
            return code

        # Search through indices for data arrays
        for expr in (list(left_index_mapping(self.indices)) +
                     list(right_index_mapping(self.indices))):
            for arg in preorder_traversal(expr):
                if isinstance(arg, Array):
                    code += insert(arg)

        for expr in self.din:
            for arg in preorder_traversal(expr):
                coef_meth = getattr(arg, "coef", None)
                axis_mem = getattr(arg, "axis", None)
                if coef_meth and axis_mem:
                    r = axis_dim[arg.axis]
                    coef = arg.coef(region[r])
                    code += insert(coef)
        return code+body

    def get_macros(self):
        """
        Get macros for each gridfunction.
        """
        macros = []
        labels = []
        args = [arg for arg in _freesymbols(self.din + self.dout)]
        for di in args:
            if isinstance(di,GridFunction):
                mac = di.macro(*self.indices)
                label = di.label
                if label not in labels and mac:
                    macros.append(mac)
                    labels.append(label)

        return macros


    def define_indices(self):
        """
        Defines indices for each dimension (i, j, k). Each index is a symbol.

        :return : A tuple containing the indices.

        """
        i, j, k = symbols('i j k')
        return (i, j, k)

    def define_index_bounds(self, indices):
        """
        Defines loop bounds for each dimension (i, j, k). 
        """
        i0, j0, k0 = symbols('bi bj bk')
        i1, j1, k1 = symbols('ei ej ek')
        bounds = ((i0, i1), (j0, j1), (k0, k1))
        
        out = [None]*len(bounds)
        for i, idx in enumerate(indices):
            if idx:
                out[i] = bounds[i]
        return out


    def define_function(self, name, left_index_mapping,
                        right_index_mapping):
        """
        Decorate the body of the function with the function definition

        Arguments:
            name: A string containing the name of the function
            left_index_mapping : Remapping of the indices in the left-hand side
            right_index_mapping : Remapping of the indices in the right-hand
                side

        Returns:
            A string containing the body prepended with coefficient definitions
        """
        dout = _freesymbols(self.dout) + self.extraout
        din = _freesymbols(self.din) + self.extrain + self.extraconst
        n = self.maxdim
        indices = list(_freesymbols(left_index_mapping([0]*n)) + 
                        _freesymbols(right_index_mapping([0]*n)))
        if self.index_bounds:
            for idx in self.index_bounds:
                if idx:
                    indices += [idx[0], idx[1]]
        indices = tuple(indices)

        if len(dout) == 0:
            raise ValueError("No output arguments found")

        if len(dout) == 0:
            raise ValueError("No output arguments found")

        din = [i for i in din if i not in dout]

        infun = lambda x: '%s %s%s' % (self.dtypes._const,  \
                                        self.dtypes.get_ptr(x.dtype), x) \
                           if isinstance(x, GridFunction) \
                           else '%s %s' % (self.cconst, x)
        outfun = lambda x: '%s%s' % (self.dtypes.get_ptr(x.dtype), x) \
                            if isinstance(x, GridFunction)  \
                            else '%s %s' % (self.cconst, x)
        insym = sorted(map(infun, din)) 
        outsym = sorted(map(outfun, dout))

        # [0:-2] is used to remove trailing ','
        str_cout = ''.join(['%s, ' % (d) for d in sorted(set(outsym))])[0:-2]
        str_cin  = ''.join([', %s' % (d) for d in sorted(set(insym))])

        str_cint = ''.join(
            [', %s %s' % (self.cint, d) for d in 
            sorted(set(map(str, self.gridsize)))])
        str_cconst = ''.join(
            [', %s %s' % (self.cconst, d) for d in 
            sorted(set(map(str, self.const)))])
        str_cintv = ''.join(
            [', %s %s' % (self.cint, d) 
            for d in sorted(set(map(str, indices)))])


        code = self.fundec
        code += '%s %s(%s%s%s%s%s)' % (
            self.cret, name, str_cout, str_cin, str_cint, 
            str_cconst, str_cintv)

        args = dout + din + self.const + list(self.gridsize)

        return code, args

    def append_launch_bounds(self, name):

        code = ''
        if self.launch_bounds:
            if self.launch_bounds[0] == 'yes' and self.launch_bounds[1] != 'yes':
                code += '__launch_bounds__(%s_MAX_THREADS_PER_BLOCK)' % (name.upper())
            if self.launch_bounds[0] == 'yes' and self.launch_bounds[1] == 'yes':
                code += '__launch_bounds__(%s_MAX_THREADS_PER_BLOCK,\
                %s_MIN_BLOCKS_PER_SM)' % ( name.upper(), name.upper())
            code += '\n\n'
        return code

    def append_body(self, name, header, body):
        """
        Appends the function body after a function declaration

        Arguments:
            name(`str`) : Function name.
            header(`str`) : Function declaration ( without `;`).
                This string is generated by calling `define_function()`.
            body : Code body. This string is generated by calling `body()`.
        """
        
        code = ''
        code += self.append_launch_bounds(name)
        code += header + '\n'
        code += '{\n'
        code += ''.join(
            ['%s %s\n' % (_tab(1), b) for b in body.split('\n')])
        code += '}\n\n'
        return code

    def define_gridsize(self, bounds, region):
        """
        Compute the grid size for the region in bounds

        :param bounds: (bound) A bounds object
        :param region: (int) The region indice
        :return: (int) The region size
        """

        if region == 0:
            return bounds.left
        elif region == 1:
            return bounds.interior
        #TODO: Remove region == -1
        elif region == -1 or region == 2:
            return bounds.right
        else:
            raise ValueError("Region index must be 0, 1 or 2")

    def _indices(self, i, bounds, region):
        """
        Gives the lhs and rhs indices.

        :param i: symbolic index
        :param bounds: (bound) A bounds object
        :param region: (int) The region index
        :return: An expr with the indices
        """
        #TODO: Replace this solution with something better
        from openfd.base.index import Left, Right

        if region == 0:
            l = Left(i)
            return l
        if region == 1:
            r = bounds.range(1)
            return i + r[0]
        if region == 2:
            ri = Right(i)
            return ri
        #TODO: Deprecate region == -1
        if region == -1:
            ri = Right(i)
            return ri

    def _debugcode(self, region):
        """
        Debug codes for checking which regions are written to by the GPU.

        :param region: A tuple selecting the region for which the kernel is
                        outputted
        :return: An int for the debug code
        """

        #TODO: Deprecate region == -1
        codes = {0: 0, 1: 1, -1: 2, 2: 2}
        if isinstance(region, int):
            return codes[region] + 1

        code = 0
        for i, r in enumerate(region):
            code += 3 ** i * codes[r]
        return code + 1

    def _expr2c(self, lhs, rhs, lhs_indices, rhs_indices, 
                debugcode=0):
        """
        Convert a sympy expression into c code.

        Arguments:
            lhs: An Expression containing the variable being assigned in the
                 code
            rhs: An Expression defining operations to assign to the lhs.
            lhs_indices: An expression for the indices of lhs.
            rhs_indices: An expression for the indices of rhs.
            debugcode: If > 0, assign the debug code to the lhs.

        Returns:
            A string containing the c code to evaluate lhs = rhs .
        """
        from sympy import ccode

        l = lhs[lhs_indices]
        r = rhs[rhs_indices]

        if isinstance(l, Variable):
            pass
        if debugcode > 0:
            return '%s = %s;\n' % (ccode(l), debugcode)
        else:
            return '%s = %s;\n' % (ccode(l), ccode(r))

    def dec_str(self):
        """
            Define how types are defined in the target language.
            Overload this function if type specifiers are different in the
            target language.
        """

        import numpy as np
        from . import types

        self.dtypes = types.C

        self.cin = 'const float *'  # Declaration of input arguments
        self.cout = 'float *'  # Declaration of output arguments
        self.cint = 'const int'  # Declaration of integer arguments
        self.cintv = ' int *'  # Declaration of pointer integer arguments
        self.cconst = 'const float'  # Declaration of float arguments
        self.ccoef = 'const float'  # Precision for stencil coefficients
        self.cret = 'void'  # Type of return value
        self.fundec = ''  # How a function is declared

        if self.precision == np.float64:
            self.cin = 'const double *'
            self.cout = 'double *'
            self.cconst = 'const double'
            self.ccoef = 'const double'


    def idgrid(self, idx, offset=0, prefix='', index_bounds=None):
        """
        Defines how grid indices are obtained.
        Overload this function to define grid position specifiers
        in the target language.

        :param idx: An int giving the dimension number (0:x, 1:y, 2:z)
        :param offset: An int giving the offset of the grid id.
        :param prefix:
        :return: A string containing a line of code defining the grid id for
                dimension idx.
        """

        raise NotImplementedError()

    def ifguard(self, indsym, bounds, region, index_bounds=None):
        """
        Defines how thread should be guarded.
        Overload this function to define grid boundaries specifiers
        in the target language.

        :param indsym: A symbol containing the indice symbol.
        :param bounds : A Bounds object that contains the ranges for the
                        different compute regions
        :param region: A tuple selecting the region for which the kernel is
                        outputted
        :return: A string containing a line of code defining the grid guard for
                dimension indsym
        """

        raise NotImplementedError()

class CGenerator(KernelGenerator):

    language = "C"

    def dec_str(self):
        """
            Define how types are defined in C.
        """
        import numpy as np
        from . import types

        self.dtypes = types.C

        self.cin = 'const float *'  # Declaration of input arguments
        self.cout = 'float *'  # Declaration of output arguments
        self.cint = 'const int'  # Declaration of integer arguments
        self.cintv = 'int *'  # Declaration of pointer integer arguments
        self.cconst = 'const float'  # Declaration of float arguments
        self.cret = 'void'  # Type of return value
        self.fundec = ''  # How a function is declared
        self.ccoef = 'const float'

        if self.precision == np.float64:
            self.cin = 'const double *'
            self.cout = 'double *'
            self.cconst = 'const double'
            self.ccoef = 'const double'

    def idgrid(self, idx, offset=0, prefix='', index_bounds=None):
        """
            No grid position is required in C.
        """
        return ''

    def ifguard(self, indsym, bounds, region, index_bounds=None):
        """
            No index guards are required in C.
        """
        return ''

class CudaGenerator(KernelGenerator):

    language = "Cuda"

    def dec_str(self):
        """
            Define how types are defined in Cuda.
        """

        import numpy as np
        from . import types

        self.dtypes = types.C

        self.cin = 'const float *'  # Declaration of input arguments
        self.cout = 'float *'  # Declaration of output arguments
        self.cint = 'const int'  # Declaration of integer arguments
        self.cintv = 'const int *'  # Declaration of pointer integer arguments
        self.cconst = 'const float'  # Declaration of float arguments
        self.ccoef = 'const float'  # Precision for coefficients
        self.cret = 'void'  # Type of return value
        self.fundec = '__global__ '

        if self.precision == np.float64:
            self.cin = 'const double *'
            self.cout = 'double *'
            self.cconst = 'const double'
            self.ccoef = 'const double'

    def idgrid(self, idx, offset=0, indices=None, index_bounds=None):
        """
        Defines how grid indices are obtained in Cuda.

        Arguments:
            idx: An int giving the dimension number (0:x, 1:y, 2:z)
            offset: An int giving the offset of the grid id.
            indices: A tuple/list, optional, containing the labels for each
                        index ('i', 'j', 'k').
            index_bounds : An optional tuple that contains the index bounds for
                the given dimension.

        Returns:
        A string containing a line of code defining the grid id for
                dimension idx.
        """

        if not indices:
            indices = ['i', 'j', 'k']
        idgrids = self.grid_order

        index = indices[idx]
        idgrid = idgrids[idx]

        idstr = ''
        if offset == 0:
            idstr = ('const int %s = threadIdx.%s + blockIdx.%s*blockDim.%s'
                    % (index, idgrid, idgrid, idgrid))
        else:
            idstr = ('const int %s = threadIdx.%s + %d'
                    % (index, idgrid, offset))
        if index_bounds:
            idstr += ' + %s'%(index_bounds[0])
        return idstr + ';\n'


    def ifguard(self, indsym, bounds, region, index_bounds=None):
        """
        Defines how a thread should be guarded in Cuda

        :param indsym: A symbol containing the indice symbol.
        :param bounds : A Bounds object that contains the ranges for the
                        different compute regions
        :param region: A tuple selecting the region for which the kernel is
                        outputted
        :return: A string containing a line of code defining the grid guard for
                dimension indsym
        """
        r = bounds.range(region)
        code = 'if ( %s >= %s) return;\n' % (indsym, r[1] - r[0])
        if index_bounds:
            code += 'if ( %s >= %s) return;\n' % (indsym, index_bounds[1])
        return code


class OpenclGenerator(KernelGenerator):

    language = "OpenCL"

    def dec_str(self):
        """
            Type specifiers for OpenCL
        """
        import numpy as np
        from . import types

        self.dtypes = types.C

        self.cin = '__global float *'  # Declaration of input arguments
        self.cout = '__global float *'  # Declaration of output arguments
        self.cint = 'const int'  # Declaration of integer arguments
        self.cintv = '__global int *'  # Declaration of pointer integer arguments
        self.cconst = 'const float'  # Declaration of float arguments
        self.ccoef = 'const float'  # Precision for coefficients
        self.cret = 'void'  # Type of return value
        self.fundec = '__kernel '

        if self.precision == np.float64:
            self.cin = '__global double *'
            self.cout = '__global double *'
            self.cconst = 'const double'
            self.ccoef = 'const double'

    def idgrid(self, idx, offset=0, prefix='', index_bounds=None):
        """
        Defines how grid indices are obtained in OpenCL

        :param idx: An int giving the dimension number (0:x, 1:y, 2:z)
        :param offset: An int giving the offset of the grid id.
        :param prefix:
        :return: A string containing a line of code defining the grid id for
                dimension idx.
        """

        names = {0: 'i', 1: 'j', 2: 'k'}
        name = names[idx]

        id_str = '' 
        if offset == 0:
            id_str += ('const int %s = get_global_id(%d)'
                       % (prefix + name, idx))
        else:
            id_str += ('const int %s = get_global_id(%d) + %d'
                       % (prefix + name, idx, offset))
        if index_bounds:
            id_str += ' + %s' % (index_bounds[0])
        return id_str + ';\n'

    def ifguard(self, indsym, bounds, region, index_bounds=None):
        """
        Defines how thread should be guarded in OpenCL

        :param indsym: A symbol containing the indice symbol.
        :param bounds : A Bounds object that contains the ranges for the
                        different compute regions
        :param region: A tuple selecting the region for which the kernel is
                        outputted
        :return: A string containing a line of code defining the grid guard for
                dimension indsym
        """
        r = bounds.range(region)
        code = 'if ( %s >= %s) return;\n' % (indsym, r[1] - r[0])
        if index_bounds:
            code += 'if ( %s >= %s) return;\n' % (indsym, index_bounds[1])
        return code

class Loop:
    def __init__(self, bounds, region, syms, use_header=True):
        self.use_header = use_header
        self.region = region
        bnds = self._bounds(bounds, region)
        self.bounds = bnds
        self.syms = syms

    def header(self, i):
        if not self.use_header:
            return ''
        bounds = self.bounds[i]
        return self.indent(i) + 'for (int %s = %s; %s < %s; ++%s) {\n' % \
                                (self.syms[i], 0, self.syms[i],
                                 bounds[1]-bounds[0], self.syms[i])

    def indent(self, i):
        return _tab(i)

    def range(self, i):
        return range(*self._loop_bounds(self.bounds[i], self.region[i]))

    def footer(self, i):
        if not self.use_header:
            return ''
        return self.indent(i-1) + '}\n'

    def _set_bounds(self, bounds, idx):
        if idx == 0 or idx == -1:
            a = bounds[1] - bounds[0]
            return (0, a)
        else:
            return (bounds[0], bounds[1])

    def _bounds(self, bounds, block):
        assert len(bounds) == len(block)
        br = lambda b, idx  : b.range(idx)
        return [self._set_bounds(br(b, idx), idx) for b, idx in zip(bounds, block)]

    def _loop_bounds(self, bounds, idx):
        if idx == 0 or idx == -1:
            return bounds
        else:
            return (0, 1)

def _tab(i, space=4):
    return ''.join([' ' for k in range(space*i)])

def _freesymbols(obj):
    ds = []
    for o in obj:
        try:
            fs = sympify(o).free_symbols
            for f in fs:
                if hasattr(f,'visible') and not f.visible:
                    incl = 0
                else:
                    ds.append(f)
        except:
            pass
    return ds

def _kernelname(label, reg, use_region):
    """
    Return the kernel name according to the labelling convention `'label_x'`
    etc. If no regions are used, then the label is simply `'label'`.

    Arguments:
        label : The label.
        reg : Current region (tuple, or int)
        use_region : Set to true if regions are active
    """
    from openfd import utils
    if not use_region:
        return label
    else:
        #FIXME: Refactor region code by placing it in its own module
        regions = {0 : 0, 1 : 1, -1 : 2, 2 : 2}
        for r in utils.to_tuple(reg):
            if r not in regions:
                raise ValueError('Region index must be 0, 1, or 2') 
        return label + '_' + ''.join([str(regions[r]) 
                                      for r in utils.to_tuple(reg)])


def _filter_kernels(language, kernels):
    """
    Select all kernels that support the same target language.

    """
    import warnings

    out = []
    skipped_lang = []

    mixed_lang = 0
    for kernel in kernels:
        if language != kernel.language:
            mixed_lang = 1
            skipped_lang.append(kernel.language)
        out.append(kernel)

    if mixed_lang:
        warnings.warn("Multiple languages in the same kernel is not "\
                      "supported. Skipping language(s): <%s> " % 
                       ', '.join(skipped_lang)) 

    return out
