from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object) 
from openfd.base import GridFunction, Constant
from openfd.dev.kernelgenerator import Kernel
from sympy import Symbol, sympify
import numpy as np



## TODO Add an initializer method to reinitialize computations
## TODO Have a Sampler operator that apply a function at sparse grid locations
### TODO Add C Evaluator 


def kernelevaluator(language, *arg, inputs={}, outputs={}):
    """
    Interface to generate a kernel function for a specific target language. 
    See `KernelEvaluator` for details. 

    :param language : A string that specifies the language to generate code for. 
                      The available options are: `C`, `Cuda`, or `Opencl`.
    :param arg : Additional input arguments passed to `KernelEvaluator`.

    """

    from warnings import warn
    warn('This function will be deprecated in the near future.')

    if language == 'Cuda': 
        return CudaEvaluator(*arg, inputs, outputs) 
    elif language == 'Opencl': 
        return OpenclEvaluator(*arg, inputs, outputs) 
    else:
        raise NotImplementedError('No kernel evaluator for language: `%s`' 
                                  % language)


class KernelEvaluator(object):
    """
    Base class to override for each implementation. KernelEvaluator allows to
    evaluate a list of kernels in the order provided.
    """
    language = ""

    def __init__(self, kernels, inputs={}, outputs={}):
        """
        The base class init defines class members required for all child classes
        The sequence of functions called to set up the computing environment
        is defined here. Define arguments of the overridden methods in the
        child's __init__ .

        :param kernels: A list of kernel objects in the order that they will be
                        called.
        :param inputs:  A dict containing the expression or symbol with their
                        initial value that must be assigned before calling
                        kernels.
        :param outputs: A dict containing expression or symbols that must be
                        accessible after calling the sequence of kernels.
        """

        if not all(isinstance(kernel, Kernel) for kernel in kernels):
            raise TypeError("arg kernels should be a list of Kernel instances")
        for kernel in kernels:
            if self.language != kernel.language:
                raise TypeError("Kernel was created for target language %s but "
                                "trying to evaluate with %s Evaluator"
                                % (kernel.language, self.language))



        self.kernels = kernels
        self.inputs = inputs
        self.outputs = outputs
        self.allinputs = {}
        self.allmems = {}
        self.kernargs = [[] for _ in range(len(kernels))]
        self.callables = [None for _ in range(len(kernels))]

        self.create_arg_list()

        # These methods are called in the base class init. If a specific
        # implementation's method require arguments, set them as child's
        # class members before calling super().__init__ .
        self.set_comp_env()
        self.prepare_inputs()
        self.set_callables()

    def eval(self):
        """
        Evaluate the provided kernels in the sequence of the kernels list input.
        """
        for ii, call in enumerate(self.callables):
            call(self.kernargs[ii])

    def get_outputs(self, outputs=None):
        """
        Obtain the desired outputs contained in the outputs dict. The output for
        of sym will be in outputs[sym]. Relies of the method transfer that must
        be overridden for each implementation.

        :param outputs: A dict containing the expr or symbols to output with the
                        buffer memory in which to output it. If not provided,
                        default to self.outputs.
        :return: A tuple containing the desired outputs
        """
        if outputs is None:
            outputs = self.outputs

        return tuple(
            self.transfer(svar, memo) for svar, memo in outputs.items())

    def transfer(self, symvar, memobj):
        """
        Transfers the result of symvar from the computing device to the local
        memory object in python.
        Override this function for the specific implementation.

        :param symvar: The sympy variable for which the output is desired
        :param memobj: The python memory object in which the output with be
                       transfered.
        :return: The memory object with after transfer occurred.
        """
        raise NotImplementedError()

    def create_arg_list(self):
        """
        Aggregate all unique inputs to the kernels in allinputs dict.
        """
        self.allinputs = {}
        for kernel in self.kernels:
            for arg in kernel.code_args:
                if arg not in self.allinputs:
                    if arg in self.inputs:
                        self.allinputs[arg] = self.inputs[arg]
                    else:
                        self.allinputs[arg] = None

    def set_comp_env(self):
        """
        Set up the computing environment of the evaluator. In Cuda and OpenCl,
        it will connect to computing devices and create a context for example.
        Default to: do nothing. Override if needed.
        """
        pass

    def prepare_inputs(self):
        """
        Generate the arrays for numerical evaluation and pack them in the dict
        self.allmems. Assign to the list self.kernargs the list of each kernel's
        memory argument.
        Each implementation has to define the create_mem method.
        :return:
        """

        for arg, val in self.allinputs.items():
            if isinstance(arg, (GridFunction, Constant)):
                if val is not None:
                    self.allmems[arg] = self.create_mem(initmem=val)
                else:
                    shape = _evalsympy(arg.shape, self.inputs)
                    bufsize = 4  # TODO We work with floats, but fix for other
                    for dim in shape:
                        bufsize *= dim
                    self.allmems[arg] = self.create_mem(size=bufsize)
            elif isinstance(arg, Symbol):
                if val is None:
                    raise ValueError("Input value for symbol %s  is"
                                     " not defined" % arg)
                self.allmems[arg] = val

            elif isinstance(arg, str):
                if val is None:
                    raise ValueError("Input value for arg %s  is"
                                     " not defined" % arg)
                self.allmems[arg] = val
            else:
                raise ValueError("Input type for arg %s  is"
                                 " not supported" % arg)

        for ii, kernel in enumerate(self.kernels):
            self.kernargs[ii] = [self.allmems[arg] for arg in kernel.code_args]

    def create_mem(self, initmem=None, size=0):
        """
        Create the memory buffer. Initialize this buffer with the content of
        initmem if provided.
        Override this function for the specific implementation.

        :param initmem: A variable, i.e. a numpy array containing the memory
                        with which the buffer is to be initialized
        :param size:    Size in bytes of the buffer to be created. If initmem is
                        provided, size is not required.
        :return: the memory buffer specific to the language
        """
        raise NotImplementedError()

    def set_callables(self):
        """
        Create a list of callable objects in self.callables. Each callable can
        be used in the form callable(args) to launch each kernel with specified
        arguments args.
        Override this function for the specific implementation.
        """
        raise NotImplementedError()

def init_h(variables, shape, precision=None):
    """
    Initialize variables to zero on the host.

    Arguments:
        variables : String of variables to initialize, e.g., 'p v'.
        shape : Array shape.
        precision(optional) : Precision. Pass `np.float32` for single precision
            (default) or `np.float64` for double precision.

    Returns:
        A dict of initialized variables.


    """
    import numpy
    from openfd import Struct
    import openfd

    if not precision:
        precision = openfd.prec

    zero = lambda : numpy.zeros(shape).astype(precision)

    cpu = {'%s'% var : zero() for var in variables.split(' ')}
    return Struct(**cpu)

def fetch_kernels(source, ftype='void'):
    """
    Fetch kernel function names and argument list from a kernel source file.

    Arguments:
        source : Contents of kernel source file.
        ftype(optional) : The function type. Defaults to `void`.

    """

    import re

    match = re.findall('%s (\w*)(\(.*\))' % (ftype), source)

    if not match:
        raise ValueError('No kernels found.')

    return match

def find_kernels(query, source, ftype='void'):
    """
    Parses a kernel string and returns the individual kernels. Each kernel
    should be separated by a space and a wildcard is accepted to search for
    kernels that share the same name, e.g., `kernel_0` `kernel_1`.

    Usage:
        `kernel_*` to find all kernels labelled `kernel_0`, `kernel_01` etc.
        `a b c` to find all kernels `a`, `b`, and `c`.

    Arguments:
        query : Kernels to search for.
        source : Contents of kernel source file.
        ftype(optional) : The function type. Defaults to `void`.

    """
    import re
    matches = []
    queries = query.split(' ')
    kernels = fetch_kernels(source, ftype)
    kernel_str = ' '.join([kernel[0] for kernel in kernels]) + ' '

    for q in queries:
        matches += re.findall(r'\b(%s)\s' % q.replace('*', '\w*'), kernel_str)

    if not matches:
        raise ValueError('No kernels found.')

    return matches

def _evalsympy(expr, inputs):
    eval_subs = []
    symexp = sympify(expr)
    for sym in symexp.free_symbols:
        try:
            val = inputs[sym]
        except KeyError:
            raise ValueError("Input value for symbol %s is"
                             " not defined" % sym)
        eval_subs.append((sym, val))
    return symexp.subs(eval_subs)

def get_dest_source_keys(dest, source, dest_vars, source_vars):
    """
    Read `source_vars` and `dest_vars` strings to determine what keys to copy.
    Use same `source_vars`  as `dest_vars` if the source str is unspecified.
    Copy all keys if `dest_vars` is empty.

    Arguments:

        dest : Dictionary of destination variables.
        source : Dictionary of source variables.
        dest_vars : String of space separated destination variables.
        source_vars : String of space separated source variables.

    """

    if dest_vars and not source_vars:
        source_vars = dest_vars

    if not source_vars:
        source_keys = [key for key in source]
    else:
        source_keys = source_vars.split(' ')

    if not dest_vars:
        dest_keys = [key for key in dest]
    else:
        dest_keys = dest_vars.split(' ')
    return dest_keys, source_keys

