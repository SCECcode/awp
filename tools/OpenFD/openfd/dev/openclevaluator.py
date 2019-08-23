import pyopencl as cl
from .kernelevaluator import KernelEvaluator, _evalsympy

class OpenclEvaluator(KernelEvaluator):
    """
        The OpenCL implementation of KernelEvaluator.
    """

    language = "OpenCL"

    def __init__(self, kernels, inputs={}, outputs={},
                 device_type=cl.device_type.GPU,
                 device_id=0,
                 platform_name=None):
        """
        Initialize the OpenCL computing environment. Afterward, use eval() and
        transfer() to evaluate kernels and obtain the results.

        :param kernels: A list of kernel objects in the order that they will be
                        called.
        :param inputs:  A dict containing the expression or symbol with their
                        initial value that must be assigned before calling
                        kernels.
        :param outputs: A dict containing expression or symbols that must be
                        accessible after calling the sequence of kernels.
        :param device_type: The type of the OpenCL compatible device to perform
                            the computation with. Default to GPU.
        :param device_id: The specific device in all available device to use.
                          Default to 0 (the first).
        :param platform_name: The name of the specific OpenCL icd to use.
                              Default to the first having the specified device
                              type available.
        """
        self.device_type = device_type
        self.device_id = device_id
        self.platform_name = platform_name

        self.device = None
        self.context = None
        self.queue = None

        self.programs = [None for _ in range(len(kernels))]
        self.clkernels = [None for _ in range(len(kernels))]

        self.kerndims = [[None, None] for _ in range(len(kernels))]

        super().__init__(kernels, inputs, outputs)

    def set_comp_env(self):
        """
        Find a suitable platform with the desired computing device type, connect
        the device, create a context and a queue.
        """
        platforms = cl.get_platforms()
        if self.platform_name is not None:
            platforms = [pf for pf in platforms if
                         pf.name == self.platform_name]
            if not platforms:
                raise ValueError("No platform with name %s could be found"
                                 % self.platform_name)
        self.device = None
        for platform in platforms:
            devices = platform.get_devices(self.device_type)
            if devices:
                try:
                    self.device = devices[self.device_id]
                except IndexError:
                    pass
            if self.device:
                break
        if not self.device:
            raise ValueError("No devices could be found")

        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

    def set_callables(self):
        """
        Provides the interface to call OpenCL kernels in the form callable(args)
        """

        def callkernel(clkernel, queue, g_dim, l_dim, args):
            for jj, arg in enumerate(args):
                clkernel.set_arg(jj, arg)
            cl.enqueue_nd_range_kernel(queue, clkernel, g_dim, l_dim)

        for ii, kernel in enumerate(self.kernels):
            self.programs[ii] = cl.Program(self.context, kernel.code).build()
            self.clkernels[ii] = getattr(self.programs[ii], kernel.name)
            self.kerndims[ii][0] = tuple(
                [_evalsympy(self.kernels[ii].gridsize[dim], self.inputs)
                 for dim in range(len(self.kernels[ii].gridsize))])

            self.callables[ii] = lambda arg, ii=ii: callkernel(self.clkernels[ii],
                                                        self.queue,
                                                        self.kerndims[ii][0],
                                                        self.kerndims[ii][1],
                                                        arg)

    def create_mem(self, initmem=None, size=0):
        """
        Create the OpenCL memory buffer. Initialize this buffer with the content
        of initmem if provided.

        :param initmem: A variable, i.e. a numpy array containing the memory
                        with which the buffer is to be initialized
        :param size:    Size in bytes of the buffer to be created. If initmem is
                        provided, size is not required.
        :return: the memory buffer specific to the language
        """
        mem_flags = cl.mem_flags
        if initmem is not None:  # This will copy the host memory
            flags = mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR
        else:
            flags = mem_flags.READ_WRITE
        return cl.Buffer(self.context, flags, hostbuf=initmem, size=size)

    def transfer(self, symvar, memobj):
        """
        Transfers the result of symvar from the computing device to the local
        memory object in python.

        :param symvar: The sympy variable for which the output is desired
        :param memobj: The python memory object in which the output with be
                       transfered.
        :return: The memory object with after transfer occurred.
        """
        # This blocks until the transfer is completed.
        cl.enqueue_copy(self.queue, memobj, self.allmems[symvar])
        return memobj


