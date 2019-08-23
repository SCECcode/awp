import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from .kernelevaluator import KernelEvaluator, _evalsympy
import numpy as np


class CudaEvaluator(KernelEvaluator):
    """
        The Cuda implementation of KernelEvaluator.
    """

    language = "Cuda"

    def __init__(self, kernels, inputs={}, outputs={},
                 device_id=0, block_size=(32, 32, 32)):
        """
        Initialize the Cuda computing environment. Afterward, use eval() and
        transfer() to evaluate kernels and obtain the results.

        :param kernels: A list of kernel objects in the order that they will be
                        called.
        :param inputs:  A dict containing the expression or symbol with their
                        initial value that must be assigned before calling
                        kernels.
        :param outputs: A dict containing expression or symbols that must be
                        accessible after calling the sequence of kernels.
        :param device_id: The specific device in all available device to use.
                          Default to 0 (the first).
        """

        self.device_id = device_id
        self.block_size = block_size

        self.device = None
        self.context = None
        self.stream = None

        self.programs = [None for _ in  range(len(kernels))]
        self.cudakernels = [None for _ in  range(len(kernels))]

        self.kerndims = [[None, None] for _ in range(len(kernels))]

        super().__init__(kernels, inputs, outputs)

    def set_comp_env(self):
        """
        Find a suitable platform with the desired computing device type, connect
        the device, create a context and a queue.
        """
        cuda.init()
        ndev = cuda.Device(0).count()
        if self.device_id>ndev:
            raise ValueError("No device with device id %d could be found"
                             % self.device_id)
        self.device = cuda.Device(self.device_id)
        self.context = self.device.make_context()
        self.stream = cuda.Stream()

    def set_callables(self):
        """
        Provides the interface to call CUDA kernels in the form callable(args)
        """

        def callkernel(clkernel, stream, nblocs, bsize, args):
            clkernel.prepared_async_call(nblocs, bsize, stream, *args)

        for ii, kernel in enumerate(self.kernels):
            self.programs[ii] = SourceModule(kernel.code)
            self.cudakernels[ii] = self.programs[ii].get_function(kernel.name)
            self.cudakernels[ii].prepare([type(arg) for arg in self.kernargs[ii]])
            self.kerndims[ii][0] = tuple(
                [np.long(_evalsympy(kernel.gridsize[dim], self.inputs))
                 for dim in range(len(kernel.gridsize))])

            bsize = self.block_size
            gsize = self.kerndims[ii][0]

            bsize = [1]*3  ## TODO supports only three dimensions for now
            nblocs = [1]*3
            for jj in range(len(gsize)):
                bsize[jj] = np.long(self.block_size[jj])
                nblocs[jj] = np.long((gsize[jj]+bsize[jj]-1)//bsize[jj])

            self.callables[ii] = lambda arg, ii=ii: callkernel(self.cudakernels[ii],
                                                               self.stream,
                                                               tuple(nblocs),
                                                               tuple(bsize),
                                                               arg)

    def create_mem(self, initmem=None, size=0):
        """
        Create the Cuda memory buffer. Initialize this buffer with the content
        of initmem if provided.

        :param initmem: A variable, i.e. a numpy array containing the memory
                        with which the buffer is to be initialized
        :param size:    Size in bytes of the buffer to be created. If initmem is
                        provided, size is not required.
        :return: the memory buffer specific to the language
        """
        if initmem is not None:  # This will copy the host memory
            mem = cuda.to_device(initmem)
        else:
            mem = cuda.mem_alloc(np.long(size))
            cuda.memset_d32(mem, 0, np.long(size/4))

        return mem

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
        cuda.memcpy_dtoh(memobj, self.allmems[symvar])
        return memobj

    def __del__(self):
        self.context.pop()

