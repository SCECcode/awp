# 1D Linear advection equation solver
This example contains a complete implementation of a high order finite
difference 1D advection equation solver that runs on CUDA-enabled GPUs. A
low-storage fourth order Runge-Kutta scheme is used to advance the solution in
time. The solver is capable of outputting the solution to `.vtk` files at
selected time steps that can be visualized in e.g., Paraview.

The purpose of this example is to demonstrate how many of the features of OpenFD
work together to give you the ability to solve time dependent problems on the
GPU. To keep the numerical details to a minimum and focus as much as possible on
learning on how to use OpenFD, we think it is a good place to start with the
advection equation (PDE) in 1D.

## Numerical details
See the document [README.pdf](README.pdf) for numerical background and
additional details about how the implementation works.

## Usage
Modify the parameters in the file `params_adv.py` for the case you want to solve. 

The provided `Makefile` can be used to perform some required initializations
(make a output directory), generate the compute kernels, and finally run the
solver. All of these tasks can be accomplished at once by the command
```bash
$ make 
```
Alternately, each task can be performed individually. To perform initialization, 
type
```bash
$ make init
```
To generate GPU compute kernels, type
```bash
$ make kernel
```
Running this command will produce the file `kernel.cu`

To run the solver, type
```bash
$ make solver
```
While the solver is running it will write output to disk.  These output files
are placed in the directory `vtk/`. If this directory does not exist, the solver
will fail. Use `make init` or create this directory first.


#TODO 
To run a convergence test, type
```bash
$ make test
```
>>>>>>> develop
