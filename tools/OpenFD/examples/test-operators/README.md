# README
This example demonstrates the usage of staggered grid SBP operators that are
configured using the same number of grid points in OpenFD. 

## Contents:
 * `kernel.py` is a Python script that uses OpenFD to generate C compute kernels
 for computing on staggered grids in 1D.
 * `test_operators.c` is a C program that verifies that the generated kernels are
 correct.

## Installation
Use the provided `Makefile` to generate the compute kernels and compile the
test program (you may need to tweak this file for your system):
```bash
$ make
```
## Usage
Type
```bash
$ ./test_operators
```
to access the help screen:
```
usage: test_derivative <a> <n> <dist> <disth>

Test the accuracy of numerical differentiation using monomials:
f(x) = x^a/a! for -1 <= x <= 1.
    a    Integer that defines the monomial basis of test functions:
        { 1, x, x^2/2, ... x^a!/a}
    n    Number of grid points
    dist Ascii file containing boundary-modified distances for the regular grid
    disth Ascii file containing boundary-modified distances for the shifted grid
```

Example output:

```
$ ./test_operators 3 20
Testing differentiation of x^a/a!.
a = 0 
	 || u - ans|| = 2.881131e-06 (left) 	 2.999024e-06 (right) 	 3.294771e-07 (interior) 
	 ||uh - ans|| = 3.0161359e-06 (left) 	 3.0051167e-06 (right) 	 3.294771e-07 (interior) 
a = 1 
	 || u - ans|| = 2.5401805e-06 (left) 	 2.590803e-06 (right) 	 1.9768625e-07 (interior) 
	 ||uh - ans|| = 2.3380423e-06 (left) 	 2.3902214e-06 (right) 	 7.7409584e-07 (interior) 
a = 2 
	 || u - ans|| = 1.1221584e-06 (left) 	 1.062903e-06 (right) 	 8.0638763e-08 (interior) 
	 ||uh - ans|| = 9.9082456e-07 (left) 	 9.3612817e-07 (right) 	 2.0292482e-07 (interior) 
a = 3 
	 || u - ans|| = 0.00079769403 (left) 	 0.0007979236 (right) 	 9.5049746e-09 (interior) 
	 ||uh - ans|| = 0.0017050586 (left) 	 0.0017051602 (right) 	 4.5872447e-08 (interior) 

Testing interpolation of x^a/a!.
a = 0 
	 || u - ans|| = 3.2036533e-07 (left) 	 3.1789145e-07 (right) 	        0 (interior) 
	 ||uh - ans|| = 3.5982868e-07 (left) 	 3.1789145e-07 (right) 	        0 (interior) 
a = 1 
	 || u - ans|| = 3.0392525e-07 (left) 	 2.6507504e-07 (right) 	 4.2855458e-08 (interior) 
	 ||uh - ans|| = 6.4257631e-07 (left) 	 6.7259117e-07 (right) 	 3.3064502e-08 (interior) 
a = 2 
	 || u - ans|| = 0.0025871233 (left) 	 0.0025871198 (right) 	 1.4909357e-08 (interior) 
	 ||uh - ans|| = 0.0027689985 (left) 	 0.0027690178 (right) 	 5.6676717e-09 (interior) 
 
Computing spectral radius rho(A) for test problem:
 A = [ 0  D ] 
     [ Dh 0 ].
h*rho(A) = 4.48287 
```

# Further details
See the description in `kernel.py` to see how to learn how to use these
operators in your own projects and also to understand some of the technical
details involved. These details are important to understand if you wish to
develop your own operators.
