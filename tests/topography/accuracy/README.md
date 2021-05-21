# README

This directory contains some additional tests that ensures that topography kernels are correctly
implemented. 

## Truncation error test
The convergence test modifies the velocity kernels so that the free surface boundary condition is no
longer imposed. This change makes it possible to check the rate at which the truncation errors go to
zero everywhere in the domain. The idea is to take the discretized spatial elastic operator and apply it to a
set of known trigonometric functions and compare it against an exact solution. The exact solution
comes from symbolically evaluating the spatial derivatives in the elastic wave equation for the
given test functions.

Since all calculations are performed in single precision, it is quite difficult to assess if the
implementation is correct. To improve the confidence, I tested not only with trigonometric functions
but also with polynomials. These polynomials get differentiated to machine precision as long as
their degree is below one, or two, depending on if the geometry is flat or not. Note that the
last boundary point for some of the field components is always zero. This is because this point is
not part of the actual computation.

To run the test, the program expects a topography profile for each grid. The directory `data`
contains topography profiles for a Gaussian hill geometry. By modifying the script `topopgraphy.py`
you can investigate the truncation errors for a different geometry.
