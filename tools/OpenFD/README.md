![Logo](docs/openfd-logo.png)

# Unstable pre-release
This code is unstable and not all tests pass. Use with caution... Currently, 15
tests fail. Run tests by going to `openfd/` and then type `pytest`.

# Outdated stuff below.
# OpenFD
OpenFD lets you generate matrix-free compute kernels using computational symbolic algebra.  The goal of the project is
to let you implement any numerical method with a tensor-like structure (e.g., finite difference methods) and apply to your unique
problem without much effort. 

# Basic demonstration #
Head over to [docs](http://ooreilly.github.io/openfd) (the docs have not gone live yet!) where you can find a much nicer presentation (because math is
properly rendered) as well as tutorials and examples.

Suppose you want to numerically compute the derivative of a function ``u(x)``.  The finite difference solution
is to discretize ```u(x)``` on a grid so that we obtain the discrete approximation ```v[0] ... v[n]```, where ```n+1``` is
the number of grid points that are (for simplicity) equidistantly spaced by ```h = 1/n```. Then derivative at a grid
point ```j``` is obtained by applying a [finite difference formula](https://en.wikipedia.org/wiki/Finite_difference_coefficient). For example, second order accuracy using a central approximation of the first derivative reads
```
v_x[j] = (v[j+1] - v[j-1])/(2*h)
```
That looks pretty straightforward to implement, so what's the problem? 
* The formula above only works for grid points ```j > 0 and j < n```. At the boundary, we need to do something else. 
* Practical problems are of course far more complicated than what is demonstrated in this example. 
* The numerical grid (given by the discretization) is unlikely to be the most optimal layout for computational
  efficiency

Our philosophy is that numerical discretization can never be simpler than the continuous problem, and therefore we
should strive to write our "code" in way that resembles the continuous as much as possible.  At the end of the day the
resulting code must solve the problem, and be efficient, maintainable, and scalable. We accomplish this by using code
generation. 

## 1D example ##
As a slightly more advanced example, suppose we have an expression of the form  ```e = a[i] + b[i] + 2``` and we wish to transform this expression
by applying a difference approximation to it. The first thing is to define symbolic vectors, or so-called
**grid functions**. The grid functions support indexing via a[0], a[1], and a[j], where j is symbolic. Suppose our grid
uses ```nx + 1``` grid points, then we initialize the index ```j``` and ```nx``` as standard **Symbolic** objects in
SymPy. 
```python
>>> from sympy import symbols
>>> nx,j = symbols('nx j')
```
Next it is time to construct the grid functions and build our expression,
```python
>>> from openfd import GridFunction
>>> a  = GridFunction("a",shape=(nx+1,))
>>> b  = GridFunction("b",shape=(nx+1,))
>>> e  = a + b
```
Finally, we transform this expression by numerically computing its first derivative. In this example, we use a
predefined derivative operator, which is a SBP (summation-by-parts) difference operator. What is special about this
operator is that it is modified near the boundaries enabling one to construct a provably stable method. 
```python
>>> from openfd import sbp_standard as sbp
>>> e_x = sbp.Derivative(e, "x", order=2)
```

In the above, the property ```order``` makes the SBP operator is second-order-accurate in the interior. To obtain the
result of the derivative, we simply access the part we are interested in
```python
>>> e_x[j]
-(a[j - 1] + b[j - 1])/2 + (a[j + 1] + b[j + 1])/2
```
That is, the variable ```e_x``` is a new GridFunction determined via lazy evaluation as soon as any of its elements are
accessed. Note that produced output this is an undivided difference (i.e. not scaled by the grid spacing). It is also possible to assign a grid
spacing to the computation by using ```Derivative(e, "x", order=2, gridspacing=1/nx)``` instead. 

As previously mentioned, the SBP operator uses different stencils for the boundary and interior. To obtain the left
boundary stencil, we simply access the first element of the grid function
```python
>>> e_x[0]
-(a[0] + b[0]) + a[1] + b[1]
```
Since this SBP operator only has one boundary stencil, we again switch to the interior stencil when accessing the
next element:
```python
>>> e_x[1]
-(a[0] + b[0])/2 + (a[2] + b[2])/2
```
We find the right boundary by accessing the last index ```e_x[n]```. Alternately, we can use ```e_x.right(0)```. Note
that ```e_x[n+1]``` throws an exception because the accessed index is out of bounds. 

## 2D example ##
The extension to 2D or 3D or is fairly straightforward. All that is needed is to change the shape of the grid functions. 
```python
    >>> a  = GridFunction("a", shape=(nx+1, ny+1))
```

Although an operator (like the SBP operator mentioned in the previous example) is restricted to only act in one direction, multiple directions are supported by nesting
operators. This enables one to construct quite complex expressions. For example, we can use this technique to
construct a mixed derivative

```python
>>> nx,ny,i,j = symbols('nx ny i j')
>>> a  = GridFunction("a", shape=(nx+1, ny+1))
>>> a_x  = sbp.Derivative(a, "x", order=2)
>>> a_xy = sbp.Derivative(a_x, "y", order=2)
>>> a_xy[i,j]
-(-a[i - 1, j - 1]/2 + a[i + 1, j - 1]/2)/2 + (-a[i - 1, j + 1]/2 + a[i + 1, j + 1]/2)/2
```

## The wave equation in 1D ##
Let's kick it up a notch by solving the 1D wave equation on a staggered grid using fourth order accuracy.  
Some step by step comments for the various parts of the code are also listed below.

The following example can be found [here](examples/wave1dstaggered.py).
```python
from sympy import symbols
from openfd import GridFunction, sbp_staggered as sbp, GridFunctionExpression as GFE, kernel as kl
from sympy.core.cache import clear_cache

order        = 4
i, n, hi, dt = symbols('i n hi dt')
rhoi, K      = symbols('rhoi K')

v            = GridFunction("v", shape=(n+1, ))
p            = GridFunction("p", shape=(n+2, ))

v_x          = sbp.Derivative(v, "x", order=order, shape=(n+2,), gridspacing=1/hi, hat = True)
p_x          = sbp.Derivative(p, "x", order=order, shape=(n+1,), gridspacing=1/hi, hat = False)

v_t          = GFE(-rhoi*p_x  ) 
p_t          = GFE(-K*v_x  ) 

rhsv         = GFE(v + dt*v_t)
rhsp         = GFE(p + dt*p_t)

# Velocity update
# 0: left, -1: right, 1: interior
vkl = ''
for i in [0, 1, -1]:
    vkl += kl.kernel1d(p_x.bounds(), i, v, rhsv) 

# Pressure update
pkl = ''
for i in [0, 1, -1]:
    pkl += kl.kernel1d(v_x.bounds(), i, p, rhsp) 

# Output to C
code = ''
code += kl.ckernel("update_velocity", n, v, rhsv[0], vkl)
code += kl.ckernel("update_pressure", n, p, rhsp[0], pkl)
print code
```

### Setup
First we define the symbols that we need to use such as the index ```i```, grid size ```n``` and material properties
```rhoi``` (reciprocal of density) and bulk modulus ```K```
```python
order = 4
i, n, hi, dt = symbols('i n hi dt')
rhoi, K      = symbols('rhoi K')
```
For simplicity, we have defined the material properties as being constant coefficients. 

We define pressure ``p`` and velocity ``v`` as grid functions (meaning that they are symbolic vectors that
can be accessed like ``p[j]`` etc. The fields are staggered and therefore stored at different locations of the grid.
There is no true way to encode this information since the fields do not know of any grid. The ``shape`` argument
specifies the number of grid points used for each grid function. Since we are using ``n+1`` grid points for ``v`` we use the
convention that this field is stored on a grid ``x = [0, 1, .., n]*h`` whereas the field ``p`` is stored on a grid
``xhat = [0, 1/2, 3/2, ... n-1/2, n]*h``. Note that we include the end points ``x = 0 and x = n*h`` in both grids.
This is because we can then use so-called **weak** boundary conditions.
```python  
v = GridFunction("v", shape=(n+1, ))
p = GridFunction("p", shape=(n+2, ))
```
We differentiate pressure and velocity using staggered difference operators. Since ``p`` and ``v`` are staggered
(stored on different grids), the difference operator needs to be compatible with the grid. The ``hat`` variable says
that a special difference operator that computes the derivative on the grid ``x`` and stores the result on
the grid ``xhat``. We also use so-called ``SBP`` (summation-by-parts) difference operators that enable us to get
a provably stable method.

```python
v_x = sbp.Derivative(v, "x", order=order, shape=(n+2,), gridspacing=1/hi, hat = True)
p_x = sbp.Derivative(p, "x", order=order, shape=(n+1,), gridspacing=1/hi, hat = False)
```

Once we are done computing all of the derivatives, we can go ahead and discretize the governing equations. To enable
symbolic manipulations of the gridfunctions it is important to wrap our equations using ```GridFunctionExpression```
(here denoted by ``GFE``).
```python
v_t = GFE(-rhoi*p_x  ) 
p_t = GFE(-K*v_x  ) 
```

We do a simple time stepping computation using leapfrog and lambda functions. The value ``rhsv`` will be used to overwrite
``v`` at each iteration and similarly for ``rhsp`` and ``p``.

```python
rhsv         = GFE(v + dt*v_t)
rhsp         = GFE(p + dt*p_t)
```
### Code generation
After discretizing the PDE, all that remains is to generate code. Since the pressure update equation ``rhsp`` depends
on the velocity computed be evaluating ``rhsv``, we build one compute kernel for each update equation. However, it is
possible to pack multiple equations into the same kernel as long as there are no dependencies. The kernel1d function
is used to generate the code for each different region of the grid. As shown below, the index ``i`` in the loop
instructs the kernel creation function to generate code for the left, right, and interior parts of the grid.

```python
# Velocity update
# 0: left, -1: right, 1: interior
vkl = ''
for i in [0, 1, -1]:
    vkl += kl.kernel1d(p_x.bounds(), i, v, rhsv) 

# Pressure update
pkl = ''
for i in [0, 1, -1]:
    pkl += kl.kernel1d(v_x.bounds(), i, p, rhsp) 
```

* The last part of the example generates a C function header by calling the function ``ckernel``. This function is
  capable of detecting what types of parameters should be passed as input to the generated C function.
```python
code = ''
code += kl.ckernel("update_velocity", n, v, rhsv, vkl)
code += kl.ckernel("update_pressure", n, p, rhsp, pkl)
```
* The following output is produced when running the example:
```C
 void update_velocity(float *v, const float *p, const float dt, const float hi, const float rhoi, const int n) {
         v[0] = -dt*hi*rhoi*(-2.39594408750282*p[0] + 2.49239516406779*p[1] + 0.00506989062147429*p[2] - 0.101520967186442*p[3]) + v[0];
         v[1] = -dt*hi*rhoi*(-0.13884643234928*p[0] - 0.739662939345099*p[1] + 0.8264419595634*p[2] + 0.0520674121309802*p[3]) + v[1];
         v[2] = -dt*hi*rhoi*(-0.019204811087795*p[0] + 0.0832763767718904*p[1] - 1.16580808180657*p[2] + 1.14900387210475*p[3] - 0.0472673559822747*p[4]) + v[2];
         v[3] = -dt*hi*rhoi*(0.0260410454675288*p[0] - 0.0238586995660473*p[1] - 0.00172715275086196*p[2] - 1.05673957740792*p[3] + 1.09691070672873*p[4] - 0.0406263224714346*p[5]) + v[3];
         for (int i = 4; i < n - 3; ++i) {
                 v[i] = -dt*hi*rhoi*(-1.125*p[i] + 0.0416666666666667*p[i - 1] + 1.125*p[i + 1] - 0.0416666666666667*p[i + 2]) + v[i];
         }
         v[n - 3] = -dt*hi*rhoi*(0.0238586995660473*p[n] + 0.0406263224714346*p[n - 4] - 1.09691070672873*p[n - 3] + 1.05673957740792*p[n - 2] + 0.00172715275086196*p[n - 1] - 0.0260410454675288*p[n + 1]) + v[n - 3];
         v[n - 2] = -dt*hi*rhoi*(-0.0832763767718904*p[n] + 0.0472673559822747*p[n - 3] - 1.14900387210475*p[n - 2] + 1.16580808180657*p[n - 1] + 0.019204811087795*p[n + 1]) + v[n - 2];
         v[n - 1] = -dt*hi*rhoi*(0.739662939345099*p[n] - 0.0520674121309802*p[n - 2] - 0.8264419595634*p[n - 1] + 0.13884643234928*p[n + 1]) + v[n - 1];
         v[n] = -dt*hi*rhoi*(-2.49239516406779*p[n] + 0.101520967186442*p[n - 2] - 0.00506989062147429*p[n - 1] + 2.39594408750282*p[n + 1]) + v[n];
         
}

void update_pressure(float *p, const float *v, const float K, const float dt, const float hi, const int n) {
         p[0] = -K*dt*hi*(-1.28868478557771*v[0] + 1.36605435673313*v[1] + 0.133945643266867*v[2] - 0.211315214422289*v[3]) + p[0];
         p[1] = -K*dt*hi*(-1.02891184978183*v[0] + 1.08673554934549*v[1] - 0.086735549345492*v[2] + 0.0289118497818307*v[3]) + p[1];
         p[2] = -K*dt*hi*(-0.00171481594549866*v[0] - 0.994855552163504*v[1] + 0.994855552163504*v[2] + 0.00171481594549866*v[3]) + p[2];
         p[3] = -K*dt*hi*(0.0356750386699345*v[0] - 0.0651183585451624*v[1] - 1.01869515638412*v[2] + 1.09004523372399*v[3] - 0.0419067574646412*v[4]) + p[3];
         for (int i = 4; i < n - 2; ++i) {
                 p[i] = -K*dt*hi*(1.125*v[i] + 0.0416666666666667*v[i - 2] - 1.125*v[i - 1] - 0.0416666666666667*v[i + 1]) + p[i];
         }
         p[n - 2] = -K*dt*hi*(-0.0356750386699345*v[n] + 0.0419067574646412*v[n - 4] - 1.09004523372399*v[n - 3] + 1.01869515638412*v[n - 2] + 0.0651183585451624*v[n - 1]) + p[n - 2];
         p[n - 1] = -K*dt*hi*(0.00171481594549866*v[n] - 0.00171481594549866*v[n - 3] - 0.994855552163504*v[n - 2] + 0.994855552163504*v[n - 1]) + p[n - 1];
         p[n] = -K*dt*hi*(1.02891184978183*v[n] - 0.0289118497818307*v[n - 3] + 0.086735549345492*v[n - 2] - 1.08673554934549*v[n - 1]) + p[n];
         p[n + 1] = -K*dt*hi*(1.28868478557771*v[n] + 0.211315214422289*v[n - 3] - 0.133945643266867*v[n - 2] - 1.36605435673313*v[n - 1]) + p[n + 1];
         
}
```

## Test-based learning
Another great way to learn how to use this package is to study some of the test cases. 
See ```openfd/base/tests/``` or ```openfd/sbp/tests``` depending on which subpackage you would like to master.

# Setup #
## Dependencies ## 
To install this Python package you need:
* [NumPy](http://numpy.org) 
* [SymPy](http://sympy.org)
* [pytest](http://pytest.org) (only for testing)

## Installation ##
After cloning the repository and navigating to the root directory, type
```bash
>>>[sudo] python setup.py [options]
```
The ```options``` available are:
* install 
* develop
* clean

Use the option ```install``` if you don't plan to make any changes to the source. 
If you want to help out with project development, or modify the source, then use ```develop``` instead. Otherwise, you will need to
reinstall the package for each change you make. 

The ```clean``` option removes any compiled .pyc files in the repository (sometimes this is helpful because installation will fail if
they are present)

## Testing ##
Make sure that you have installed [pytest](http://pytest.org). With pytest installed, navigate to the root directory and
run
```bash
>>>pytest
```
