"""
Module that contains implementations of Runge-Kutta time stepping methods.

"""

class LSRK(object):
    """ 
    Base class for 2N Low storage Runge-Kutta methods that is used for time
    integration of ODEs:

        du/dt = f(u, t),

    where `f` is some known function.

    To advance from time `t` to `t + dt`, loop over the `nstages` and at each
    stage set the rates to

        du = ak*du + f(u, t+c(k)*dt)

    and then proceed by updating the solution to

        u = u + dt*bk*du

    The class methods `rates` and `update` take a symbolic expression as input
    and return the RK update.
    For example,

    ```

    Attributes:
        nstages : Number of RK stages
        a : RK coefficients used for computing rates (array of length
            `nstages`). 
        b : RK coefficients used for updating the solution (array of length
            `nstages`). 
        c : Runge-Kutta coefficients used for evaluating forcing functions at
            the stage time `t + c[s]` (array of length `nstages).

        debug : When debugging is enabled, the rate computation is modifed so
            that the 

    References:
        M.H. Carpenter and C.A. Kennedy. 
        Fourth-order 2N-storage Runge-Kutta schemes. 
        TechnicalReport NASA TM-109112,


    """

    def __init__(self, prec=None, dt=None, debug=False):
        """
        Initialize low storage Runge-Kutta time integrator

        Arguments:
            prec(optional) : Precision to use. Pass `np.float32` for single
                (default) or `np.float64` for double precision`.

        """
        from sympy import symbols
        import numpy as np
        self.debug = debug
        self.ak = symbols('ak')
        self.bk = symbols('bk')
        
        self.ck = symbols('ck')

        if not prec:
            self.prec = np.float32
        else:
            self.prec = prec

        if not dt:
            self.dt = symbols('dt')

        self.init()

    def init(self):
        """
        Function to call at initialization. 
        Override this function to perform additional initializations operators
        in child classes
        """
        pass

    def rates(self, lhs, rhs, append=False):
        """
        Rate computation for Low-storage Runge-Kutta 4. 

        Set rates to
            lhs = append*ak*lhs + rhs(y, t+c(k)*dt)

        Arguments:
            lhs: list of left-hand side gridfunctions (rates)
            rhs: list of right-hand side expressions
            append : Append to rate if `True`. Default to `False`.
                    Use `append = True` when multiple kernels assign to the same
                    rate. 

        Returns:
            Updated rhs

        """
        from openfd import GridFunctionExpression as Expr, utils

        # Disable rate transformation when running in debug mode
        if self.debug:
            #FIXME: `a` has to be included to not alter the function argument
            # list. A work-around is to make it so small that it doesn't matter
            # in practice.
            eps = 1e-6
            formula = lambda a, l, r: Expr(r + eps*a)
        else:
            formula = lambda a, l, r:  Expr(a*l + r)

        lhs = utils.to_tuple(lhs)
        rhs = utils.to_tuple(rhs)

        expr = []
        for l, r in zip(lhs, rhs):
            if not append:
                expr.append(formula(self.ak, l, r))
            else:
                expr.append(formula(1, l, r))

        return utils.to_atom(lhs), utils.to_atom(expr)

    def update(self, lhs, rhs):
        """
        Solution update for low-storage Runge-Kutta 4

        Update is computed as 
        lhs = lhs + dt*b(k)*rhs

        Arguments:
            lhs : Solution to update
           dlhs : Rate of solution (use rk4 to compute)

        """
        from openfd import GridFunctionExpression as Expr, utils
        from sympy import symbols

        if self.debug:
            #FIXME: `a` has to be included to not alter the function argument
            # list. A work-around is to make it so small that it doesn't matter
            # in practice.
            eps = 1e-6
            formula = lambda b, l, r: Expr(r + eps*b)
        else:
            formula = lambda b, l, r:  Expr(l + b*r)

        lhs = utils.to_tuple(lhs)
        rhs = utils.to_tuple(rhs)

        expr = []
        for l, r in zip(lhs, rhs):
            expr.append(formula(self.dt*self.bk, l, r))

        return utils.to_atom(lhs), utils.to_atom(expr)

class LSRK4(LSRK):
    import numpy as np
    """
    Low storage fourth order Runge-Kutta (RK) implementation (5-4 solution 3)
    
    """

    def init(self):
        import numpy as np
        self._a = np.array([0,
                      -567301805773.0/1357537059087,
                      -2404267990393.0/2016746695238,
                      -3550918686646.0/2091501179385,
                      -1275806237668.0/842570457699]).astype(self.prec)
        self._b = np.array([1432997174477.0/9575080441755,
                      5161836677717.0/13612068292357,
                      1720146321549.0/2090206949498,
                      3134564353537.0/4481467310338,
                      2277821191437.0/14882151754819]).astype(self.prec)
        self._c = np.array([0,
                      1432997174477.0/9575080441755,
                      2526269341429.0/6820363962896,
                      2006345519317.0/3224310063776,
                      2802321613138.0/2924317926251]).astype(self.prec)
        self.nstages = 5

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c
