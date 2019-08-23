from . variable import Variable
class Equation(object):

    def __init__(self, lhs, rhs=None):
        if type(lhs) == Variable and lhs.declarable and not lhs.isdeclared: 
                self.lhs = lhs.declare()
                self.rhs = lhs.val
        else:
            self.lhs = lhs
            self.rhs = rhs


def equations(eqs):
    lhs = []
    rhs = []
    for eq in eqs:
        if type(eq) == tuple:
            eq = Equation(eq[0], eq[1])
        elif not type(eq) == Equation:
            eq = Equation(eq)
        lhs.append(eq.lhs)
        rhs.append(eq.rhs)
    return lhs, rhs

