class Macro:

    def __init__(self, prefix, name, args, expr, guard=1):
        self.prefix = prefix
        self.name = name
        self.args = args
        self.expr = expr
        self.useguard = guard

    def define(self):
        from sympy.printing import ccode
        expr = self.guard()
        return "#define %s %s[%s]" % (self.str_base(), self.name, ccode(expr))

    def str_base(self):
        from sympy.printing import ccode
        if self.args:
            return "%s%s(%s)"% (self.prefix, self.name, 
                    ','.join(self.args))
        return "%s" % self.name

    def guard(self):
        """
        Encapsulate args in ( ) 

        Example:
            >>> args = [x, y]
            >>> value = 'x + y'
            >>> guard(value, args)
            '(x) + (y)'

        """
        import sympy
        if not self.useguard:
            return self.expr

        new_args = {}
        for argi in self.args:
            new_args[argi] = sympy.symbols('(%s)' % argi)
        return self.expr.subs(new_args)

    def undefine(self):
        return "#undef %s%s" % (self.prefix, self.name)

