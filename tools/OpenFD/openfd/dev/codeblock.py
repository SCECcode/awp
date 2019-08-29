class CodeBlock(object):

    def __init__(self, code):
        self.code = code
        self.args = []

    def __str__(self):
        return str(self.code)

    def _sympystr(self):
        return str(self.code)

    def _ccode(self, p):
        return str(self.code)

    def __getitem__(self, idx):
        return self
