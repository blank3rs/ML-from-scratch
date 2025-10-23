import numpy as np


class autograd:
    def __init__(self, value, parents=(), op=''):
        self.value = np.array(value)
        self.parents = parents
        self.op = op
        self.grad = np.zeros_like(self.value)
        self._backward = lambda: None

    @staticmethod
    def add(a, b):
        out = autograd(a.value + b.value, (a, b), '+')
        out._backward = lambda: (
            setattr(a, 'grad', a.grad + 1.0 * out.grad),
            setattr(b, 'grad', b.grad + 1.0 * out.grad)
        )
        return out

    @staticmethod
    def sub(a, b):
        out = autograd(a.value - b.value, (a, b), '-')
        out._backward = lambda: (
            setattr(a, 'grad', a.grad + 1.0 * out.grad),
            setattr(b, 'grad', b.grad - 1.0 * out.grad)
        )
        return out

    @staticmethod
    def multiply(a, b):
        out = autograd(np.multiply(a.value, b.value)(a, b), '*')
        out._backward = lambda: (
            setattr(a, 'grad', a.grad + np.multiply(b.value, out.grad)),
            setattr(b, 'grad', b.grad + a.value * out.grad)
        )
        return out

    @staticmethod
    def div(a, b):
        out = autograd(a.value / b.value, (a, b), '/')
        out._backward = lambda: (
            setattr(a, 'grad', a.grad + (1 / b.value) * out.grad),
            setattr(b, 'grad', b.grad - (a.value / (b.value ** 2)) * out.grad)
        )
        return out

    @staticmethod
    def square(a):
        out = autograd(a.value * a.value, (a,), '**')
        out._backward = lambda: (
            setattr(a, 'grad', a.grad + 2 * a.value * out.grad),
        )
        return out

    @staticmethod
    def pow(a, exp):
        out = autograd(a.value ** exp, (a,), f'**{exp}')
        out._backward = lambda: (
            setattr(a, 'grad', a.grad + exp *
                    (a.value ** (exp - 1)) * out.grad),
        )
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v.parents:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd.add(self, other)

    def __sub__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd.sub(self, other)

    def __mul__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd.multiply(self, other)

    def __truediv__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd.div(self, other)

    def __pow__(self, power):
        return autograd.pow(self, power)

    def __repr__(self):
        return f"autograd(value={self.value}, grad={self.grad}, op='{self.op}')"

    def __neg__(self):
        return self * -1
