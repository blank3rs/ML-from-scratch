import numpy as np


class autograd:
    def __init__(self, value):
        self.value = np.array(value)
        self.parents = []
        self.grad = np.zeros_like(self.value)
        self.op = None

    def _accumulate_grad(self, grad, target):
        if grad.shape == target.shape:
            return target + grad
        while grad.shape != target.shape:
            if len(grad.shape) > len(target.shape):
                extra_dims = len(grad.shape) - len(target.shape)
                for _ in range(extra_dims):
                    grad = np.sum(grad, axis=0)
            elif len(grad.shape) == len(target.shape):
                for i in range(len(grad.shape)):
                    if grad.shape[i] != target.shape[i]:
                        grad = np.sum(grad, axis=i, keepdims=True)
                        if grad.shape == target.shape:
                            break
            else:
                break
        return target + grad.reshape(target.shape)

    def _backward(self):
        if self.op == 'add':
            self.parents[0].grad = self._accumulate_grad(self.grad, self.parents[0].grad)
            self.parents[1].grad = self._accumulate_grad(self.grad, self.parents[1].grad)
        elif self.op == 'sub':
            self.parents[0].grad = self._accumulate_grad(self.grad, self.parents[0].grad)
            self.parents[1].grad = self._accumulate_grad(-self.grad, self.parents[1].grad)
        elif self.op == 'mul':
            grad0 = self.parents[1].value * self.grad
            self.parents[0].grad = self._accumulate_grad(grad0, self.parents[0].grad)
            grad1 = self.parents[0].value * self.grad
            self.parents[1].grad = self._accumulate_grad(grad1, self.parents[1].grad)
        elif self.op == 'div':
            grad0 = (1 / self.parents[1].value) * self.grad
            self.parents[0].grad = self._accumulate_grad(grad0, self.parents[0].grad)
            grad1 = (self.parents[0].value / (self.parents[1].value ** 2)) * self.grad
            self.parents[1].grad = self._accumulate_grad(-grad1, self.parents[1].grad)
        elif self.op == 'pow':
            power = getattr(self, '_power', None)
            if power is not None:
                grad = power * (self.parents[0].value ** (power - 1)) * self.grad
                self.parents[0].grad = self._accumulate_grad(grad, self.parents[0].grad)
        elif self.op == 'matmul':
            self.parents[0].grad = self.parents[0].grad + np.dot(self.grad, self.parents[1].value.T)
            self.parents[1].grad = self.parents[1].grad + np.dot(self.parents[0].value.T, self.grad)

    def get_parents(self):
        if len(self.parents) == 0:
            return None
        else:
            return self.parents

    def set_parents(self, parent1, parent2):
        self.parents.extend([parent1, parent2])

    @staticmethod
    def check_for_mul(node1, node2):
        val1 = np.atleast_2d(node1.value)
        val2 = np.atleast_2d(node2.value)

        if val1.shape[1] == val2.shape[0]:
            return True
        elif val1.shape[1] == val2.T.shape[0]:
            return True
        else:
            raise ValueError("matrix's k value dont match")

    def __add__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        out = autograd(self.value + other.value)
        out.parents = [self, other]
        out.op = 'add'
        return out

    def __sub__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        out = autograd(self.value - other.value)
        out.parents = [self, other]
        out.op = 'sub'
        return out

    def __mul__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        out = autograd(self.value * other.value)
        out.parents = [self, other]
        out.op = 'mul'
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        out = autograd(self.value / other.value)
        out.parents = [self, other]
        out.op = 'div'
        return out

    def __pow__(self, power):
        out = autograd(self.value ** power)
        out.parents = [self]
        out.op = 'pow'
        out._power = power
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        autograd.check_for_mul(self, other)
        out = autograd(np.dot(self.value, other.value))
        out.parents = [self, other]
        out.op = 'matmul'
        return out

    def __repr__(self):
        return f"autograd(value={self.value}, grad={self.grad})"

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return autograd(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return autograd(other) / self

    def __neg__(self):
        return self * -1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.value)
        for node in reversed(topo):
            node._backward()
