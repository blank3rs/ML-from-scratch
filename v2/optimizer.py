
import numpy as np


class optimizer:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate

    def adjust(self, param):
        param.value = param.value - (self.learning_rate * param.grad)
        param.grad = np.zeros_like(param.grad)
