import numpy as np

SIGMOID_L = 1.0
SIGMOID_TAU = 5.0


class BaseCognitiveMap:
    def __init__(self, n):
        self.n = n

    def f(x):
        return SIGMOID_L / (1 + np.exp(-SIGMOID_TAU*x))

    def set_class(self, class_name):
        self.class_name = class_name

    def get_class(self):
        return self.class_name

    def train(self, input_in_time):
        pass

    def get_error(self, input_in_time):
        pass
