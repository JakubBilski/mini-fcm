import numpy as np

from . import displaying

SIGMOID_L = 1.0
SIGMOID_TAU = 5.0


class BaseCognitiveMap:
    def __init__(self, weights=None):
        self.class_name = ""
        self.weights = weights
        self.conv_pnt = None

    def f(x):
        return SIGMOID_L / (1 + np.exp(-SIGMOID_TAU*x))

    def fprim(x):
        pom = BaseCognitiveMap.f(x)
        return SIGMOID_TAU*pom*(SIGMOID_L-pom)

    def get_convergence_point(self, input_data, max_iterations=100):
        if self.conv_pnt is None:
            self._calculate_convergence_pnt(input_data, max_iterations)
        return self.conv_pnt

    def _calculate_convergence_pnt(self, input_data, max_iterations):
        output = input_data
        for i in range(max_iterations):
            buffer = BaseCognitiveMap.f(self.weights.dot(output))
            if (buffer == output).all():
                # print(f"fixed-point attractor found after {i} steps")
                break
            # print(output[1])
            output, buffer = buffer, output
        self.conv_pnt = output

    def display_plot(self, save_path=None):
        displaying.draw_cognitive_map(self.weights, self.class_name, save_path)

    def set_class(self, class_name):
        self.class_name = class_name

    def get_class(self):
        return self.class_name

    def train(self, input_in_time):
        pass

    def get_error(self, input_in_time):
        pass
