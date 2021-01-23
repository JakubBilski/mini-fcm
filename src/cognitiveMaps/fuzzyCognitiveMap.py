import numpy as np

from . import displaying


class FuzzyCognitiveMap:
    def __init__(self, weights=None):
        self.class_name = ""
        self.weights = weights
        self.conv_pnt = None

    def get_convergence_point(self, input_data, max_iterations=100):
        if self.conv_pnt is None:
            self._calculate_convergence_pnt(input_data, max_iterations)
        return self.conv_pnt

    def f(x):
        return 1 / (1 + np.exp(-x))

    def fprim(x):
        pom = FuzzyCognitiveMap.f(x)
        return pom*(1-pom)
    
    def _calculate_convergence_pnt(self, input_data, max_iterations):
        output = input_data.transpose()
        weights = np.matrix(self.weights)
        for i in range(max_iterations):
            buffer = FuzzyCognitiveMap.f(weights @ output)
            if (buffer == output).all():
                # print(f"fixed-point attractor found after {i} steps")
                break
            # print(output[1])
            output, buffer = buffer, output
        self.conv_pnt = output.transpose()

    def train(self, input_in_time):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        expected_output = np.array(expected_output)
        input_in_time = np.array(input_in_time)
        F_minus_Y = -np.log(-expected_output+1.001)
        self.weights = np.linalg.pinv(input_in_time).dot(F_minus_Y)
        # print("trained weights")
        # for vector in self.weights:
        #     print(vector)
        # weightsGeneration.set_w1_weights(self.model, trained_weights)

    def get_error(self, input_in_time):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        expected_output = np.array(expected_output)
        input_in_time = np.array(input_in_time)
        error = 0
        for i in range(len(input_in_time)-1):
            weights = np.matrix(self.weights)
            result = FuzzyCognitiveMap.f(weights.dot(input_in_time[i]))
            result -= expected_output[i]
            error += result.dot(np.transpose(result))
        return error

    def display_plot(self, save_path=None):
        displaying.draw_cognitive_map(self.weights, self.class_name, save_path)

    def set_class(self, class_name):
        self.class_name = class_name

    def get_class(self):
        return self.class_name
