import numpy as np

from cognitiveMaps.baseCognitiveMap import BaseCognitiveMap
from scipy.optimize import differential_evolution


class DEShrinkedCognitiveMap(BaseCognitiveMap):
    def __init__(self, n):
        self.class_name = ""
        self.weights = np.random.rand(n, n)
        self.n = n
        self.conv_pnt = None

    def _minimized_function(x, *args):
        n = args[0]
        expected_input = args[1]
        expected_output = args[2]
        buff = DEShrinkedCognitiveMap.f((x.reshape(n, -1)).dot(expected_input)) - expected_output
        return np.mean(np.multiply(buff, buff))

    def train(self, input_in_time, maxiter=100):
        expected_output = input_in_time[1:]
        expected_input = input_in_time[:-1]
        expected_output = np.array(expected_output)[:, :-1].transpose()
        expected_input = np.array(expected_input).transpose()
        bounds = [(-1, 1) for _ in range(self.n*(self.n-1))]
        result = differential_evolution(
            DEShrinkedCognitiveMap._minimized_function,
            bounds,
            (self.n-1, expected_input, expected_output),
            maxiter=maxiter,
            seed=1)
        self.weights = result.x.reshape(self.n-1, -1)
        # print(self.weights)

    def get_error(self, input_in_time):
        expected_output = input_in_time[1:]
        expected_input = input_in_time[:-1]
        expected_output = np.array(expected_output)[:, :-1].transpose()
        expected_input = np.array(expected_input).transpose()
        error = DEShrinkedCognitiveMap.f(self.weights.dot(expected_input)) - expected_output
        error = np.mean(np.multiply(error, error))
        return error/(len(expected_input)*(self.n-1))

    def predict(self, input_in_time):
        expected_input = input_in_time[:-1]
        expected_input = np.array(expected_input).transpose()
        output = DEShrinkedCognitiveMap.f(self.weights.dot(expected_input)).transpose()
        output = np.append(output, np.sum(output, axis=0), axis=1)
        return output