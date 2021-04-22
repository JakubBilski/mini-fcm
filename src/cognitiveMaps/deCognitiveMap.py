import numpy as np

from cognitiveMaps.baseCognitiveMap import BaseCognitiveMap
from scipy.optimize import differential_evolution


class DECognitiveMap(BaseCognitiveMap):
    def __init__(self, n):
        self.n = n

    def _minimized_function(x, *args):
        n = args[0]
        expected_input = args[1]
        expected_output = args[2]
        buff = DECognitiveMap.f((x.reshape(n, -1)).dot(expected_input)) - expected_output
        return np.sum(np.multiply(buff, buff))

    def train(self, inputs_in_time, max_iter, mutation, recombination, popsize):
        expected_input = []
        expected_output = []
        for input_in_time in inputs_in_time:
            expected_output.extend(input_in_time[1:])
            expected_input.extend(input_in_time[:-1])
        expected_output = np.array(expected_output).transpose()
        expected_input = np.array(expected_input).transpose()
        bounds = [(-1, 1) for _ in range(self.n*self.n)]
        result = differential_evolution(
            DECognitiveMap._minimized_function,
            bounds,
            (self.n, expected_input, expected_output),
            maxiter=max_iter,
            strategy='rand1bin',
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            init='random',
            seed=1)
        self.weights = result.x.reshape(self.n, -1)
        return result.nit

    def get_error(self, input_in_time):
        expected_output = input_in_time[1:]
        expected_input = input_in_time[:-1]
        expected_output = np.array(expected_output).transpose()
        expected_input = np.array(expected_input).transpose()
        error = DECognitiveMap.f(self.weights.dot(expected_input)) - expected_output
        return np.mean(np.multiply(error, error))

    def predict(self, input_in_time):
        expected_input = input_in_time[:-1]
        expected_input = np.array(expected_input).transpose()
        return DECognitiveMap.f(self.weights.dot(expected_input)).transpose()
