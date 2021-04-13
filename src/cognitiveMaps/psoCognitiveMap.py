import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cognitiveMaps.baseCognitiveMap import BaseCognitiveMap
import pyswarms as ps


class PSOCognitiveMap(BaseCognitiveMap):
    def __init__(self, n):
        self.class_name = ""
        self.weights = np.random.rand(n, n)
        self.n = n
        self.conv_pnt = None

    def _minimized_function(xs, n, expected_input, expected_output):
        #todo: optimize this
        results = []
        for i in range(xs.shape[0]):
            buff = PSOCognitiveMap.f((xs[i].reshape(n, -1)).dot(expected_input)) - expected_output
            results.append(np.mean(np.multiply(buff, buff)))
        return results


    def train(self, inputs_in_time, max_iter=500):
        expected_input = []
        expected_output = []
        for input_in_time in inputs_in_time:
            expected_output.extend(input_in_time[1:])
            expected_input.extend(input_in_time[:-1])
        expected_output = np.array(expected_output).transpose()
        expected_input = np.array(expected_input).transpose()
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        x_max = np.ones(self.n*self.n)
        x_min = -1 * x_max
        bounds = (x_min, x_max)

        # Call instance of PSO
        optimizer = ps.single.GlobalBestPSO(
            n_particles=10*self.n*self.n,
            dimensions=self.n*self.n,
            options=options,
            bounds=bounds)

        # Perform optimization
        cost, pos = optimizer.optimize(
            PSOCognitiveMap._minimized_function,
            iters=max_iter,
            n=self.n,
            expected_input=expected_input,
            expected_output=expected_output)

        self.weights = pos.reshape(self.n, -1)

    def get_error(self, input_in_time):
        expected_output = input_in_time[1:]
        expected_input = input_in_time[:-1]
        expected_output = np.array(expected_output).transpose()
        expected_input = np.array(expected_input).transpose()
        error = PSOCognitiveMap.f(self.weights.dot(expected_input)) - expected_output
        return np.mean(np.multiply(error, error))

    def predict(self, input_in_time):
        expected_input = input_in_time[:-1]
        expected_input = np.array(expected_input).transpose()
        return PSOCognitiveMap.f(self.weights.dot(expected_input)).transpose()