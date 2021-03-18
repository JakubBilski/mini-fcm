import numpy as np

from cognitiveMaps.baseCognitiveMap import BaseCognitiveMap


class MppiCognitiveMap(BaseCognitiveMap):
    def __init__(self, n):
        self.class_name = ""
        self.weights = np.random.rand(n, n)
        self.n = n
        self.conv_pnt = None

    def train(self, input_in_time):
        expected_output = input_in_time[1:]
        expected_input = input_in_time[:-1]
        expected_output = np.array(expected_output)
        expected_input = np.array(expected_input)
        F_minus_Y = np.log(np.divide(expected_output,(1-expected_output)))
        self.weights = np.transpose(np.linalg.pinv(expected_input).dot(F_minus_Y))

    def get_error(self, input_in_time):
        expected_output = input_in_time[1:]
        expected_input = input_in_time[:-1]
        expected_output = np.array(expected_output).transpose()
        expected_input = np.array(expected_input).transpose()
        error = MppiCognitiveMap.f(self.weights.dot(expected_input)) - expected_output
        error = np.mean(np.multiply(error, error))
        return error/(len(expected_input)*self.n)
