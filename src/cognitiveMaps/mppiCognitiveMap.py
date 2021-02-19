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
        error = 0
        for i in range(len(expected_input)-1):
            result = MppiCognitiveMap.f(self.weights.dot(expected_input[i]))
            result -= expected_output[i]
            error += result.dot(np.transpose(result))
        return error/(len(expected_input)*self.n)
