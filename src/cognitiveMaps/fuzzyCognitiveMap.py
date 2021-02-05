import numpy as np

from cognitiveMaps.baseCognitiveMap import BaseCognitiveMap

from . import displaying


class FuzzyCognitiveMap(BaseCognitiveMap):
    def __init__(self, weights=None):
        self.class_name = ""
        self.weights = weights
        self.conv_pnt = None


    def train(self, input_in_time):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        expected_output = np.array(expected_output)
        input_in_time = np.array(input_in_time)
        F_minus_Y = -np.log(-expected_output+1.001)
        self.weights = np.linalg.pinv(input_in_time).dot(F_minus_Y)


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

