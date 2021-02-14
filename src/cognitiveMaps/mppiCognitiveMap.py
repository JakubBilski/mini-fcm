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
        input_in_time = input_in_time[:-1]
        expected_output = np.array(expected_output)
        input_in_time = np.array(input_in_time)
        F_minus_Y = np.log(np.divide(expected_output,(1-expected_output)))
        self.weights = np.transpose(np.linalg.pinv(input_in_time).dot(F_minus_Y))

    def get_error(self, input_in_time):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        error = 0
        for i in range(len(input_in_time)-1):
            result = MppiCognitiveMap.f(self.weights.dot(input_in_time[i]))
            result -= expected_output[i]
            error += result.dot(np.transpose(result))
        return error/(len(input_in_time)*self.n)
