import numpy as np

from cognitiveMaps.baseCognitiveMap import BaseCognitiveMap


class FuzzyCognitiveMap(BaseCognitiveMap):
    def __init__(self, n):
        self.class_name = ""
        self.n = n
        # np.random.seed = 0
        self.weights = np.random.rand(n, n)
        self.conv_pnt = None

    def train_step(self, input_in_time, learning_rate):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        n = len(input_in_time[0])
        Pwprimes = np.zeros(shape=(n, n))

        for time in range(len(expected_output)):
            buff = self.weights.dot(input_in_time[time])
            buff = np.multiply(
                FuzzyCognitiveMap.f(buff)-expected_output[time],
                FuzzyCognitiveMap.fprim(buff)
                )
            buff = np.outer(buff, input_in_time[time])
            Pwprimes += buff
        self.weights += -learning_rate*Pwprimes

    def train(self, input_in_time):
        raise Exception("Not implemented, lazy programmer")

    def get_error(self, input_in_time):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        error = 0
        for i in range(len(input_in_time)-1):
            result = FuzzyCognitiveMap.f(self.weights.dot(input_in_time[i]))
            result -= expected_output[i]
            error += result.dot(np.transpose(result))
        return error/len(input_in_time)
