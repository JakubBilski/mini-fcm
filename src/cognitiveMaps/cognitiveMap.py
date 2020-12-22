import numpy as np

from . import weightsGeneration
from . import displaying


class FuzzyCognitiveMap:
    def __init__(self, weights=None):
        self.class_name = ""
        self.weights = weights
    
    def infere(self, input_data, max_iterations):
        output = input_data.transpose()
        weights = np.matrix(self.weights)
        for i in range(max_iterations):
            buffer = 1 / (1 + np.exp(-weights @ output))
            if (buffer == output).all():
                print(f"fixed-point attractor found after {i} steps")
                break
            # print(output[1])
            output, buffer = buffer, output
        return output.transpose()

    def step(self, input_data):
        return self.model.predict(input_data)

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

    def display_plot(self, save_path=None):
        displaying.draw_cognitive_map(self.weights, self.class_name, save_path)

    def set_class(self, class_name):
        self.class_name = class_name


    def get_class(self):
        return self.class_name