import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.merge import concatenate
import numpy as np

from . import weightsGeneration
from . import displaying



def custom_activation(x, li, ui, lambdai, hi):
    return li + (ui - li) / (1 + keras.backend.exp(-lambdai * (x - hi)))


def create_fcm_model(input_size):
    visible = Input(shape=(input_size,))
    neurons = []
    for i in range(input_size):
        neurons.append(Dense(1, activation=(lambda x : custom_activation(x, 0.0, 1.0, 1.0, 0.0)))(visible))
    layer = concatenate(neurons)
    model = Model(inputs=visible, outputs=layer)
    return model


class FuzzyCognitiveMap:
    def __init__(self, input_size, weights=None):
        self.class_name = ""
        self.model = create_fcm_model(input_size)
        if weights:
            weightsGeneration.set_w1_weights(self.model, weights)
    
    def infere(self, input_data, max_iterations):
        output = input_data
        for i in range(max_iterations):
            buffer = self.model.predict(output)
            if (buffer == output).all():
                print(f"fixed-point attractor found after {i} steps")
                break
            # print(output[1])
            output, buffer = buffer, output
        return output

    def step(self, input_data):
        return self.model.predict(input_data)

    def train(self, input_in_time):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        expected_output = np.array(expected_output)
        input_in_time = np.array(input_in_time)
        F_minus_Y = -np.log(-expected_output+1.001)
        trained_weights = np.linalg.pinv(input_in_time).dot(F_minus_Y)
        self.weights = trained_weights
        weightsGeneration.set_w1_weights(self.model, trained_weights)

    def display_plot(self, save_path=None):
        displaying.draw_cognitive_map(self.weights, self.class_name, save_path)

    def set_class(self, class_name):
        self.class_name = class_name


    def get_class(self):
        return self.class_name