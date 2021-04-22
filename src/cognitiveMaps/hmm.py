from hmmlearn.hmm import GaussianHMM
import numpy as np


class HMM:
    def __init__(self, n):
        self.n = n

    def train(self, inputs_in_time, max_iter, no_random_initializations, covariance_type):
        concatenated_inputs = np.concatenate(inputs_in_time)
        lengths = [len(x) for x in inputs_in_time]
        models_with_scores = []

        for i in range(no_random_initializations):
            new_model = GaussianHMM(n_components=self.n, covariance_type=covariance_type, n_iter=max_iter, random_state=i)
            new_model.fit(concatenated_inputs, lengths)
            score = 0
            for x in inputs_in_time:
                score += new_model.score(x)
            models_with_scores.append((new_model, score))

        self.model = max(models_with_scores, key=lambda ms: ms[1])[0]
        return self.model.monitor_.iter

    def get_emission_probability(self, input_in_time):
        return self.model.score(input_in_time)

    def set_class(self, class_name):
        self.class_name = class_name

    def get_class(self):
        return self.class_name

    def get_gauss_means(self):
        return self.model.means_.tolist()

    def get_gauss_covars(self):
        return self.model.covars_.tolist()
