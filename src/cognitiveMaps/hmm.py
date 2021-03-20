from hmmlearn.hmm import GaussianHMM
import numpy as np
import warnings

class HMM:
    def __init__(self, n):
        self.class_name = ""
        self.n = n

    def train(self, inputs_in_time, no_random_initializations):
        concatenated_inputs = np.concatenate(inputs_in_time)
        lengths = [len(x) for x in inputs_in_time]
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        models_with_scores = []
        for i in range(no_random_initializations):
            new_model = GaussianHMM(n_components=self.n, covariance_type="diag", n_iter=1000, init_params="")
            random_transmat = np.random.rand(self.n, self.n)
            random_transmat = random_transmat / random_transmat.sum(1, keepdims=True)
            new_model.transmat_ = random_transmat
            new_model.means_ = np.random.rand(self.n, len(concatenated_inputs[0]))
            random_startprob = np.random.rand(self.n)
            random_startprob = random_startprob / random_startprob.sum(0, keepdims=True)
            new_model.startprob_ = random_startprob
            random_covars = np.array([
                np.random.rand(len(concatenated_inputs[0])) for _ in range(self.n)
            ])
            new_model.covars_ = random_covars
            new_model.fit(concatenated_inputs, lengths)
            # sometimes, despite converging, the model will be invalid
            # and it will hopefully raise an error during score()
            if new_model.monitor_.converged:
                try:
                    score = 0
                    for x in inputs_in_time:
                        score += new_model.score(x)
                    models_with_scores.append((new_model, score))
                except:
                    pass
        # for m, s in models_with_scores:
        #     print(s)
        # print()
        warnings.resetwarnings()
        if len(models_with_scores) == 0:
            raise Exception("Unable to learn a valid model")
        # print(sorted([s for m, s in models_with_scores]))
        self.model = max(models_with_scores, key=lambda ms: ms[1])[0]

    def get_emission_probability(self, input_in_time):
        return self.model.score(input_in_time)

    def set_class(self, name):
        self.class_name = name

    def get_class(self):
        return self.class_name
    
    def get_gauss_means(self):
        return self.model.means_