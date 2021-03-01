from hmmlearn.hmm import GaussianHMM

class HMM:
    def __init__(self, n):
        self.model = GaussianHMM(n_components=n, covariance_type="full", n_iter=100)
        self.class_name = ""
        self.n = n

    def train(self, input_in_time):
        self.model.fit(input_in_time)

    def get_emission_probability(self, input_in_time):
        return self.model.score(input_in_time)

    def set_class(self, name):
        self.class_name = name

    def get_class(self):
        return self.class_name