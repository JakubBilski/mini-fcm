import pathlib
import os
import warnings
import numpy as np
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm

from loadingData import loadSktime
from transformingData import derivatives, normalizing, cmeans
from examiningData import displaying

plots_dir = pathlib.Path(f'plots\\{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}\\')

if __name__ == "__main__":

    os.mkdir(plots_dir)

    test_path=pathlib.Path('./data/Univariate/ACSF1/ACSF1_TEST.ts')
    train_path=pathlib.Path('./data/Univariate/ACSF1/ACSF1_TRAIN.ts')
    derivative_order=1
    test_xses_series, test_ys = loadSktime.load_sktime(test_path)
    test_xses_series = derivatives.transform(test_xses_series, derivative_order)
    
    train_xses_series, train_ys = loadSktime.load_sktime(train_path)
    train_xses_series = derivatives.transform(train_xses_series, derivative_order)

    mins, maxs = normalizing.get_mins_and_maxs(test_xses_series + train_xses_series)
    test_xses_series = normalizing.transform(test_xses_series, mins, maxs)
    train_xses_series = normalizing.transform(train_xses_series, mins, maxs)

    input_size = 2
    no_states = 3

    hmm_train_models = []
    hmm_error_occured = False

    input_in_time = train_xses_series[0]

    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    models_with_scores = []
    for i in tqdm(range(100)):
        new_model = GaussianHMM(n_components=no_states, covariance_type="diag", n_iter=1000, init_params='')
        random_transmat = np.random.rand(no_states, no_states)
        random_transmat = random_transmat / random_transmat.sum(1, keepdims=True)
        new_model.transmat_ = random_transmat
        new_model.means_ = np.random.rand(no_states, input_size)
        random_startprob = np.random.rand(no_states)
        random_startprob = random_startprob / random_startprob.sum(0, keepdims=True)
        new_model.startprob_ = random_startprob
        random_covars = np.array([
            np.random.rand(input_size) for _ in range(no_states)
        ])
        new_model.covars_ = random_covars
        new_model.fit(input_in_time)
        # sometimes, despite converging, the model will be invalid
        # and it will hopefully raise an error during score()
        if new_model.monitor_.converged:
            try:
                score = new_model.score(input_in_time)
                models_with_scores.append((new_model, score))
            except:
                pass
    warnings.resetwarnings()
    if len(models_with_scores) == 0:
        raise Exception("Unable to learn a valid model")
    # print(sorted([s for m, s in models_with_scores]))
    unique_scores = {}
    unique_score_coordinates = {}
    for m, s in models_with_scores:
        s_rounded = f'{s:.6f}'
        if s_rounded in unique_scores.keys():
            unique_scores[s_rounded] += 1
        else:
            unique_scores[s_rounded] = 1
            unique_score_coordinates[s_rounded] = m
    print("Unique maxima")
    for score, no_occ in unique_scores.items():
        print(f"{no_occ} times converged to maximum with score {score}")
