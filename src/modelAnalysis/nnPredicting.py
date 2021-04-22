def predict_nn_fcm(train_models, test_xs):
    predicted_ys = []
    for xs in test_xs:
        fit_class = train_models[0].get_class()
        best_prediction_err = train_models[0].get_error(xs)
        for train_model in train_models[1:]:
            prediction_err = train_model.get_error(xs)
            if prediction_err < best_prediction_err:
                fit_class = train_model.get_class()
                best_prediction_err = prediction_err
        predicted_ys.append(fit_class)
    return predicted_ys


def predict_nn_hmm(train_models, test_xs):
    predicted_ys = []
    for xs in test_xs:
        fit_class = train_models[0].get_class()
        best_em_prob = train_models[0].get_emission_probability(xs)
        for train_model in train_models[1:]:
            em_prob = train_model.get_emission_probability(xs)
            if em_prob > best_em_prob:
                fit_class = train_model.get_class()
                best_em_prob = em_prob
        predicted_ys.append(fit_class)
    return predicted_ys
