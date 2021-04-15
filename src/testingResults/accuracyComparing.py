def get_accuracy_fcm_best_prediction(train_models, test_xs, test_classes):
    mistakes = 0
    for y, xs in zip(test_classes, test_xs):
        fit_class = train_models[0].get_class()
        best_prediction_err = train_models[0].get_error(xs)
        for train_model in train_models[1:]:
            prediction_err = train_model.get_error(xs)
            if prediction_err < best_prediction_err:
                fit_class = train_model.get_class()
                best_prediction_err = prediction_err
        if fit_class != y:
            mistakes += 1
    return 1-mistakes/len(test_classes)


def get_accuracy_hmm_best_prediction(train_models, test_xs, test_classes):

    mistakes = 0
    for test_class, xs in zip(test_classes, test_xs):
        fit_class = train_models[0].get_class()
        best_emission_prob = train_models[0].get_emission_probability(xs)
        for train_model in train_models[1:]:
            emission_prob = train_model.get_emission_probability(xs)
            if emission_prob > best_emission_prob:
                fit_class = train_model.get_class()
                best_emission_prob = emission_prob
        if fit_class != test_class:
            mistakes += 1
    return 1-mistakes/len(test_classes)
