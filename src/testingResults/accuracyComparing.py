from . import basicMapsComparing
from . import rfClassifier
from . import svmClassifier


def get_accuracy(train_models, test_models, test_xs, input_size, no_classes, classification_method="rf"):
    if classification_method == "nn_weights":
        mistakes = 0
        for test_model in test_models:
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            best_fit, best_cost = basicMapsComparing.nn_weights(train_models_without_same, test_model, input_size, input_size)
            if best_fit.get_class() != test_model.get_class():
                good_predictions = [m for m in train_models_without_same if m.get_class()==test_model.get_class()]
                best_correct_fit, best_correct_cost = basicMapsComparing.nn_weights(good_predictions, test_model, input_size, input_size)
                mistakes += 1
            return 1-mistakes/len(test_models)


    if classification_method == "nn_convergence":
        mistakes = 0
        for test_model, xs in zip(test_models, test_xs):
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            best_fit = basicMapsComparing.nn_convergence(train_models_without_same, test_model, xs[0])
            if best_fit.get_class() != test_model.get_class():
                mistakes += 1
        return 1-mistakes/len(test_models)

    if classification_method == "best_mse_sum":
        mistakes = 0
        for test_model in test_models:
            train_models_without_same = [m for m in train_models if (m.weights != test_model.weights).any()]
            fit_class, _ = basicMapsComparing.best_mse_sum(train_models_without_same, test_model, no_classes)
            if fit_class != test_model.get_class():
                mistakes += 1
        return 1-mistakes/len(test_models)

    if classification_method == "rf":
        mistakes = 0
        cfr = rfClassifier.RFClassifier(
            [tm.weights for tm in train_models],
            [tm.class_name for tm in train_models]
            )
        for test_model in test_models:
            fit_class = cfr.predict(test_model.weights)
            if fit_class != test_model.get_class():
                mistakes += 1
        return 1-mistakes/len(test_models)

    if classification_method == "svm":
        mistakes = 0
        cfr = svmClassifier.SVMClassifier(
            [tm.weights for tm in train_models],
            [tm.class_name for tm in train_models]
            )
        for test_model in test_models:
            fit_class = cfr.predict(test_model.weights)
            if fit_class != test_model.get_class():
                mistakes += 1
        return 1-mistakes/len(test_models)

    if classification_method == "best_prediction":
        mistakes = 0
        for test_model, xs in zip(test_models, test_xs):
            fit_class = train_models[0].get_class()
            best_prediction_err = train_models[0].get_error(xs)
            for train_model in train_models[1:]:
                prediction_err = train_model.get_error(xs)
                if prediction_err < best_prediction_err:
                    fit_class = train_model.get_class()
                    best_prediction_err = prediction_err
            if fit_class != test_model.get_class():
                mistakes += 1
        return 1-mistakes/len(test_models)

    raise Exception(f"Classification method {classification_method} not recognized")


def get_accuracy_hmm(train_models, test_xs, test_classes, input_size, no_classes, classification_method="best_prediction"):
    if classification_method == "best_prediction":
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

    raise Exception(f"Classification method {classification_method} not recognized")

def get_accuracy_best_prediction_multicenter(train_models_by_ys, test_models, test_xs_by_ys):
    mistakes = 0
    for i in range(len(test_models)):
        fit_class = None
        best_prediction_err = 100000000
        for y in range(len(train_models_by_ys)):
            test_xs = test_xs_by_ys[y][i]
            for train_model in train_models_by_ys[y]:
                prediction_err = train_model.get_error(test_xs)
                if prediction_err < best_prediction_err:
                    fit_class = y
                    best_prediction_err = prediction_err
        if fit_class != test_models[i].get_class():
            mistakes += 1

    return 1-mistakes/len(test_models)