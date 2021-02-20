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

    raise Exception(f"Classification method {classification_method} not recognized")
