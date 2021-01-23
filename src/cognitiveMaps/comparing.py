from . import consts

def weights_distance(weights_a, weights_b):
    result = 0
    for i in range(len(weights_a)):
        for j in range(len(weights_a[0])):
            result += abs(weights_a[i][j]-weights_b[i][j])
    return result


def nn_weights(models, m):
    best_cost = 1000
    best_model = None
    for model in models:
        cost = weights_distance(model.weights, m.weights)
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model


def nn_weights_and_start_values(models, m, input_size, extend_size):
    best_cost = 1000
    best_model = None
    for model in models:
        cost = weights_distance(model.weights, m.weights)
        cost += sum(consts.E(input_size+extend_size, input_size).dot(model.start_values-m.start_values))
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model


def nn_convergence(models, m, first_input):
    best_cost = 100000
    best_model = None
    pnt = m.get_convergence_point(first_input)
    for model in models:
        m_pnt = model.get_convergence_point(first_input)
        cost = sum(pnt-m_pnt)
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model


def best_prediction(models, xs):
    best_cost = 100000
    best_model = None
    for model in models:
        predicted_xs = model.predict(xs, len(xs))
        # print(f"Model {model.get_class()} predicted {predicted_xs[len(xs)-1]} (should be {xs[len(xs)-1]})")
        cost = ((predicted_xs - xs)**2).mean()
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model