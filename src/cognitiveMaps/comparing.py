from . import consts

maps = [
    [6, 7, 8]
    [6, 8, 7]
    [7, 6, 8]
    [7, 8, 6]
    [8, 6, 7]
    [8, 7, 6]
]

def weights_distance(weights_a, weights_b, n, k):
    best_result = 10000
    for map in maps:
        result = 0
        for i in range(n):
            for j in range(n):
                if i > k or j > k:
                    result += abs(weights_a[i][j]-weights_b[map[i-k]][map[j-k]])
        if result < best_result:
            best_result = result

    for i in range(k):
        for j in range(k):
            best_result += abs(weights_a[i][j]-weights_b[i][j])

    return best_result


def nn_weights(models, m, n, k):
    best_cost = 1000
    best_model = None
    for model in models:
        cost = weights_distance(model.weights, m.weights, n, k)
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