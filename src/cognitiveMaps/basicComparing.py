from sklearn.cluster import KMeans
import numpy as np
import warnings

from . import consts

maps = [
    [6, 7, 8],
    [6, 8, 7],
    [7, 6, 8],
    [7, 8, 6],
    [8, 6, 7],
    [8, 7, 6]
]


def weights_distance(weights_a, weights_b, n, k):
    if n == k:
        return weights_distance_old(weights_a, weights_b)

    best_result = 10000
    for map in maps:
        result = 0
        for i in range(n):
            for j in range(n):
                if i > k:
                    if j > k:
                        result += abs(weights_a[i][j]-weights_b[map[i-k]][map[j-k]])  # noqa: E501
                    else:
                        result += abs(weights_a[i][j]-weights_b[map[i-k]][j])
                elif j > k:
                    result += abs(weights_a[i][j]-weights_b[i][map[j-k]])

        if result < best_result:
            best_result = result

    for i in range(k):
        for j in range(k):
            best_result += abs(weights_a[i][j]-weights_b[i][j])

    return best_result


def weights_distance_old(weights_a, weights_b):
    return np.absolute(weights_a-weights_b).sum()


def nn_weights(models, m, n, k):
    best_cost = 100000
    best_model = None
    for model in models:
        cost = weights_distance(model.weights, m.weights, n, k)
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model, best_cost


def nn_weights_and_start_values(models, m, input_size, extend_size):
    best_cost = 1000
    best_model = None
    for model in models:
        cost = weights_distance_old(model.weights, m.weights)
        cost += sum(consts.E(
                input_size+extend_size, input_size
            ).dot(model.start_values-m.start_values))
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
        cost = ((predicted_xs - xs)**2).mean()
        if cost < best_cost:
            best_model = model
            best_cost = cost
    return best_model


def best_mse_sum(models, m, no_classes):
    costs = [0 for _ in range(no_classes)]
    dividers = [0 for _ in range(no_classes)]
    for model in models:
        cost = weights_distance_old(model.weights, m.weights)
        costs[int(model.get_class())-1] += cost
        dividers[int(model.get_class())-1] += 1
    mean_cost = [costs[i]/dividers[i] for i in range(no_classes)]
    return mean_cost.index(min(mean_cost))+1, min(mean_cost)


def get_grouping_factor(models, input_size, extend_size, no_clusters):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vects = [model.weights.flatten().tolist() for model in models]
        if extend_size > 0:
            for i in range(len(models)):
                vects[i] = np.append(vects[i],
                                    consts.E(
                        input_size+extend_size,
                        input_size
                        ).dot(models[i].start_values).flatten().tolist()
                    )

        centers = np.zeros(shape=(no_clusters, len(vects[0])))
        no_center_members = np.zeros(shape=(no_clusters))
        for i in range(len(vects)):
            cluster_class = int(models[i].get_class())-1
            centers[cluster_class] += vects[i]
            no_center_members[cluster_class] += 1
        for c in range(centers.shape[0]):
            centers[c] /= no_center_members[c]

        kmeans = KMeans(n_clusters=no_clusters, init=centers).fit(vects)

        classes = [int(model.get_class()-1) for model in models]

        # print(kmeans.labels_)
        # print(classes)
    return sum(kmeans.labels_ == classes)/len(models)
