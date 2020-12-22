import random
import numpy as np

def correlation_coefficients(data, i, j):
    k = len(data.index)
    result = k*(data.iloc[:,i]*data.iloc[:,j]).sum()-data.iloc[:,i].sum()*data.iloc[:,j].sum()
    result /= k*(data.iloc[:,j]*data.iloc[:,j]).sum()-data.iloc[:,j].sum()*data.iloc[:,j].sum()
    return result

def get_computed_weights(data):
    weights = []
    n = len(data[0])
    for j in range(n):
        to_add = [[correlation_coefficients(data, i, j)] for i in range(n) if j!=i]
        to_add.insert(j, [0.0])
        weights.append(to_add)
    return weights


def get_random_weights(n):
    weights = []    
    for i in range(n):
        to_add = [random.uniform(0, 1) for _ in range(n)]
        to_add[i] = 1.0
        weights.append(to_add)
    return weights


def get_random_high_weights(n):
    weights = []    
    for i in range(n):
        to_add = [random.uniform(0.9, 1) for _ in range(n)]
        to_add[i] = 0.0
        weights.append(to_add)
    return weights


def get_random_sparse_weights(n, no_edges_from_node):
    weights = []    
    for i in range(n):
        to_add = [0 for _ in range(n)]
        edges_from_node = random.choices(range(n), k=no_edges_from_node)
        for edge in edges_from_node:
            to_add[edge] = random.uniform(0, 1)
        to_add[i] = 0.0
        weights.append(to_add)
    return weights

def set_w1_weights(model, weights):
    weights = [(np.array([[w] for w in weight]), np.array([0.0])) for weight in weights]
    for i in range(len(weights)):
        model.layers[i+1].set_weights(weights[i])
