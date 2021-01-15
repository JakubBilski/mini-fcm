import numpy as np



def A(n, k):
    res = np.zeros(shape=(n,k))
    for i in range(k):
        res[i][i] = 1
    return res


def B(n, k):
    res = np.zeros(shape=(n,n))
    for i in range(k, n):
        res[i][i] = 1
    return res


def C(n, k):
    res = np.zeros(shape=(n,n))
    for i in range(k):
        res[i][i] = 1
    return res


def D(n, k):
    res = np.zeros(shape=(k,n))
    for i in range(k):
        res[i][i] = 1
    return res