# flake8: noqa
import numpy as np

from .baseCognitiveMap import BaseCognitiveMap
from . import consts


class ExtendedCognitiveMap(BaseCognitiveMap):
    def __init__(self, k, n):
        super().__init__(weights=None)
        self.k = k
        self.n = n
        np.random.seed = 0
        self.start_values = np.random.rand(n)
        self.weights = np.random.rand(n, n)

    def train_step(self, input_in_time, learning_rate=0.002):
        k = self.k
        n = self.n
        xs = input_in_time
        p = len(xs)
        A = consts.A(n, k)
        B = consts.B(n, k)
        C = consts.C(n, k)
        yprimes = np.zeros(shape=(n*n, p, n))
        ys = np.zeros(shape=(p, n))
        ys[0] = A.dot(xs[0])+B.dot(self.start_values)
        W = self.weights
        Pwprimes = np.zeros(shape=(n, n))
        Py0primes = np.zeros(shape=(p, n))
        Py0primes2 = np.zeros(shape=(p, n, n))
        Py0primes2[0] = np.eye(n)
        Ps = np.zeros(shape=(p, n))
        P = 0
        for t in range(1, p):
            ys[t] = ExtendedCognitiveMap.f(W.dot(A.dot(xs[t-1]) + B.dot(ys[t-1])))
            buff = ExtendedCognitiveMap.fprim(W.dot(A.dot(xs[t-1])+B.dot(ys[t-1])))
            Ps[t] = C.dot(ys[t]) - A.dot(xs[t])
            for b in range(n):
                for a in range(k):
                    yprimes[a*n+b][t] = W.dot(B.dot(yprimes[a*n+b][t-1]))
                    yprimes[a*n+b][t][b] += xs[t-1][a]
                    yprimes[a*n+b][t] = np.multiply(buff, yprimes[a*n+b][t])
                    Pwprimes[a][b] += np.transpose(2*Ps[t]).dot(yprimes[a*n+b][t])
                for a in range(k, n):
                    yprimes[a*n+b][t] = W.dot(B.dot(yprimes[a*n+b][t-1]))
                    yprimes[a*n+b][t][b] += ys[t-1][a]
                    yprimes[a*n+b][t] = np.multiply(buff, yprimes[a*n+b][t])
                    Pwprimes[a][b] += np.transpose(2*Ps[t]).dot(yprimes[a*n+b][t])
            Py0primes2[t] = Py0primes2[t-1].dot(buff.dot(B))
            P += np.transpose(Ps[t]).dot(Ps[t])
        for t in range(1, p):
            Py0primes[t] = np.transpose(2*Ps[t]).dot(Py0primes2[t])
            Py0primes[t] = Py0primes[t] + Py0primes[t-1]
        W += -learning_rate*np.transpose(Pwprimes)
        ys[0] += B.dot(-learning_rate*Py0primes[p-1])
        self.weights = W
        self.start_values = ys[0]

    def train(self, input_in_time, learning_rate, steps):
        k = self.k
        n = self.n
        xs = input_in_time
        p = len(xs)
        A = consts.A(n, k)
        B = consts.B(n, k)
        C = consts.C(n, k)
        yprimes = np.zeros(shape=(n*n, p, n))
        ys = np.zeros(shape=(p, n))
        ys[0] = A.dot(xs[0])+B.dot(self.start_values)
        W = self.weights
        P = 0
        for step in range(steps):
            Pwprimes = np.zeros(shape=(n, n))
            Py0primes = np.zeros(shape=(p, n))
            Py0primes2 = np.zeros(shape=(p, n, n))
            Py0primes2[0] = np.eye(n)
            Ps = np.zeros(shape=(p, n))
            P = 0
            for t in range(1, p):
                ys[t] = ExtendedCognitiveMap.f(W.dot(A.dot(xs[t-1]) + B.dot(ys[t-1])))
                buff = ExtendedCognitiveMap.fprim(W.dot(A.dot(xs[t-1])+B.dot(ys[t-1])))
                Ps[t] = C.dot(ys[t]) - A.dot(xs[t])
                for b in range(n):
                    for a in range(k):
                        yprimes[a*n+b][t] = W.dot(B.dot(yprimes[a*n+b][t-1]))
                        yprimes[a*n+b][t][b] += xs[t-1][a]
                        yprimes[a*n+b][t] = np.multiply(buff, yprimes[a*n+b][t])
                        Pwprimes[a][b] += np.transpose(2*Ps[t]).dot(yprimes[a*n+b][t])
                    for a in range(k, n):
                        yprimes[a*n+b][t] = W.dot(B.dot(yprimes[a*n+b][t-1]))
                        yprimes[a*n+b][t][b] += ys[t-1][a]
                        yprimes[a*n+b][t] = np.multiply(buff, yprimes[a*n+b][t])
                        Pwprimes[a][b] += np.transpose(2*Ps[t]).dot(yprimes[a*n+b][t])
                Py0primes2[t] = Py0primes2[t-1].dot(buff.dot(B))
                P += np.transpose(Ps[t]).dot(Ps[t])
            for t in range(1, p):
                Py0primes[t] = np.transpose(2*Ps[t]).dot(Py0primes2[t])
                Py0primes[t] = Py0primes[t] + Py0primes[t-1]
            W += -learning_rate*np.transpose(Pwprimes)
            ys[0] += B.dot(-learning_rate*Py0primes[p-1])
        self.weights = W
        self.start_values = ys[0]

    def get_error(self, input_in_time):
        n = self.n
        k = self.k
        weights = self.weights
        A = consts.A(n, k)
        B = consts.B(n, k)
        D = consts.D(n, k)
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        error = 0
        result = A.dot(input_in_time[0])+B.dot(self.start_values)
        for i in range(len(input_in_time)-1):
            result = A.dot(input_in_time[i])+B.dot(result)
            result = BaseCognitiveMap.f(weights.dot(result))
            error += np.transpose(D.dot(result)-expected_output[i]).dot(
                D.dot(result)-expected_output[i])
        return error

    def _calculate_convergence_pnt(self, input_data, max_iterations):
        n = self.n
        k = self.k
        weights = self.weights
        A = consts.A(n, k)
        B = consts.B(n, k)
        result = A.dot(input_data)+B.dot(self.start_values)
        for i in range(max_iterations):
            buffer = BaseCognitiveMap.f(weights.dot(result))
            if (buffer == result).all():
                # print(f"fixed-point attractor found after {i} steps")
                break
            # print(output[1])
            result, buffer = buffer, result
        self.conv_pnt = result
