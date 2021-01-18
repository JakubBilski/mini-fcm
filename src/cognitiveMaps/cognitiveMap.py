import numpy as np

from . import weightsGeneration
from . import displaying
from . import consts


class FuzzyCognitiveMap:
    def __init__(self, weights=None):
        self.class_name = ""
        self.weights = weights
        self.conv_pnt = None

    def get_convergence_point(self, input_data, max_iterations=100):
        if self.conv_pnt is None:
            self._calculate_convergence_pnt(input_data, max_iterations)
        return self.conv_pnt

    def f(x):
        return 1 / (1 + np.exp(-x))

    def fprim(x):
        pom = FuzzyCognitiveMap.f(x)
        return pom*(1-pom)
    
    def _calculate_convergence_pnt(self, input_data, max_iterations):
        output = input_data.transpose()
        weights = np.matrix(self.weights)
        for i in range(max_iterations):
            buffer = FuzzyCognitiveMap.f(weights @ output)
            if (buffer == output).all():
                # print(f"fixed-point attractor found after {i} steps")
                break
            # print(output[1])
            output, buffer = buffer, output
        self.conv_pnt = output.transpose()

    def train(self, input_in_time):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        expected_output = np.array(expected_output)
        input_in_time = np.array(input_in_time)
        F_minus_Y = -np.log(-expected_output+1.001)
        self.weights = np.linalg.pinv(input_in_time).dot(F_minus_Y)
        # print("trained weights")
        # for vector in self.weights:
        #     print(vector)
        # weightsGeneration.set_w1_weights(self.model, trained_weights)

    def get_error(self, input_in_time):
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        expected_output = np.array(expected_output)
        input_in_time = np.array(input_in_time)
        error = 0
        for i in range(len(input_in_time)-1):
            weights = np.matrix(self.weights)
            result = FuzzyCognitiveMap.f(weights.dot(input_in_time[i]))
            result -= expected_output[i]
            error += result.dot(np.transpose(result))
        return error

    def display_plot(self, save_path=None):
        displaying.draw_cognitive_map(self.weights, self.class_name, save_path)

    def set_class(self, class_name):
        self.class_name = class_name

    def get_class(self):
        return self.class_name

class ExtendedCognitiveMap(FuzzyCognitiveMap):
    def __init__(self, k, n):
        super().__init__(weights=None)
        self.k = k
        self.n = n
        self.start_values = np.random.rand(n)
        self.weights = np.random.rand(n,n)

    def train_step(self, input_in_time, learning_rate=0.002):
        k = self.k
        n = self.n
        xs = input_in_time
        p = len(xs)
        A = consts.A(n,k)
        B = consts.B(n,k)
        C = consts.C(n,k)
        yprimes = np.zeros(shape=(n*n,p,n))
        ys = np.zeros(shape=(p,n))
        ys[0] = A.dot(xs[0])+B.dot(self.start_values)
        W = self.weights
        P=0
        Pwprimes = np.zeros(shape=(n,n))
        Py0primes = np.zeros(shape=(p,n))
        Py0primes2 = np.zeros(shape=(p,n,n))
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

        #No idea why need to transpose here
        W += -learning_rate*np.transpose(Pwprimes)
        ys[0] += B.dot(-learning_rate*Py0primes[p-1])
        self.weights = W
        self.start_values = ys[0]


    def predict(self, input_in_time, steps):
        k = self.k
        n = self.n
        A = consts.A(n,k)
        B = consts.B(n,k)
        D = consts.D(n,k)
        y = A.dot(input_in_time[0])+B.dot(self.start_values)
        W = self.weights
        step = 1
        output = np.zeros(shape=(steps, k))
        output[0] = input_in_time[0]
        for t in range(1,steps):
            y = FuzzyCognitiveMap.f(W.dot(A.dot(input_in_time[t])+B.dot(y)))
            output[step] = D.dot(y)
        return output


    def train(self, input_in_time, learning_rate=0.002, steps=20):
        k = self.k
        n = self.n
        xs = input_in_time
        p = len(xs)
        A = consts.A(n,k)
        B = consts.B(n,k)
        C = consts.C(n,k)
        yprimes = np.zeros(shape=(n*n,p,n))
        ys = np.zeros(shape=(p,n))
        ys[0] = A.dot(xs[0])+B.dot(self.start_values)
        W = self.weights
        P=0

        step = 0
        for i in range(steps):
            Pwprimes = np.zeros(shape=(n,n))
            Py0primes = np.zeros(shape=(p,n))
            Py0primes2 = np.zeros(shape=(p,n,n))
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

            step+=1
            # print(f"Step {step}, error: {P}")
            #No idea why need to transpose here
            W += -learning_rate*np.transpose(Pwprimes)
            ys[0] += B.dot(-learning_rate*Py0primes[p-1])
        # print(f"Final-1 cost: {P}")
        self.weights = W
        self.start_values = ys[0]


    def get_error(self, input_in_time):
        n=self.n
        k=self.k
        weights = self.weights
        A = consts.A(n,k)
        B = consts.B(n,k)
        D = consts.C(n,k)
        expected_output = input_in_time[1:]
        input_in_time = input_in_time[:-1]
        error = 0
        result = A.dot(input_in_time[0])+B.dot(self.start_values)
        for i in range(len(input_in_time)-1):
            result = A.dot(input_in_time[i])+B.dot(result)
            result = FuzzyCognitiveMap.f(weights.dot(result))
            error += np.transpose(D.dot(result)-expected_output[i]).dot(D.dot(result)-expected_output[i])
        return error

    def _calculate_convergence_pnt(self, input_data, max_iterations):
        n=self.n
        k=self.k
        weights = self.weights
        A = consts.A(n,k)
        B = consts.B(n,k)
        result = A.dot(input_data)+B.dot(self.start_values)
        for i in range(max_iterations):
            buffer = FuzzyCognitiveMap.f(weights.dot(result))
            if (buffer == result).all():
                # print(f"fixed-point attractor found after {i} steps")
                break
            # print(output[1])
            result, buffer = buffer, result
        self.conv_pnt = result