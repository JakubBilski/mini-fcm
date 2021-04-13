from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def _minimized_function(x, *args):
        return np.abs(0.5-np.mean(np.multiply(x,x)))

if __name__ == "__main__":
    bounds = [(-1, 1) for _ in range(4)]
    step = 10
    ceiling = 1000
    iterrange = list(range(1,ceiling,step))
    errors = []
    last_maxiter = ceiling
    for maxiter in tqdm(iterrange):
        result = differential_evolution(
            _minimized_function,
            bounds,
            None,
            maxiter=maxiter,
            strategy='rand1bin',
            mutation=0.8,
            recombination=0.9,
            init='random',
            tol=0.00000001,
            seed=1000)
        print(f"maxiter: {maxiter} no_iterations: {result.nit}")
        print(f"Solution: {result.x}")
        errors.append(result.fun)
        if result.nit < maxiter:
            last_maxiter = maxiter
            break
    
    fig, ax = plt.subplots()
    ax.plot(list(range(1,last_maxiter+1,step)), errors)
    plt.show()
    plt.close()