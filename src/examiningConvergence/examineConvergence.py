import numpy as np
import itertools

from cognitiveMaps import weightsGeneration
from cognitiveMaps import cognitiveMap
from cognitiveMaps import displaying

MAX_INFERE_ITERATIONS = 1000

def examineConvergence(fcm):
    input_size = len(fcm.weights)
    step = 0.1
    xs = itertools.product(*[np.arange(0, 1, step) for _ in range(input_size)])
    fcm_output = fcm.infere(np.array([[a for a in x] for x in xs]), MAX_INFERE_ITERATIONS)
    unique_outputs = np.unique(ar=fcm_output, axis=0)
    print(f"Unique attractors: {unique_outputs}")
