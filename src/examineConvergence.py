import numpy as np
import itertools

from cognitiveMaps import weightsGeneration
from cognitiveMaps import cognitiveMap

MAX_INFERE_ITERATIONS = 1000

if __name__ == "__main__":
    input_size = 3
    weights = weightsGeneration.get_random_weights(input_size)
    # print(weights)
    step = 0.1
    xs = itertools.product(*[np.arange(0, 1, step) for _ in range(input_size)])        
    fcm = cognitiveMap.FuzzyCognitiveMap(input_size, weights)
    fcm_output = fcm.infere(np.array([[a for a in x] for x in xs], MAX_INFERE_ITERATIONS))
    unique_outputs = np.unique(ar=fcm_output, axis=0)
