from cognitiveMaps import weightsGeneration
from cognitiveMaps import cognitiveMap
from examiningConvergence import examineConvergence

if __name__ == "__main__":
    input_size = 5
    weights = weightsGeneration.get_random_sparse_weights(input_size, 3)
    fcm = cognitiveMap.FuzzyCognitiveMap(weights)
    examineConvergence.examineConvergence(fcm)
    fcm.display_plot()