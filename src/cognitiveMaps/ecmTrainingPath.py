import json
import numpy as np

from .extendedCognitiveMap import ExtendedCognitiveMap

class ECMTrainingPath:
    def __init__(self, learning_rate, class_name) -> None:
        self.points = []
        self.learning_rate = learning_rate
        self.class_name = class_name
    
    def to_json(self):
        d = {}
        d['weights'] = [p.weights.tolist() for p in self.points]
        d['start_values'] = [p.start_values.tolist() for p in self.points]
        d['learning_rate'] = self.learning_rate
        d['class_name'] = self.class_name
        d['k'] = self.points[0].k
        d['n'] = self.points[0].n
        return json.dumps(d)

    def from_json(source):
        d = json.loads(source)
        learning_rate = d['learning_rate']
        class_name = d['class_name']
        n = d['n']
        k = d['k']
        training_path = ECMTrainingPath(learning_rate, class_name)
        for ws, sv in zip(d['weights'], d['start_values']):
            ecm = ExtendedCognitiveMap(k, n)
            ecm.weights = np.asarray(ws) 
            ecm.start_values = np.asarray(sv)
            ecm.set_class(class_name)
            training_path.points.append(ecm)
        return training_path

        