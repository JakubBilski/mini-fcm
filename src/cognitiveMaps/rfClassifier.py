from sklearn.ensemble import RandomForestClassifier

class RFClassifier:
    def __init__(self, weightss, ys) -> None:
        self.clf = RandomForestClassifier(n_estimators=500, random_state=0)
        squashed_weightss = []
        for weights in weightss:
            squashed_weightss.append(weights.flatten().tolist())
        self.clf.fit(squashed_weightss, ys)
    
    def predict(self, weights):
        return self.clf.predict([weights.flatten().tolist()])