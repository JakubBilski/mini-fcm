import sklearn.svm as svm

class SVMClassifier:
    def __init__(self, weightss, ys) -> None:
        self.clf = svm.SVC(kernel='rbf')
        squashed_weightss = []
        for weights in weightss:
            squashed_weightss.append(weights.flatten().tolist())
        self.clf.fit(squashed_weightss, ys)
    
    def predict(self, weights):
        return self.clf.predict([weights.flatten().tolist()])