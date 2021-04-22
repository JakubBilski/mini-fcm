import sklearn.metrics

def get_accuracy(predicted, real):
    correct = 0
    for p, r in zip(predicted, real):
        if p == r:
            correct += 1
    return correct/len(predicted)


def get_mcc(predicted, real):
    return sklearn.metrics.matthews_corrcoef(real, predicted)
