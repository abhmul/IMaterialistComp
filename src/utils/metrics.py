import numpy as np


def f1_score(y_true, y_pred):

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum(y_pred * y_true, axis=0)
    fp = np.sum(y_pred * (1. - y_true), axis=0)
    fn = np.sum((1. - y_pred) * y_true, axis=0)

    smoothing = 1e-9

    # Calculate the f1 score
    f1 = (2 * tp + smoothing) / (2 * tp + fp + fn + smoothing)

    return f1
