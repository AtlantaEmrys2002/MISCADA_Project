# PLEASE NOTE THAT THIS IS NOT MY CODE
# This is a pre-implemented LogitBoost Classifier from:
# https://livebook.manning.com/concept/machine-learning/logitboost
# and https://github.com/gkunapuli/ensemble-methods-notebooks/blob/master/Ch4.5-LogitBoost-boosting-with-the-logistic-
# loss.ipynb

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.special import expit


def predict_logit_boosting(X, estimators):
    pred = np.zeros((X.shape[0], ))

    for h in estimators:
        pred += h.predict(X)

    y = (np.sign(pred) + 1) / 2

    return y


def fit_logitboosting(X, y, n_estimators=10):

    n_samples, n_features = X.shape
    D = np.ones((n_samples,)) / n_samples
    p = np.full((n_samples,), 0.5)
    estimators = []

    for t in range(n_estimators):
        z = (y - p) / (p * (1 - p))
        D = p * (1 - p)

        h = DecisionTreeRegressor(max_depth=1)
        h.fit(X, z, sample_weight=D)
        estimators.append(h)

        if t == 0:
            margin = np.array([h.predict(X) for h in estimators]).reshape(-1, )
        else:
            margin = np.sum(np.array([h.predict(X) for h in estimators]), axis=0)

        p = expit(margin)

    return estimators
