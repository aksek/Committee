# Aniela Kosek

from typing import List, Any

import numpy as np
import pandas as pd
from scipy import stats


class CommitteeClassifier:
    classifiers = []
    is_fitted = False

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):

        # split the dataframe
        split_X: List[pd.DataFrame] = []
        split_y: List[List[Any]] = []

        i = 0
        while i < len(self.classifiers):
            temp_X = X[i::len(self.classifiers)]
            split_X.append(temp_X)
            temp_y = y[i::len(self.classifiers)]
            split_y.append(temp_y)
            i += 1

        # train all algorithms
        i = 0
        for clf in self.classifiers:
            clf.fit(split_X[i], split_y[i])
            i += 1

        self.is_fitted = True
        return self

    def predict(self, X):
        if self.is_fitted:
            predictions: np.ndarray = np.asarray([clf.predict(X) for clf in self.classifiers])
            majority = stats.mode(predictions)
            return majority[0][0]
        else:
            raise RuntimeError("The estimator was not fitted yet")
