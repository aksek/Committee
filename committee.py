# Aniela Kosek

import random
from typing import List, Any

import numpy as np
import pandas as pd
from numpy.core._multiarray_umath import ndarray


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
            temp_X = X.iloc[lambda x: x.index % len(self.classifiers) == i]
            split_X.append(temp_X)
            temp_y = y[i::len(self.classifiers)]
            split_y.append(temp_y)
            i += 1

        print(X)
        print(split_X[0])
        # print(split_y)

        # i = 0
        # for index, row in X.iterrows():
        #
        #     split_X[i].append(row)
        #     i = (i + 1) % len(self.classifiers)
        #
        # i = 0
        # for row in y.tolist():
        #     split_y[i].append(row)
        #     i = (i + 1) % len(self.classifiers)

        # train all algorithms
        i = 0
        for clf in self.classifiers:
            clf.fit(split_X[i], split_y[i])
            i += 1

        self.is_fitted = True
        return self

    def predict(self, X):
        if self.is_fitted:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers])
            majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions),
            return majority
        else:
            raise RuntimeError("The estimator was not fitted yet")
