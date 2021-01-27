# Aniela Kosek
import random
from statistics import mean

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from committee import CommitteeClassifier
from decisionTree import DecisionTree
from utility import calculate_metrics
from utility import encode

# load data
df = pd.read_csv("mushrooms.csv")

# split columns
X = df.drop("class", axis=1)
y = df["class"].values
encoded_X = encode(X)

# initialize classifiers
clf1 = DecisionTree()
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
clf4 = GaussianNB()

accuracy = []
precision = []
recall = []
f_score = []

test_cases = [[clf1], [clf2], [clf3], [clf4], [clf1, clf2, clf3, clf4], [clf2, clf3, clf4]]

test_number = 1
for classifiers in test_cases:

    for i in range(25):
        # split into training and testing sets
        rand = random.randrange(100000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)
        encoded_X_train, encoded_X_test = train_test_split(encoded_X, test_size=0.2, random_state=rand)

        # initialize and fit the voting classifier
        eclf = CommitteeClassifier([clf1, clf2])
        eclf = eclf.fit(X_train, y_train, encoded_X=encoded_X_train)

        # test the result
        acc, prec, rec, f = calculate_metrics(eclf, X_test, y_test, encoded_X=encoded_X_test)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f_score.append(f)
        print(str((i + 1) * 4) + "%")

    print("test: ", test_number)
    print("Accuracy: " + str(mean(accuracy)))
    print("Precision: " + str(mean(precision)))
    print("Recall: " + str(mean(recall)))
    print("F_score: " + str(mean(f_score)))
    print()
    test_number += 1

