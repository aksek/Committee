# Aniela Kosek
from statistics import mean

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import committee as cmt
from decisionTree import DecisionTree
from utility import calculate_metrics

# load data
df = pd.read_csv("IRIS.csv")

# split columns
X = df.drop("species", axis=1)
y = df["species"].values

# initialize classifiers
clf1 = DecisionTree()
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
clf4 = GaussianNB()

test_cases = [[clf1], [clf2], [clf3], [clf4], [clf1, clf2, clf3, clf4], [clf2, clf3, clf4]]

test_number = 1
for classifiers in test_cases:

    accuracy = []
    precision = []
    recall = []
    f_score = []

    for i in range(25):
        # split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # initialize and fit the voting classifier
        eclf = cmt.CommitteeClassifier(classifiers)
        eclf = eclf.fit(X_train, y_train)

        # test the result
        acc, prec, rec, f = calculate_metrics(eclf, X_test, y_test)
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
