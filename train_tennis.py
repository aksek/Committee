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
df = pd.read_csv("play_tennis.csv")

# drop irrelevant columns
df = df.drop("day", axis=1)

# split columns and drop irrelevant columns
X = df.drop("play", axis=1)
y = df["play"].values
encoded_X = encode(X)

accuracy = []
precision = []
recall = []
f_score = []

for i in range(25):
    # split into training and testing sets
    rand = random.randrange(100000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)
    encoded_X_train, encoded_X_test = train_test_split(encoded_X, test_size=0.2, random_state=rand)

    # initialize classifiers
    clf1 = DecisionTree()
    clf2 = KNeighborsClassifier(n_neighbors=2)
    clf3 = SVC(kernel='rbf', probability=True)
    clf4 = GaussianNB()

    # initialize and fit the voting classifier
    eclf = CommitteeClassifier([clf2])
    eclf = eclf.fit(X_train, y_train, encoded_X=encoded_X_train)

    # test the result
    acc, prec, rec, f = calculate_metrics(eclf, X_test, y_test, encoded_X=encoded_X_test)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f_score.append(f)
    print(str(i * 4) + "%")

print("Accuracy: " + str(mean(accuracy)))
print("Precision: " + str(mean(precision)))
print("Recall: " + str(mean(recall)))
print("F_score: " + str(mean(f_score)))
