# Aniela Kosek
import random
from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

import committee as cmt
from decisionTree import DecisionTree
from utility import calculate_metrics


# load data
df = pd.read_csv("IRIS.csv")

# split columns
X = df.drop("species", axis=1)
y = df["species"].values

mean_accuracy = []
mean_precision = []
mean_recall = []
mean_f_score = []

for size in range(86, 0, -2):

    accuracy = []
    precision = []
    recall = []
    f_score = []

    for i in range(25):
        # split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(size)/100)

        # initialize classifiers
        clf1 = DecisionTree()
        clf2 = KNeighborsClassifier(n_neighbors=4)
        clf3 = SVC(kernel='rbf', probability=True)
        clf4 = GaussianNB()

        # initialize and fit the voting classifier
        eclf = cmt.CommitteeClassifier([clf1])
        eclf = eclf.fit(X_train, y_train)

        # test the result
        acc, prec, rec, f = calculate_metrics(eclf, X_test, y_test)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f_score.append(f)
        print("size: ", size, " ", i * 4, "%")

    mean_accuracy.append(mean(accuracy))
    mean_precision.append(mean(precision))
    mean_recall.append(mean(recall))
    mean_f_score.append(mean(f_score))

f, ax = plt.subplots(1)
plt.plot(range(86, 0, -2), mean_accuracy, 'b-')
plt.plot(range(86, 0, -2), mean_precision, 'r-')
plt.plot(range(86, 0, -2), mean_recall, 'g-')
plt.plot(range(86, 0, -2), mean_f_score, 'y-')
ax.set_ylim(ymin=0)
plt.savefig('metrics_testsize_one_4_2.png')