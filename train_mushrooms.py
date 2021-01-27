# Aniela Kosek
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

import committee as cmt
from decisionTree import DecisionTree


def calculate_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    f_score = f1_score(y_test, pred, average='micro')
    # print('Confusion matrix:\n', cm)
    # print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1_score: {}'.format(
    #     acc, precision, recall, f_score))
    return acc, precision, recall, f_score


# load data
df = pd.read_csv("mushrooms.csv")

# split columns
X = df.drop("class", axis=1)
y = df["class"].values

# Encode categorical columns
categoricals = list(X.select_dtypes(include=['O']).columns)
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(X[categoricals])

# create a data frame with the encoded data and append it
train_ohe = pd.DataFrame(encoded, columns=np.hstack(encoder.categories_))
X = pd.concat((X, train_ohe), axis=1).drop(categoricals, axis=1)

accuracy = []
precision = []
recall = []
f_score = []

for i in range(25):
    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # initialize classifiers
    clf1 = DecisionTree()
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    clf4 = GaussianNB()

    # initialize and fit the voting classifier
    eclf = cmt.CommitteeClassifier([clf2, clf3, clf4])
    eclf = eclf.fit(X_train, y_train)

    # test the result
    acc, prec, rec, f = calculate_metrics(eclf, X_test, y_test)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f_score.append(f)
    print(str(i * 4) + "%")

print("Accuracy: " + str(mean(accuracy)))
print("Precision: " + str(mean(precision)))
print("Recall: " + str(mean(recall)))
print("F_score: " + str(mean(f_score)))
