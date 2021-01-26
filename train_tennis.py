# Aniela Kosek
from statistics import mean

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import committee as cmt
from decisionTree import DecisionTree


def calculate_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, pos_label=1)
    recall = recall_score(y_test, pred, pos_label=1)
    f_score = f1_score(y_test, pred, pos_label=1)
    # print('Confusion matrix:\n', cm)
    # print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1_score: {}'.format(
    #     acc, precision, recall, f_score))
    return acc, precision, recall, f_score


# load data
df = pd.read_csv("play_tennis.csv")

# drop irrelevant columns
df = df.drop("day", axis=1)

# Encode categorical columns
categoricals = list(df.select_dtypes(include=['O']).columns)
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[categoricals])

# create a data frame with the encoded data and append it
train_ohe = pd.DataFrame(encoded, columns=np.hstack(encoder.categories_))
df = pd.concat((df, train_ohe), axis=1).drop(categoricals, axis=1)

# split columns and drop irrelevant columns
X = df.drop("No", axis=1)
X = X.drop("Yes", axis=1)
y = df["Yes"].values

accuracy = []
precision = []
recall = []
f_score = []

for i in range(25):
    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # initialize classifiers
    clf1 = DecisionTree()
    clf2 = KNeighborsClassifier(n_neighbors=2)
    # clf3 = SVC(kernel='rbf', probability=True)
    # clf4 = GaussianNB()

    # initialize and fit the voting classifier
    eclf = cmt.CommitteeClassifier([clf1, clf2])
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
