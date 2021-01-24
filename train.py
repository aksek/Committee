# Aniela Kosek

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import committee as cmt


def calculate_metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, pos_label='grapefruit')
    recall = recall_score(y_test, pred, pos_label='grapefruit')
    f_score = f1_score(y_test, pred, pos_label='grapefruit')
    print('Confusion matrix:\n', cm)
    print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1_score: {}'.format(
        acc, precision, recall, f_score))
    return cm


# load data
df = pd.read_csv("citrus.csv")

# split columns
X = df.drop("name", axis=1)
y = df["name"].values

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=71830)

# initialize classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)

# initialize and fit the voting classifier
eclf = cmt.CommitteeClassifier([clf1, clf2, clf3])
eclf = eclf.fit(X_train, y_train)

# test the result
calculate_metrics(eclf, X_test, y_test)
