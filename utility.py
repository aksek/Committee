# Aniela Kosek
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


def encode(df):
    X = df
    # Encode categorical columns
    categoricals = list(X.select_dtypes(include=['O']).columns)
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(X[categoricals])

    # create a data frame with the encoded data and append it
    if not (encoded.size == 0):
        train_ohe = pd.DataFrame(encoded, columns=np.hstack(encoder.categories_))
        X = pd.concat((X, train_ohe), axis=1).drop(categoricals, axis=1)
        X = X.dropna()
        return X
    else:
        return df


def calculate_metrics(model, X_test, y_test, *, encoded_X=None):
    pred = model.predict(X_test, X_encoded=encoded_X)
    cm = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0, average='macro')
    recall = recall_score(y_test, pred, zero_division=0, average='macro')
    f_score = f1_score(y_test, pred, zero_division=0, average='macro')
    # print('Confusion matrix:\n', cm)
    # print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1_score: {}'.format(
    #     acc, precision, recall, f_score))
    return acc, precision, recall, f_score
