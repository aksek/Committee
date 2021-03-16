# Aniela Kosek

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
        return X
    else:
        return df