# Author: Patryk Karbownik
import numpy as np
import pandas as pd


class Node:
    qualification = 0

    def __init__(self, q):
        self.qualification = q


def get_entropy(df):
    unique_values = pd.unique(df)
    entropy = 0;
    for v in unique_values:
        incidence = df.value_counts()[v]
        incidence /= df.shape[0]
        entropy -= incidence * np.log2(incidence)

    return entropy

# def create_tree(classes, input_attributes, Node = None):
# sprawdź, czy zbiór
# sprawdź, czy elementy są tej samej klasy

# sprawdź, czy zbiór atrybutów jest pusty
