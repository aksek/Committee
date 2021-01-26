# Author: Patryk Karbownik
from math import log2

import numpy as np
import pandas as pd
from scipy.linalg.matfuncs import eps


class Node:
    def __init__(self, attribute, the_class, children):
        self.attribute = attribute
        self.the_class = the_class
        self.children = children


def get_entropy(df):
    unique_values = pd.unique(df['play'])
    entropy = 0
    for v in unique_values:
        incidence = df['play'].value_counts()[v]
        incidence /= len(df['play'])
        entropy -= incidence * np.log2(incidence)

    return entropy


def get_attribute_with_the_highest_inf_gain(df, set_entropy, attributes):
    inf_gain = 0
    attribute_name = ''
    classes = df['play'].unique()
    for att in attributes:
        t_inf_gain = 0
        att_entropy = 0
        unique_values = df[att].unique()
        for u_v in unique_values:
            all_with_the_unique_value = df[df[att] == u_v].shape[0] #zlicza ile jest wszystkich z ta wartoscia unikalna
            temp_entropy = 0
            for c in classes:
                temp_amount = df[(df[att] == u_v) & (df['play'] == c)].shape[0]
                if temp_amount != 0:
                    temp_value = log2(temp_amount/all_with_the_unique_value)
                    temp_value *= temp_amount/all_with_the_unique_value
                    temp_entropy -= temp_value

            temp_entropy *= all_with_the_unique_value
            temp_entropy /= df.shape[0]
            att_entropy += temp_entropy
        #att_entropy /= df.shape[0]
        #att_entropy = abs(att_entropy)
        t_inf_gain = set_entropy - att_entropy
        if t_inf_gain > inf_gain:
            inf_gain = t_inf_gain
            attribute_name = att

    return attribute_name


# zakładam, że classes zawiera nazwe kolumny z klasami
def create_tree(classes, input_attributes, df):
    if df.shape[0] == 0:
        print('Error: The set is empty')
        return None

    # sprawdź, czy elementy są tej samej klasy
    if len(df[classes].unique()) == 1:
        return Node(None, df[classes].unique()[0], None)
    # sprawdź, czy zbiór atrybutów jest pusty
    if len(input_attributes) == 0:
        return Node(None, df[classes].mode(), None)

    df_copy = df.copy()

    se = get_entropy(df)
    attribute_d = get_attribute_with_the_highest_inf_gain(df_copy, se, input_attributes)
    children_list = []
    unique_values = df_copy[attribute_d].unique()
    input_attributes.remove(attribute_d)
    for v in unique_values:
        # podziel zbior
        df_bis = df_copy[df_copy[attribute_d] == v]
        children_list.append(create_tree(classes, input_attributes, df_bis))

    return Node(attribute_d, unique_values, children_list)


def predict(tree, dF):
    if tree.children is None:
        return tree.the_class
    iteration = 0
    for i in tree.the_class:
        if i == dF.tree.attribute:
            return predict(tree.children[iteration], dF)
        iteration += 1


def main():
    df = pd.read_csv("tennis.csv")
    attr = list(df.columns.values)

    attr.remove('play')
    attr.remove('day')
    print(attr)
    tree = create_tree('play', attr, df)
    print("cosik")


if __name__ == '__main__':
    main()
