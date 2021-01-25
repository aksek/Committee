# Author: Patryk Karbownik
import numpy as np
import pandas as pd


class Node:
    def __init__(self, attribute, the_class, children):
        self.attribute = attribute
        self.the_class = the_class
        self.children = children


def get_entropy(df):
    unique_values = pd.unique(df)
    entropy = 0
    for v in unique_values:
        incidence = df.value_counts()[v]
        incidence /= df.shape[0]
        entropy -= incidence * np.log2(incidence)

    return entropy


def get_attribute_with_the_highest_inf_gain(df, set_entropy):
    attributes = list(df.columns)
    inf_gain = 0
    attribute_name = ''

    for att in attributes:
        att_entropy = 0
        unique_values = df.att.unique()
        for u_v in unique_values:
            incidence = df.att.value_counts()[u_v]
            att_entropy = incidence * get_entropy(df.loc[df[att] == u_v])
        att_entropy /= df.shape[0]

        t_inf_gain = set_entropy - att_entropy
        if t_inf_gain > inf_gain:
            inf_gain = t_inf_gain
            attribute_name = att

    return attribute_name


# zakładam, że classes zawiera nazwe kolumny z klasami
def create_tree(classes, input_attributes, df):
    if df.shape[0] == 0:
        print('Error: The set is empty')
        return

    # sprawdź, czy elementy są tej samej klasy
    if len(df[classes].unique()) == 1:
        return Node(None, df[classes].unique()[0], None)
    # sprawdź, czy zbiór atrybutów jest pusty
    if len(input_attributes) == 0:
        return Node(None, df.classes.mode(), None)

    df_copy = df.copy()
    set_entropy = get_entropy(df_copy)
    attribute_d = get_attribute_with_the_highest_inf_gain(df, set_entropy)

    children_list = []
    unique_values = df.attribute_d.unique()

    for v in unique_values:
        # podziel zbior
        df_bis = df[df.attribute_d != v]
        input_attributes_bis = input_attributes.remove(attribute_d)
        children_list.append(create_tree(classes, input_attributes_bis, df_bis))

    return Node(attribute_d, unique_values, children_list)


def predict(tree, dF):
    if tree.children is None:
        return tree.the_class
    iteration = 0
    for i in tree.the_class:
        if i == dF.tree.attribute:
            return predict(tree.children[iteration], dF)
        iteration += 1
