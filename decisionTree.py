# Author: Patryk Karbownik
from math import log2
from typing import List, Any

import numpy as np
import pandas as pd


class DecisionTree:
    root = None

    class Node:
        def __init__(self, attribute, the_class, children, is_string):
            self.attribute = attribute
            self.the_class = the_class
            self.children = children
            self.is_string = is_string

    def get_entropy(self, df, classes):
        unique_values = pd.unique(df[classes])
        entropy = 0
        for v in unique_values:
            incidence = df[classes].value_counts()[v]
            incidence /= len(df[classes])
            entropy -= incidence * np.log2(incidence)

        return entropy

    def get_attribute_with_the_highest_inf_gain(self, df, set_entropy, attributes, classes_t, sp):
        inf_gain = 0
        attribute_name = ''
        classes = df[classes_t].unique()
        split_value = -1
        for att in attributes:
            t_inf_gain = 0
            att_entropy = 1
            unique_values = df[att].unique()
            if isinstance(unique_values[0], str):
                # unique_values = df[att].unique()
                att_entropy = 0
                for u_v in unique_values:
                    all_with_the_unique_value = df[df[att] == u_v].shape[
                        0]  # zlicza ile jest wszystkich z ta wartoscia unikalna
                    temp_entropy = 0
                    for c in classes:
                        temp_amount = df[(df[att] == u_v) & (df[classes_t] == c)].shape[0]
                        if temp_amount != 0:
                            temp_value = log2(temp_amount / all_with_the_unique_value)
                            temp_value *= temp_amount / all_with_the_unique_value
                            temp_entropy -= temp_value

                    temp_entropy *= all_with_the_unique_value
                    temp_entropy /= df.shape[0]
                    att_entropy += temp_entropy
            else:
                split_value = DecisionTree.find_split_value(self, df, set_entropy, att, classes_t)
                set_a_size = df[df[att] < split_value].shape[0]
                set_b_size = df[df[att] >= split_value].shape[0]
                entropy_in_a = 0
                entropy_in_b = 0
                for c in classes:
                    temp_amount_for_a = df[(df[att] < split_value) & (df[classes_t] == c)].shape[0]
                    temp_amount_for_b = df[(df[att] >= split_value) & (df[classes_t] == c)].shape[0]
                    if temp_amount_for_a != 0:
                        temp_value = log2(temp_amount_for_a / set_a_size)
                        temp_value *= temp_amount_for_a / set_a_size
                        entropy_in_a -= temp_value
                    if temp_amount_for_b != 0:
                        temp_value = log2(temp_amount_for_b / set_b_size)
                        temp_value *= temp_amount_for_b / set_b_size
                        entropy_in_b -= temp_value
                entropy_in_a *= set_a_size
                entropy_in_a /= df.shape[0]

                entropy_in_b *= set_b_size
                entropy_in_b /= df.shape[0]

                att_entropy = entropy_in_a + entropy_in_b

            t_inf_gain = set_entropy - att_entropy
            if t_inf_gain > inf_gain:
                inf_gain = t_inf_gain
                attribute_name = att
                if isinstance(unique_values[0], str):
                    sp[0] = -1
                else:
                    sp[0] = split_value

        return attribute_name

    def find_split_value(self, df, set_entropy, attribute, classes):
        inf_gain = 0
        unique_values = df[attribute].unique()
        split_value = unique_values[0]
        classes_values = df[classes].unique()
        for uv in unique_values:
            # zbior a to zbior, dla ktorego wartosci atrybutu sa mniejsze od aktualnego uv
            # zbior b to zbior, dla ktorego wartosci atrybutu sa równe lub wieksze od aktualnego uv
            set_a_size = df[df[attribute] < uv].shape[0]
            set_b_size = df[df[attribute] >= uv].shape[0]
            entropy_in_a = 0
            entropy_in_b = 0
            for c in classes_values:
                temp_amount_for_a = df[(df[attribute] < uv) & (df[classes] == c)].shape[0]
                temp_amount_for_b = df[(df[attribute] >= uv) & (df[classes] == c)].shape[0]
                if temp_amount_for_a != 0:
                    temp_value = log2(temp_amount_for_a / set_a_size)
                    temp_value *= temp_amount_for_a / set_a_size
                    entropy_in_a -= temp_value
                if temp_amount_for_b != 0:
                    temp_value = log2(temp_amount_for_b / set_b_size)
                    temp_value *= temp_amount_for_b / set_b_size
                    entropy_in_b -= temp_value
            entropy_in_a *= set_a_size
            entropy_in_a /= df.shape[0]

            entropy_in_b *= set_b_size
            entropy_in_b /= df.shape[0]

            entropy = entropy_in_a + entropy_in_b
            if set_entropy - entropy > inf_gain:
                inf_gain = set_entropy - entropy
                split_value = uv

        return split_value

    # zakładam, że classes zawiera nazwe kolumny z klasami
    def create_tree(self, classes, input_attributes, df):
        if df.shape[0] == 0:
            print('Error: The set is empty')
            return None

        # sprawdź, czy elementy są tej samej klasy
        if len(df[classes].unique()) == 1:
            return DecisionTree.Node(None, df[classes].unique()[0], None, isinstance(df[classes].unique()[0], str))
        # sprawdź, czy zbiór atrybutów jest pusty
        if len(input_attributes) == 0:
            return DecisionTree.Node(None, df[classes].mode(), None, isinstance(df[classes].mode(), str))

        df_copy = df.copy()

        se = DecisionTree.get_entropy(self, df, classes)
        is_string = [-1]
        attribute_d = DecisionTree.get_attribute_with_the_highest_inf_gain(self, df_copy, se, input_attributes, classes, is_string)
        children_list = []
        input_attributes.remove(attribute_d)
        if is_string[0] == -1:
            unique_values = df_copy[attribute_d].unique()
            for v in unique_values:
                # podziel zbior
                df_bis = df_copy[df_copy[attribute_d] == v]
                children_list.append(DecisionTree.create_tree(self, classes, input_attributes, df_bis))
            return DecisionTree.Node(attribute_d, unique_values, children_list, isinstance(df[attribute_d].unique()[0], str))
        else:
            split_value = is_string[0]
            df_a = df_copy[df_copy[attribute_d] < split_value]
            df_b = df_copy[df_copy[attribute_d] >= split_value]
            children_list.append(DecisionTree.create_tree(self, classes, input_attributes, df_a))
            children_list.append(DecisionTree.create_tree(self, classes, input_attributes, df_b))
            return DecisionTree.Node(attribute_d, split_value, children_list, False)

    def innerPredict(self, attributes, X):
        y = self.root
        while len(y.children) != 0:
            i = -1
            if y.is_string:
                i = 0
                for c in y.the_class:
                    if X[y.attribute] == y.the_class[i]:
                        y = y.children[i]
                        break
                    i +=1
            else:
                split_value = y.the_class
                if X[y.attribute] < split_value:
                    y = y.children[0]
                else:
                    y = y.children[1]
        return y.the_class

    def fit(self, X, y):
        attributes: List[Any] = list(X.columns.values)
        df = X.copy()
        df.insert(0, "classes", y, False)
        self.root = DecisionTree.create_tree(self, "classes", attributes, df)

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("The estimator was not fitted yet")
        else:
            y = []
            for x in X:
                y.append(self.innerPredict(list(X.columns.values), x))
            return y

def main():
    df = pd.read_csv("citrus.csv")
    attr = list(df.columns.values)

    attr.remove('name')
    print(attr)
    #        tree = create_tree('name', attr, df)
    print("cosik")


if __name__ == '__main__':
    main()
