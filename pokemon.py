import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def dummy_creation(_df, dummy_categories):  # one hot vector I think!
    for i in dummy_categories:
        df_dummy = pd.get_dummies(_df[i])
        _df = pd.concat([_df, df_dummy], axis=1)
        _df = df.drop(i, axis=1)
    return _df


""" 
A stupid splitter that 
drops one column! 
"""


# noinspection PyRedundantParentheses
def train_test_splitter(DataFrame, column):
    df_train1 = DataFrame.loc[df[column] != 1]
    df_test1 = DataFrame.loc[df[column] == 1]

    df_train1 = df_train1.drop(column, axis=1)
    df_test1 = df_test1.drop(column, axis=1)

    return (df_train1, df_test1)


# noinspection PyRedundantParentheses
def label_delineator(df_train, df_test, label):  # x_train, y_train, x_test, y_test
    train_data = df_train.drop(label, axis=1).values  # x_train
    train_labels = df_train[label].values  # y_train

    test_data = df_test.drop(label, axis=1).values  # x_test
    test_labels = df_test[label].values  # y_test

    return (train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    # read csv, and create dataFrame df
    df = pd.read_csv("./pokemon_alopez247.csv")
    print(df.columns)
    df["isLegendary"] = df["isLegendary"].astype(int)  # converts bool into int
    df = dummy_creation(_df=df, dummy_categories=['Egg_Group_1', 'Body_Style', 'Color', 'Type_1', 'Type_2'])

    df_train, df_test = train_test_splitter(DataFrame=df, column="Generation")
    train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, "isLegendary")
