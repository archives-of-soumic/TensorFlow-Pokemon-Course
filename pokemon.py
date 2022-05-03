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


if __name__ == '__main__':
    # read csv, and create dataFrame df
    df = pd.read_csv("./pokemon_alopez247.csv")
    print(df.columns)
    df["isLegendary"] = df["isLegendary"].astype(int)  # converts bool into int
    df = dummy_creation(_df=df, dummy_categories=['Egg_Group_1', 'Body_Style', 'Color', 'Type_1', 'Type_2'])
