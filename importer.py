import pandas as pd
import numpy as np

def import_data():
    try:
        df_train = pd.read_csv('data/df_train.csv', index_col='Unnamed: 0')
        df_test = pd.read_csv('data/df_test.csv', index_col='Unnamed: 0')
    except:
        df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
        df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

        df_train.to_csv('data/df_train.csv')
        df_test.to_csv('data/df_test.csv')

    subset = df_train.columns.tolist()
    subset.remove('selling_price')
    df_train = df_train.drop_duplicates(subset=subset)

    X_train = df_train.drop('selling_price', axis=1)
    X_test = df_test.drop('selling_price', axis=1)

    y_train = df_train['selling_price']
    y_test = df_test['selling_price']

    print("Train data shape:", X_train.shape, y_train.shape)
    print("Test data shape: ", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test