import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # eliminar columnas innecesarias
    columns_to_drop = ["ID", "age_desc", "result"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    df = df.replace("?", np.nan)

    return df


def split_features_target(df: pd.DataFrame):
    y = df["Class/ASD"]
    X = df.drop("Class/ASD", axis=1)
    return X, y


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(X)
    return X.astype(int)


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    return X.fillna(X.mean())