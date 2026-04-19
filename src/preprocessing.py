from typing import Tuple
import pandas as pd


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga CSV con columna `label` y convierte respuestas yes/no a 1/0.

    Espera: archivo CSV donde la columna `label` contiene 0/1 y
    las demás columnas son respuestas ("yes"/"no" u 0/1).
    """
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError("El CSV debe contener la columna 'label' con 0/1")
    y = df['label'].astype(int)
    X = df.drop(columns=['label'])

    def map_bool(v):
        if pd.isna(v):
            return 0
        s = str(v).strip().lower()
        if s in ('yes', 'y', '1', 'true', 't'):
            return 1
        if s in ('no', 'n', '0', 'false', 'f'):
            return 0
        try:
            return float(v)
        except Exception:
            return 0

    X = X.applymap(map_bool)
    X = X.fillna(0)
    return X, y
