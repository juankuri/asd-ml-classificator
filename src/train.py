import argparse
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data


def train_model(train_csv: str, model_path: str):
    X, y = load_data(train_csv)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    return pipeline, (X_val, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrena un modelo simple.')
    parser.add_argument('--train', required=True, help='CSV de entrenamiento')
    parser.add_argument('--model', default='models/model.pkl', help='Ruta para guardar el modelo')
    args = parser.parse_args()
    model, _ = train_model(args.train, args.model)
    print(f'Modelo guardado en {args.model}')
