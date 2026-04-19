import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocessing import load_data


def evaluate_model(model_path: str, test_csv: str, output_path: str) -> dict:
    model = joblib.load(model_path)
    X, y = load_data(test_csv)
    preds = model.predict(X)
    metrics = {
        'accuracy': float(accuracy_score(y, preds)),
        'precision': float(precision_score(y, preds, zero_division=0)),
        'recall': float(recall_score(y, preds, zero_division=0)),
        'f1': float(f1_score(y, preds, zero_division=0)),
    }
    with open(output_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evalúa un modelo guardado')
    parser.add_argument('--model', default='models/model.pkl')
    parser.add_argument('--test', required=True)
    parser.add_argument('--out', default='outputs/metrics.txt')
    args = parser.parse_args()
    m = evaluate_model(args.model, args.test, args.out)
    print('Métricas:', m)
