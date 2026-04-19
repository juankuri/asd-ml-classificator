import argparse
from src import train as train_mod
from src import evaluate as eval_mod


def run_pipeline(train_csv: str, test_csv: str, model_path: str, metrics_out: str):
    print('Entrenando...')
    model, _ = train_mod.train_model(train_csv, model_path)
    print('Entrenamiento terminado. Evaluando...')
    metrics = eval_mod.evaluate_model(model_path, test_csv, metrics_out)
    print('Evaluación completada. Métricas:')
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline simple AQ10 -> clasificación')
    parser.add_argument('--train', default='data/train.csv')
    parser.add_argument('--test', default='data/test.csv')
    parser.add_argument('--model', default='models/model.pkl')
    parser.add_argument('--out', default='outputs/metrics.txt')
    args = parser.parse_args()
    run_pipeline(args.train, args.test, args.model, args.out)
