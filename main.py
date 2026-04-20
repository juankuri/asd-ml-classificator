import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    encode_features,
    handle_missing_values
)
from src.train import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
)
from src.evaluate import evaluate_model, split_data

from sklearn.metrics import classification_report, roc_curve


def main():
    # === 0. Crear carpetas necesarias ===
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # === 1. Cargar y preparar datos ===
    df = load_data("data/train.csv")
    df = clean_data(df)

    X, y = split_features_target(df)
    X = encode_features(X)
    X = handle_missing_values(X)

    # Guardar columnas para inferencia futura (Streamlit / CLI)
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    # === 2. Split ===
    X_train, X_test, y_train, y_test = split_data(X, y)

    # === 3. Entrenamiento ===
    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    gb = train_gradient_boosting(X_train, y_train)

    # === 4. Evaluación ===
    acc_lr, rep_lr, roc_lr, y_pred_lr, y_prob_lr = evaluate_model(lr, X_test, y_test)
    acc_rf, rep_rf, roc_rf, y_pred_rf, y_prob_rf = evaluate_model(rf, X_test, y_test)
    acc_gb, rep_gb, roc_gb, y_pred_gb, y_prob_gb = evaluate_model(gb, X_test, y_test)

    print("=== Logistic Regression ===")
    print("Accuracy:", acc_lr)
    print("ROC-AUC:", roc_lr)
    print(rep_lr)

    print("\n=== Random Forest ===")
    print("Accuracy:", acc_rf)
    print("ROC-AUC:", roc_rf)
    print(rep_rf)

    print("\n=== Gradient Boosting ===")
    print("Accuracy:", acc_gb)
    print("ROC-AUC:", roc_gb)
    print(rep_gb)

    # === 5. Ajuste de threshold (enfocado a salud) ===
    threshold = 0.3
    if y_prob_lr is not None:
        y_pred_custom = (np.array(y_prob_lr) > threshold).astype(int)

        print("\n=== Logistic Regression (Threshold 0.3) ===")
        print(classification_report(y_test, y_pred_custom, zero_division=0))

        # Guardamos threshold para uso en app
        joblib.dump(threshold, "models/threshold.pkl")

    # === 6. Curva ROC ===
    plt.figure()

    for name, y_prob in [
        ("Logistic", y_prob_lr),
        ("Random Forest", y_prob_rf),
        ("Gradient Boosting", y_prob_gb),
    ]:
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=name)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig("outputs/roc_comparison.png")
    plt.close()

    # === 7. Selección automática del mejor modelo ===
    results = {
        "Logistic": roc_lr,
        "Random Forest": roc_rf,
        "Gradient Boosting": roc_gb,
    }

    best_model_name = max(
        results,
        key=lambda k: results[k] if results[k] is not None else 0
    )

    print("\nBest model based on ROC-AUC:", best_model_name)

    best_model = {
        "Logistic": lr,
        "Random Forest": rf,
        "Gradient Boosting": gb,
    }[best_model_name]

    # Guardar modelo final
    joblib.dump(best_model, "models/final_model.pkl")

    # Guardar nombre del modelo (para mostrar en app)
    joblib.dump(best_model_name, "models/model_name.pkl")

    print("Saved best model to models/final_model.pkl")


if __name__ == "__main__":
    main()