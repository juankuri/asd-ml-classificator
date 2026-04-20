from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test, output_path: str = None):
    """Evalúa un modelo y devuelve métricas y probabilidades si están disponibles.

    Retorna: (accuracy, classification_report_str, roc_auc_or_None, y_pred, y_prob_or_None)
    """
    # Si recibimos una ruta al modelo, cargarla
    try:
        import joblib
    except Exception:
        joblib = None

    if isinstance(model, str) and joblib is not None:
        model = joblib.load(model)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        try:
            roc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc = None
    else:
        y_prob = None
        roc = None

    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0)

    if output_path:
        with open(output_path, "w") as f:
            f.write(f"accuracy: {accuracy:.4f}\n")
            if roc is not None:
                f.write(f"roc_auc: {roc:.4f}\n")
            f.write(report)

    return accuracy, report, roc, y_pred, y_prob