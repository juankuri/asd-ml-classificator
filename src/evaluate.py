from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report