from src.preprocessing import *
from src.train import *
from src.evaluate import *

# 1. Cargar y preparar datos
df = load_data("data/train.csv")
df = clean_data(df)

X, y = split_features_target(df)
X = encode_features(X)
X = handle_missing_values(X)

# 2. Split
X_train, X_test, y_train, y_test = split_data(X, y)

# 3. Modelos
lr = train_logistic_regression(X_train, y_train)
rf = train_random_forest(X_train, y_train)

# 4. Evaluación
acc_lr, rep_lr = evaluate_model(lr, X_test, y_test)
acc_rf, rep_rf = evaluate_model(rf, X_test, y_test)

print("=== Logistic Regression ===")
print("Accuracy:", acc_lr)
print(rep_lr)

print("\n=== Random Forest ===")
print("Accuracy:", acc_rf)
print(rep_rf)