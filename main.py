from src.preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    encode_features,
    handle_missing_values
)

df = load_data("data/train.csv")
df = clean_data(df)

X, y = split_features_target(df)

X = encode_features(X)
X = handle_missing_values(X)

print("Shape X:", X.shape)
print("Shape y:", y.shape)
print(X.head(11))