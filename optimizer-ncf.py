import optuna
import tensorflow as tf
import pandas as pd
import numpy as np
from ncfModel import build_ncf_model

# Load data
train_df = pd.read_parquet("data/train.parquet")
val_df = pd.read_parquet("data/val.parquet")
user_map = pd.read_csv("data/user_map.csv", index_col=0)["index"].to_dict()
movie_map = pd.read_csv("data/movie_map.csv", index_col=0)["index"].to_dict()
n_users = len(user_map)
n_movies = len(movie_map)

# Prepare training data
user_ids = train_df["user_idx"].values
movie_ids = train_df["movie_idx"].values
interactions = train_df["interaction"].values


# Optuna objective function
def objective(trial):
    embedding_dim = trial.suggest_int("embedding_dim", 32, 128)
    dense_units = trial.suggest_int("dense_units", 64, 256)

    model = build_ncf_model(n_users, n_movies, embedding_dim)
    model.fit(
        [user_ids, movie_ids],
        interactions,
        batch_size=256,
        epochs=3,
        validation_split=0.1,
        verbose=0,
    )

    # Evaluate precision@10
    user_id = 1
    user_idx = user_map.get(user_id, None)
    if user_idx is None:
        return 0.0
    user_array = np.array([user_idx] * n_movies)
    movie_array = np.arange(n_movies)
    predictions = model.predict([user_array, movie_array], batch_size=256)
    top_indices = np.argsort(predictions.flatten())[::-1][:10]
    relevant = set(val_df[val_df["user_idx"] == user_idx]["movie_idx"])
    precision = len(set(top_indices) & relevant) / 10
    return precision


# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Reduced trials for speed

# Retrain with best params
best_params = study.best_params
best_model = build_ncf_model(n_users, n_movies, best_params["embedding_dim"])
best_model.fit(
    [user_ids, movie_ids], interactions, batch_size=256, epochs=5, validation_split=0.1
)

# Quantize model
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()
with open("data/ncf_optimized.tflite", "wb") as f:
    f.write(tflite_model)


# Evaluate optimized model
def evaluate_precision(model, val_df, n_movies, top_k=10):
    precisions = []
    for user_idx in val_df["user_idx"].unique()[:100]:  # Sample 100 users
        user_array = np.array([user_idx] * n_movies)
        movie_array = np.arange(n_movies)
        predictions = model.predict([user_array, movie_array], batch_size=256)
        top_indices = np.argsort(predictions.flatten())[::-1][:top_k]
        relevant = set(val_df[val_df["user_idx"] == user_idx]["movie_idx"])
        precision = len(set(top_indices) & relevant) / top_k
        precisions.append(precision)
    return np.mean(precisions)


print(f"Precision@10: {evaluate_precision(best_model, val_df, n_movies)}")
