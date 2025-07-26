import tensorflow as tf
import pandas as pd
import numpy as np

# Load preprocessed data
train_df = pd.read_parquet("data/train.parquet")
n_users = len(pd.read_csv("data/user_map.csv"))
n_movies = len(pd.read_csv("data/movie_map.csv"))

# Prepare data
user_ids = train_df["user_idx"].values
movie_ids = train_df["movie_idx"].values
interactions = train_df["interaction"].values


# Build NCF model
def build_ncf_model(n_users, n_movies, embedding_dim=64):
    user_input = tf.keras.Input(shape=(1,), name="user_input")
    movie_input = tf.keras.Input(shape=(1,), name="movie_input")

    user_embedding = tf.keras.layers.Embedding(
        n_users, embedding_dim, name="user_embedding"
    )(user_input)
    movie_embedding = tf.keras.layers.Embedding(
        n_movies, embedding_dim, name="movie_embedding"
    )(movie_input)

    user_flat = tf.keras.layers.Flatten()(user_embedding)
    movie_flat = tf.keras.layers.Flatten()(movie_embedding)

    concat = tf.keras.layers.Concatenate()([user_flat, movie_flat])
    dense = tf.keras.layers.Dense(128, activation="relu")(concat)
    dense = tf.keras.layers.Dense(64, activation="relu")(dense)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Train model
model = build_ncf_model(n_users, n_movies)
model.fit(
    [user_ids, movie_ids], interactions, batch_size=256, epochs=5, validation_split=0.1
)

# Save model
model.save("data/ncf_model")


# Generate recommendations
def get_recommendations(user_id, model, n_movies, movie_map, top_k=10):
    user_idx = pd.read_csv("data/user_map.csv", index_col=0).to_dict()["index"][user_id]
    user_array = np.array([user_idx] * n_movies)
    movie_array = np.arange(n_movies)
    predictions = model.predict([user_array, movie_array], batch_size=256)
    top_indices = np.argsort(predictions.flatten())[::-1][:top_k]
    movie_ids = [
        list(movie_map.keys())[list(movie_map.values()).index(idx)]
        for idx in top_indices
    ]
    return movie_ids


# Example
if __name__ == "__main__":
    movie_map = pd.read_csv("data/movie_map.csv", index_col=0).to_dict()["index"]
    print(
        "Recommendations for user 1:",
        get_recommendations(1, model, n_movies, movie_map),
    )
