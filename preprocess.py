import pandas as pd
import numpy as np
import json

# Load data
movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")

# Validate data
print("Missing values:", ratings_df.isnull().sum())
invalid_movies = ratings_df[~ratings_df["movieId"].isin(movies_df["movieId"])]
if not invalid_movies.empty:
    print(f"Removing {len(invalid_movies)} invalid movie IDs")
    ratings_df = ratings_df[ratings_df["movieId"].isin(movies_df["movieId"])]

# Handle duplicates
ratings_df = ratings_df.sort_values("timestamp").drop_duplicates(
    subset=["userId", "movieId"], keep="last"
)

# Filter sparse users and movies
min_user_ratings, min_movie_ratings = 5, 10
user_counts = ratings_df["userId"].value_counts()
movie_counts = ratings_df["movieId"].value_counts()
ratings_df = ratings_df[
    (ratings_df["userId"].isin(user_counts[user_counts >= min_user_ratings].index))
    & (
        ratings_df["movieId"].isin(
            movie_counts[movie_counts >= min_movie_ratings].index
        )
    )
]

# Convert to implicit feedback
ratings_df["interaction"] = (ratings_df["rating"] >= 3).astype(int)

# Normalize timestamps
ratings_df["timestamp_norm"] = (
    ratings_df["timestamp"] - ratings_df["timestamp"].min()
) / (ratings_df["timestamp"].max() - ratings_df["timestamp"].min())

# Encode genres
genres = movies_df["genres"].str.get_dummies("|")
movies_df = pd.concat([movies_df[["movieId", "title"]], genres], axis=1)

# Extract year
movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)").astype(float)


# Generate negative samples
def generate_negative_samples(ratings_df, movies_df, neg_ratio=4):
    interactions = set(zip(ratings_df["userId"], ratings_df["movieId"]))
    negative_samples = []
    all_movies = set(movies_df["movieId"])
    for user_id in ratings_df["userId"].unique():
        watched = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
        unwatched = list(all_movies - watched)
        neg_samples = np.random.choice(
            unwatched, size=len(watched) * neg_ratio, replace=False
        )
        for movie_id in neg_samples:
            negative_samples.append(
                {"userId": user_id, "movieId": movie_id, "interaction": 0}
            )
    return pd.DataFrame(negative_samples)


ratings_df = pd.concat(
    [ratings_df, generate_negative_samples(ratings_df, movies_df)], ignore_index=True
)

# Create index mappings
user_map = {uid: idx for idx, uid in enumerate(ratings_df["userId"].unique())}
movie_map = {mid: idx for idx, mid in enumerate(ratings_df["movieId"].unique())}
pd.DataFrame.from_dict(user_map, orient="index", columns=["index"]).to_csv(
    "data/user_map.csv"
)
pd.DataFrame.from_dict(movie_map, orient="index", columns=["index"]).to_csv(
    "data/movie_map.csv"
)
ratings_df["user_idx"] = ratings_df["userId"].map(user_map)
ratings_df["movie_idx"] = ratings_df["movieId"].map(movie_map)

# Sort and split data
ratings_df = ratings_df.sort_values("timestamp")
n = len(ratings_df)
train_df = ratings_df.iloc[: int(0.8 * n)]
val_df = ratings_df.iloc[int(0.8 * n) : int(0.9 * n)]
test_df = ratings_df.iloc[int(0.9 * n) :].copy()

# Save splits as Parquet
train_df.to_parquet("data/train.parquet")
val_df.to_parquet("data/val.parquet")
test_df.to_parquet("data/test.parquet")

# Prepare Bloom filter keys
bloom_keys = train_df[train_df["interaction"] == 1][["userId", "movieId"]].apply(
    lambda x: f"{int(x['userId'])}:{int(x['movieId'])}", axis=1
)
bloom_keys.to_csv("data/bloom_keys.csv", index=False, header=False)


# Prepare Kafka messages
def safe_json(x):
    if pd.isnull(x["timestamp"]):
        return None
    return json.dumps(
        {
            "userId": int(x["userId"]),
            "movieId": int(x["movieId"]),
            "interaction": int(x["interaction"]),
            "timestamp": int(x["timestamp"]),
        }
    )


test_df["kafka_message"] = test_df.apply(safe_json, axis=1)
test_df = test_df.dropna(subset=["kafka_message"])
# Print stats
print(
    f"Users: {ratings_df['userId'].nunique()}, Movies: {ratings_df['movieId'].nunique()}"
)
print(
    f"Interactions: {len(ratings_df)}, Sparsity: {1 - len(ratings_df) / (ratings_df['userId'].nunique() * ratings_df['movieId'].nunique()):.4f}"
)
