from bloom_filter2 import BloomFilter
import pandas as pd

# Initialize Bloom Filter
# Capacity: ~1M interactions, error_rate: 0.01%
bloom = BloomFilter(
    max_elements=1000000, error_rate=0.01, filename=("data/interactions.bloom", -1)
)


# Populate from bloom_keys.csv
def populate_bloom_filter(bloom_keys_file):
    with open(bloom_keys_file, "r") as f:
        for key in f:
            bloom.add(key.strip())


# Check interaction
def has_interacted(user_id, movie_id):
    key = f"{int(user_id)}:{int(movie_id)}"
    return key in bloom


# Main execution
if __name__ == "__main__":
    # Populate Bloom filter
    populate_bloom_filter("data/bloom_keys.csv")

    # Test with sample interactions
    test_df = pd.read_parquet("data/test.parquet")
    sample = test_df.head(40)
    for _, row in sample.iterrows():
        user_id, movie_id = row["userId"], row["movieId"]
        print(
            f"User {user_id} interacted with movie {movie_id}: {has_interacted(user_id, movie_id)}"
        )
