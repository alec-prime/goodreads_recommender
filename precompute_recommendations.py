"""Phase 1 — precompute every user's Top-10 + candidate pool offline.

The app reads the resulting recommendations.pkl, so scikit-surprise is NOT a
runtime dependency. Ratings are static during app use, so there is no reason to
score the model live. Mirrors the notebook's ubcf_pear config and MIN_RATINGS=20.

Run once:  python precompute_recommendations.py
"""

import pickle
import pandas as pd
from surprise import KNNBasic, Dataset, Reader

RATINGS_PATH = "Ratings.csv"
OUT_PATH = "recommendations.pkl"
MIN_RATINGS = 20  # popularity floor for candidate eligibility (notebook EDA)
TOP_N = 10  # displayed list size
POOL_SIZE = 40  # larger candidate pool handed to the LLM re-ranker

ratings = pd.read_csv(RATINGS_PATH)

# Train ubcf_pear on ALL ratings (deployment uses every observation).
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
trainset = data.build_full_trainset()
model = KNNBasic(
    k=50, sim_options={"name": "pearson", "user_based": True}, verbose=False
)
model.fit(trainset)

# Eligible (popular) books: >= MIN_RATINGS ratings across our users.
counts = ratings["book_id"].value_counts()
popular_books = set(counts[counts >= MIN_RATINGS].index)
print(f"Candidate pool source: {len(popular_books):,} popular books")

seen_by_user = ratings.groupby("user_id")["book_id"].apply(set).to_dict()
user_ids = sorted(ratings["user_id"].unique())

recommendations = {}
for i, uid in enumerate(user_ids):
    candidates = popular_books - seen_by_user.get(uid, set())
    scored = [(int(bid), float(model.predict(uid, bid).est)) for bid in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    recommendations[int(uid)] = {
        "top10": [bid for bid, _ in scored[:TOP_N]],
        "pool": scored[:POOL_SIZE],
    }
    if (i + 1) % 200 == 0:
        print(f"  scored {i + 1:,}/{len(user_ids):,} users")

with open(OUT_PATH, "wb") as f:
    pickle.dump(recommendations, f)

print(f"Wrote {len(recommendations):,} users -> {OUT_PATH}")
