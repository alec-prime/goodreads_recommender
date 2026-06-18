# save_model.py
"""Train and pickle the deployed CF model (ubcf_pear) for the Streamlit app.

ubcf_pear == KNNBasic(k=50, pearson, user-based) — the model chosen in the notebook
bake-off (project_2-source.ipynb §2). Fit on the FULL ratings (no held-out split)
for deployment, then pickled so app.py can predict at runtime without re-fitting.

Run once. Re-run if you rebuild the venv: a pickled surprise model is coupled to the
scikit-surprise/numpy versions that created it, so a version drift can break unpickling.

    source ../ai_venv/bin/activate && python save_model.py
"""

import pickle

import pandas as pd
from surprise import Dataset, KNNBasic, Reader

RATINGS_PATH = "Ratings.csv"
MODEL_PATH = "ubcf_pear.pkl"

ratings = pd.read_csv(RATINGS_PATH)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
trainset = data.build_full_trainset()

# Same config as the deployed model in the notebook bake-off.
model = KNNBasic(
    k=50, sim_options={"name": "pearson", "user_based": True}, verbose=False
)
model.fit(trainset)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Saved {MODEL_PATH}: {trainset.n_users} users, {trainset.n_items} items")
