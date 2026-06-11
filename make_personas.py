"""Step 1 — assign a stable, realistic pseudo-name to every MovieLens/Goodbooks user_id.

The app's "login" is a persona picker, so each real user_id needs a human-friendly
name to show in the dropdown. Names are generated deterministically (fixed seed,
user_ids processed in sorted order) so the same user_id always maps to the same name
on any machine and across runs. Output is cached to user_personas.csv and committed,
so the app reads names instead of regenerating them.

Run once:  python make_personas.py
"""

import pandas as pd
from faker import Faker

RATINGS_PATH = "Ratings.csv"
OUT_PATH = "user_personas.csv"
SEED = 6604  # course number, just a fixed constant for reproducibility

fake = Faker("en_US")
Faker.seed(SEED)

user_ids = sorted(pd.read_csv(RATINGS_PATH)["user_id"].unique())

# Generate one unique full name per user_id. Faker can repeat names, so we
# keep drawing until we get one we haven't used, guaranteeing a 1:1 mapping.
seen, rows = set(), []
for uid in user_ids:
    name = fake.name()
    while name in seen:
        name = fake.name()
    seen.add(name)
    rows.append((int(uid), name))

personas = pd.DataFrame(rows, columns=["user_id", "name"])
personas.to_csv(OUT_PATH, index=False)

print(f"Wrote {len(personas):,} personas -> {OUT_PATH}")
print(f"Unique names: {personas['name'].nunique():,} (should equal {len(personas):,})")
print(personas.head(8).to_string(index=False))
