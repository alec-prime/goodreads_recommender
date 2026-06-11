"""Step 1.5 — upgrade all book-cover links in the catalog to higher resolution.

The Goodreads cover URLs shipped in Books.csv use the medium ('m') size suffix
(~98px wide). The same Amazon-backed CDN serves a larger ('l') variant (~315px
wide, ~3x resolution) at an otherwise-identical URL — verified live on a 40-cover
sample. We rewrite image_url in place to the 'l' variant so the app reads
ready-to-use high-res links with no per-render work.

Only real Goodreads cover URLs (.../books/{photo_id}m/{id}.jpg) are rewritten.
"nophoto" placeholder URLs don't match the pattern and are left untouched.
The change is reversible (l -> m) and small_image_url is preserved.

Run once:  python upgrade_cover_resolution.py
"""

import re
import pandas as pd

CATALOG = "Books.csv"
# Matches the size suffix in .../books/<photo_id>m/<book_id>.jpg -> swap m for l.
PATTERN = re.compile(r"(/books/\d+)m/")

books = pd.read_csv(CATALOG)
is_real = books["image_url"].str.contains("/books/", na=False) & ~books[
    "image_url"
].str.contains("nophoto", na=False)

before = books.loc[is_real, "image_url"].copy()
books.loc[is_real, "image_url"] = before.str.replace(PATTERN, r"\1l/", regex=True)
changed = (books.loc[is_real, "image_url"] != before).sum()

books.to_csv(CATALOG, index=False)

print(f"Catalog rows           : {len(books):,}")
print(f"Real cover URLs         : {int(is_real.sum()):,}")
print(f"Placeholders untouched  : {int((~is_real).sum()):,}")
print(f"URLs upgraded m -> l    : {int(changed):,}")
print("Example:", books.loc[is_real, "image_url"].iloc[0])
