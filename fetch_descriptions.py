"""Offline — fetch a short description for every book that can appear in the app,
caching results to book_descriptions.csv so the running app makes NO API calls.

Source: Open Library (keyless). For each book we search by title + first author,
then read the matched work's description. Coverage is partial (~80%); misses are
written as empty strings and the app shows "No description available".

Resumable: re-running skips books already present in the CSV, so an interrupted
run can be continued. Throttled to stay polite to Open Library.

Run:  python fetch_descriptions.py
"""

import json
import os
import pickle
import re
import ssl
import time
import urllib.parse
import urllib.request

import pandas as pd

BOOKS_PATH = "Books.csv"
RECS_PATH = "recommendations.pkl"
OUT_PATH = "book_descriptions.csv"
DELAY = 0.34  # seconds between HTTP calls (two calls per book)
UA = {"User-Agent": "goodreads-recommender-coursework/1.0 (OPAN6604)"}

# macOS Python often can't verify the SSL chain; this is read-only public data.
_CTX = ssl._create_unverified_context()


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, context=_CTX, timeout=20))


def _clean_title(title):
    """Drop series suffixes like ' (The Hunger Games, #1)' for better matching."""
    return re.split(r"\s*\(", title, 1)[0].strip()


def description_for(title, author):
    """Search Open Library for the work and return its description text, or None."""
    q = urllib.parse.urlencode(
        {
            "title": _clean_title(title),
            "author": str(author).split(",")[0],
            "fields": "key",
            "limit": 1,
        }
    )
    search = _get("https://openlibrary.org/search.json?" + q)
    time.sleep(DELAY)
    docs = search.get("docs", [])
    if not docs:
        return None
    work = _get("https://openlibrary.org" + docs[0]["key"] + ".json")
    desc = work.get("description")
    if isinstance(desc, dict):  # OL sometimes nests {"value": "..."}
        desc = desc.get("value")
    return desc.strip() if isinstance(desc, str) and desc.strip() else None


def main():
    books = pd.read_csv(BOOKS_PATH).set_index("book_id")
    recs = pickle.load(open(RECS_PATH, "rb"))
    targets = sorted({b for d in recs.values() for b, _ in d["pool"]})
    print(f"Books that can appear (pool union): {len(targets):,}")

    done = {}
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH)
        done = dict(zip(prev["book_id"], prev["description"].fillna("")))
        print(f"Resuming — {len(done):,} already cached")

    rows = dict(done)
    try:
        for i, bid in enumerate(targets):
            if bid in done:
                continue
            row = books.loc[bid]
            try:
                desc = description_for(row["title"], row["authors"])
            except Exception as e:
                print(f"  [{bid}] error: {e}")
                desc = None
            rows[bid] = desc or ""
            time.sleep(DELAY)
            if (i + 1) % 50 == 0:
                # Persist progress periodically so an interruption loses little.
                _save(rows)
                hit = sum(1 for v in rows.values() if v)
                print(
                    f"  {i + 1:,}/{len(targets):,} processed | {hit:,} with descriptions"
                )
    finally:
        _save(rows)

    hit = sum(1 for v in rows.values() if v)
    print(f"Done. {len(rows):,} books cached, {hit:,} with descriptions -> {OUT_PATH}")


def _save(rows):
    (
        pd.DataFrame({"book_id": list(rows), "description": list(rows.values())})
        .sort_values("book_id")
        .to_csv(OUT_PATH, index=False)
    )


if __name__ == "__main__":
    main()
