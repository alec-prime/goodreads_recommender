# recommender.py
"""Backend for the Streamlit recommender app.

Loads the catalog, genres, personas, and the precomputed recommendations, and
exposes lookups + the Gemini re-rank layer. No scikit-surprise here — the app
reads recommendations.pkl produced by precompute_recommendations.py.
"""

import json
import pickle
import re

import pandas as pd

BOOKS_PATH = "Books.csv"
GENRES_PATH = "book_genres.csv"
PERSONAS_PATH = "user_personas.csv"
RECS_PATH = "recommendations.pkl"
DESCRIPTIONS_PATH = "book_descriptions.csv"
SECRETS_PATH = "secrets.json"

_books = pd.read_csv(BOOKS_PATH).set_index("book_id")
TITLE = _books["title"].to_dict()
AUTHOR = _books["authors"].to_dict()
IMAGE_URL = _books["image_url"].to_dict()
RATING = _books["average_rating"].to_dict()
REVIEWS = _books["ratings_count"].to_dict()  # number of ratings (shown as review count)

# Descriptions are fetched offline (fetch_descriptions.py) and cached to CSV, so the
# app never calls an external API. Guarded so a missing/partial file can't break startup.
try:
    _descriptions = (
        pd.read_csv(DESCRIPTIONS_PATH).set_index("book_id")["description"].to_dict()
    )
except Exception:
    _descriptions = {}


def description_of(book_id):
    """Cached book description, or a friendly fallback when none was found."""
    d = _descriptions.get(book_id)
    return (
        d.strip() if isinstance(d, str) and d.strip() else "No description available."
    )


_genres = pd.read_csv(GENRES_PATH).set_index("book_id")["genres"].to_dict()


def genres_of(book_id):
    """Pipe-joined genre string -> list[str] (empty if unknown)."""
    g = _genres.get(book_id)
    return g.split("|") if isinstance(g, str) and g else []


_personas = pd.read_csv(PERSONAS_PATH)
persona_name = dict(zip(_personas["user_id"], _personas["name"]))  # user_id -> name
NAME_TO_ID = {v: k for k, v in persona_name.items()}


def cover_url(book_id):
    """High-res cover URL, or None for Goodreads 'nophoto' placeholders."""
    url = IMAGE_URL.get(book_id, "")
    if not isinstance(url, str) or "nophoto" in url:
        return None
    return url


def load_recommendations():
    """dict: user_id -> {'top10': [book_id...], 'pool': [(book_id, score)...]}."""
    with open(RECS_PATH, "rb") as f:
        return pickle.load(f)


import json as _json
from google import genai
from google.genai import types

_client = None
_RERANK_MODELS = None


def _get_client():
    """Lazily create the Gemini client + the ordered list of flash models to try.

    Returns the models best-first (newest version first) so rerank() can fall back
    to an older, stable model if the newest one is transiently unavailable (503).
    """
    global _client, _RERANK_MODELS
    if _client is None:
        with open(SECRETS_PATH) as f:
            _client = genai.Client(api_key=_json.load(f)["GOOGLE_API_KEY"])
        names = [
            m.name.split("/")[-1]
            for m in _client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        avoid = ("preview", "exp", "thinking", "lite", "image")
        cand = [n for n in names if "flash" in n and not any(a in n for a in avoid)]
        ver = lambda n: float(m.group(1)) if (m := re.search(r"(\d+\.\d+)", n)) else 0.0
        _RERANK_MODELS = sorted(cand, key=ver, reverse=True) or ["gemini-2.0-flash"]
    return _client, _RERANK_MODELS


def _parse_rerank_response(text, pool_ids):
    """Parse model JSON -> [(book_id, reason)], dropping ids not in pool_ids."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        items = _json.loads(text)
    except Exception:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        try:
            bid = int(it["book_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if bid in pool_ids:
            out.append((bid, str(it.get("reason", "")).strip()))
    return out


def _build_rerank_prompt(pool, mood):
    lines = []
    for bid, _score in pool:
        g = ", ".join(genres_of(bid)) or "unknown"
        lines.append(
            f'{bid} | "{TITLE.get(bid)}" by {AUTHOR.get(bid)} | genres: {g} | avg {RATING.get(bid)}'
        )
    catalog = "\n".join(lines)
    return (
        "You are a book recommender. Re-rank ONLY the candidate books below to best "
        f'match the reader\'s mood: "{mood}".\n'
        'Return a JSON array, best match first, of objects {"book_id": int, '
        '"reason": one short sentence}. Use ONLY book_ids from the candidates. '
        "Return the 10 best-matching books. Do not invent books.\n\n"
        f"Candidates:\n{catalog}"
    )


def rerank(pool, mood):
    """Re-rank a candidate pool [(book_id, score)...] by a free-text mood.
    Returns [(book_id, reason)...], validated to in-pool ids only. Tries the
    available flash models best-first, so a transient outage on the newest model
    falls back to an older one; returns [] only if every model fails (the caller
    then shows the plain Top-10)."""
    client, models = _get_client()
    prompt = _build_rerank_prompt(pool, mood)
    pool_ids = {bid for bid, _ in pool}
    for model in models:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    # Flash models think by default (~4-6x slower here). This re-rank is
                    # shallow — order a list, write one-line reasons — so disable it.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception:
            continue
        parsed = _parse_rerank_response(resp.text, pool_ids)
        if parsed:
            return parsed
    return []
