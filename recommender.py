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
SECRETS_PATH = "secrets.json"

_books = pd.read_csv(BOOKS_PATH).set_index("book_id")
TITLE = _books["title"].to_dict()
AUTHOR = _books["authors"].to_dict()
IMAGE_URL = _books["image_url"].to_dict()
RATING = _books["average_rating"].to_dict()

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
_RERANK_MODEL = None


def _get_client():
    """Lazily create the Gemini client + pick the latest flash model."""
    global _client, _RERANK_MODEL
    if _client is None:
        with open(SECRETS_PATH) as f:
            _client = genai.Client(api_key=_json.load(f)["GOOGLE_API_KEY"])
        names = [
            m.name.split("/")[-1]
            for m in _client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        avoid = ("preview", "exp", "thinking", "lite")
        cand = [n for n in names if "flash" in n and not any(a in n for a in avoid)]
        ver = lambda n: float(m.group(1)) if (m := re.search(r"(\d+\.\d+)", n)) else 0.0
        _RERANK_MODEL = max(cand, key=ver) if cand else "gemini-2.0-flash"
    return _client, _RERANK_MODEL


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
        "Include at most 10. Do not invent books.\n\n"
        f"Candidates:\n{catalog}"
    )


def rerank(pool, mood):
    """Re-rank a candidate pool [(book_id, score)...] by a free-text mood.
    Returns [(book_id, reason)...], validated to in-pool ids only."""
    client, model = _get_client()
    prompt = _build_rerank_prompt(pool, mood)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2, response_mime_type="application/json"
        ),
    )
    return _parse_rerank_response(resp.text, {bid for bid, _ in pool})
