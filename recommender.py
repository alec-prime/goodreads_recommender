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
