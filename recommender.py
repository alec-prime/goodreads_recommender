# recommender.py
"""Backend for the Streamlit recommender app.

Loads the catalog, genres, personas, descriptions, and the trained CF model, and
exposes the two-stage recommend pipeline:
  score_unseen -> build_candidates (CF anchor ∪ genre-matched) -> rerank (LLM).
Mirrors project_2-source.ipynb §3. scikit-surprise IS a runtime dependency here:
the app loads ubcf_pear.pkl (produced by save_model.py) and predicts at runtime.
"""

import json as _json
import pickle
import re
from enum import Enum

import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

BOOKS_PATH = "Books.csv"
RATINGS_PATH = "Ratings.csv"
GENRES_PATH = "book_genres.csv"
PERSONAS_PATH = "user_personas.csv"
DESCRIPTIONS_PATH = "book_descriptions.csv"
MODEL_PATH = "ubcf_pear.pkl"
SECRETS_PATH = "secrets.json"

MIN_RATINGS = 20  # popularity floor for the candidate universe (EDA-motivated)

# ---- Catalog lookups (unchanged) -------------------------------------------
_books = pd.read_csv(BOOKS_PATH).set_index("book_id")
TITLE = _books["title"].to_dict()
AUTHOR = _books["authors"].to_dict()
IMAGE_URL = _books["image_url"].to_dict()
RATING = _books["average_rating"].to_dict()
REVIEWS = _books["ratings_count"].to_dict()  # number of ratings (shown as review count)

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


# ---- CF model + scoring (runtime) ------------------------------------------
try:
    with open(MODEL_PATH, "rb") as _f:
        _model = pickle.load(_f)
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"{MODEL_PATH} not found — run `python save_model.py` in ai_venv first."
    ) from e

_ratings = pd.read_csv(RATINGS_PATH)
_counts = _ratings["book_id"].value_counts()
popular_books = set(_counts[_counts >= MIN_RATINGS].index)
seen_by_user = _ratings.groupby("user_id")["book_id"].apply(set).to_dict()


def score_unseen(user_id):
    """Predict a rating for every POPULAR book this user hasn't rated.
    Returns [(book_id, cf_pred), ...] sorted high -> low. Reused by the display
    Top-N and the LLM candidate pool, so they never diverge."""
    seen = seen_by_user.get(user_id, set())
    candidates = popular_books - seen
    scored = [(bid, _model.predict(user_id, bid).est) for bid in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---- Genre vocabulary + Enum-constrained extraction ------------------------
GENRE_VOCAB = sorted(
    {g for gs in _genres.values() if isinstance(gs, str) and gs for g in gs.split("|")}
)
# Enum built from our data-driven vocab so the model is CONSTRAINED to real genres
# at generation time (Week 4 enum-mode idea) — it cannot invent a genre.
Genre = Enum("Genre", {g.upper(): g for g in GENRE_VOCAB})

_client = None
_MODEL_NAME = None


def _get_client():
    """Lazily create the Gemini client and pick the extract/rerank model.

    Mirrors the notebook's pick_latest: highest-version flash model, excluding
    preview/exp/thinking/lite variants."""
    global _client, _MODEL_NAME
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
        ver = lambda n: (
            float(hit.group(1)) if (hit := re.search(r"(\d+\.\d+)", n)) else 0.0
        )
        _MODEL_NAME = max(cand, key=ver) if cand else "gemini-2.0-flash"
    return _client, _MODEL_NAME


def extract_genres(preference):
    """Map a free-text preference onto our genre vocabulary via Enum-constrained
    generation. response_schema=list[Genre] forces real genres — the model cannot
    invent one — so no post-hoc filtering is needed. Returns list[str], or [] if
    nothing matched OR the call failed (callers treat [] as "no genre signal" and
    fall back to the plain CF Top-N)."""
    client, model = _get_client()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=(
                f"Map this reader preference to 1-4 genres that best capture it.\n"
                f'Preference: "{preference}"'
            ),
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=list[Genre],
                # Flash thinks by default (~4-6x slower); this is a shallow mapping.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception:
        return []
    return [g.value for g in (resp.parsed or [])]


# ---- Stratified candidate pool ---------------------------------------------
def build_candidates(user_id, preference, base_n=10, pref_n=15, floor=3.5, scored=None):
    """Stratified CF candidate pool = CF Top-N anchor ∪ genre-matched picks (>= floor).
    Returns (candidates, matched_genres) where candidates = [(book_id, cf_pred), ...].
    Pass `scored` (from score_unseen) to reuse a cached scoring instead of recomputing."""
    if scored is None:
        scored = score_unseen(user_id)
    base = scored[:base_n]  # preference-blind grounding (anchor)
    want = set(extract_genres(preference))
    pref = [(b, p) for b, p in scored if p >= floor and want & set(genres_of(b))][
        :pref_n
    ]
    pool = dict(base)
    pool.update(pref)  # union, dedup by book_id
    return [(b, p) for b, p in pool.items()], want


# ---- LLM re-rank (pydantic-typed output) -----------------------------------
class Pick(BaseModel):
    book_id: int = Field(description="A book_id taken EXACTLY from the candidate list.")
    reason: str = Field(
        description="One sentence on why this book fits the preference."
    )


def _rerank_llm(cands, preference, top_n):
    """Make the Gemini re-rank call; returns the raw list[Pick] (resp.parsed)."""
    client, model = _get_client()
    block = "\n".join(
        f'- id={b} | "{TITLE.get(b)}" by {AUTHOR.get(b)} '
        f"| cf_pred={p:.2f} | genres={'/'.join(genres_of(b))}"
        for b, p in cands
    )
    prompt = (
        f'Reader preference: "{preference}".\n'
        f"From the candidate books below, pick and RANK the {top_n} that best fit the "
        f"preference. cf_pred is this user's model-predicted rating — use it as grounding "
        f"and as a tiebreaker. Use ONLY book_ids from the list; never invent a book.\n\n"
        f"Candidates:\n{block}"
    )
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=list[Pick],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.parsed or []


def rerank(cands, preference, top_n=10):
    """LLM re-ranks CF candidates to the preference. Returns
    [{"book_id", "title", "reason"}, ...], validated to the candidate ids —
    a schema guarantees shape/types but can't stop a well-formed-but-wrong id."""
    valid = {b for b, _ in cands}
    picks = [p for p in _rerank_llm(cands, preference, top_n) if p.book_id in valid][
        :top_n
    ]
    return [
        {"book_id": p.book_id, "title": TITLE.get(p.book_id), "reason": p.reason}
        for p in picks
    ]
