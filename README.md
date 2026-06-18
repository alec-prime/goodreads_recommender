# 📚 BookMatch — Goodreads Recommender (Streamlit)

A two-stage book recommender built for **OPAN 6604 (Applied AI), Project 2**. You
"log in" as one of 1,192 readers, see that reader's personalized Top-10, then type
what you're in the mood for — a Gemini LLM maps it to genres, builds a preference-aware
candidate set from the CF model, re-ranks it, and explains each pick.

## How it works

**Stage 1 — Collaborative filtering (the retriever).** A user-based KNN model
(`ubcf_pear`, k=50, Pearson) scores every popular unrated book for the selected user
**at runtime** via `model.predict`. The trained model is pickled once by `save_model.py`
into `ubcf_pear.pkl`; the app loads it on startup, so `scikit-surprise` is a runtime
dependency.

**Stage 2 — LLM re-ranking + explanation (the ranker).** When the reader types a
preference:
1. **Genre extraction** — Gemini maps the free-text preference onto the app's
   data-driven genre vocabulary using *Enum-constrained generation* (`response_schema`),
   so it cannot invent a genre.
2. **Stratified candidate pool** — `build_candidates` unions the user's CF Top-N anchor
   with genre-matched books that clear a `cf_pred` floor, so the pool is grounded in the
   user's taste *and* covers the stated mood.
3. **Re-rank + explain** — Gemini re-ranks that pool to the preference and writes a
   one-line reason per book, returned as a *pydantic-typed* schema and validated against
   the pool ids, so it can never surface a book outside the CF candidates.

The model layer tries the available Gemini **flash** models newest-first and falls back
to the next healthy one on a transient outage (503) or exhausted quota (429); if every
model is down it degrades gracefully to the plain CF Top-10 instead of erroring.

The "login" is a **persona picker**: the accessor isn't a real account holder, so they
choose whose recommendations to view. Each `user_id` is mapped to a stable, realistic
pseudo-name; the preference signal comes entirely from that user's ratings.

## Run locally

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt` (installs `scikit-surprise`, which the app needs at runtime)
3. Add `secrets.json` in this folder: `{"GOOGLE_API_KEY": "your-key"}` (gitignored, never committed).
4. `python save_model.py` — trains and writes `ubcf_pear.pkl`. The pickle is committed, so
   this is only needed on first setup or if a `scikit-surprise`/`numpy` version change
   breaks unpickling.
5. `streamlit run app.py`

Then: pick a name → browse the Top-10 cover row → type what you're in the mood for →
the row re-ranks; click a cover to read why the AI chose it.

## Project layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI — persona login + cover row + preference composer |
| `recommender.py` | Data lookups, `cover_url`, runtime CF `score_unseen`, `extract_genres`, `build_candidates`, Gemini `rerank` |
| `save_model.py` | Offline: trains UBCF on the full data → `ubcf_pear.pkl` |
| `make_personas.py` | Offline: `user_personas.csv` (stable pseudo-names) |
| `upgrade_cover_resolution.py` | Offline: upgrades cover URLs in `Books.csv` to high-res |
| `tests/test_recommender.py` | Unit tests for cover handling, candidate pool, rerank validation, model fallback |

Data files (`Books.csv`, `Ratings.csv`, `book_genres.csv`) are a sample of the
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset.

## Run the tests

`python -m pytest tests/ -v` (run from this directory — the bare `pytest` binary won't
put the project root on the import path).

## AI assistance disclosure

Per the course AI policy, AI tools used are cited as sources:
- **Anthropic Claude** — design and code of the Streamlit app and supporting scripts.
- **Google Gemini** (`gemini-*-flash`) — powers the runtime genre extraction and
  re-rank/explanation layer.
