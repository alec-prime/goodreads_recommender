# 📚 BookMatch — Goodreads Recommender (Streamlit)

A two-stage book recommender built for **OPAN 6604 (Applied AI), Project 2**. You
"log in" as one of 1,192 readers, see that reader's personalized Top-10, then type a
free-text *mood* and a Gemini LLM re-ranks a broader candidate set and explains each pick.

## How it works

**Stage 1 — Collaborative filtering (the retriever).** A user-based KNN model
(`ubcf_pear`, k=50, Pearson) scores every popular unrated book for each user. Because
the ratings are static, every user's Top-10 and a larger 40-book candidate pool are
**precomputed offline** into `recommendations.pkl` — so the running app needs no
`scikit-surprise` and starts instantly.

**Stage 2 — LLM re-ranking + explanation (the ranker).** When the reader types a mood,
Google Gemini re-ranks the candidate pool to fit that mood and writes a one-line reason
per book. The LLM only ever ranks/explains books from the CF pool — its output is
validated against the pool, so it can never invent a title. If the newest model is
briefly unavailable, it falls back across available Gemini flash models.

The "login" is a **persona picker**: the accessor isn't a real account holder, so they
choose whose recommendations to view. Each `user_id` is mapped to a stable, realistic
pseudo-name; the preference signal comes entirely from that user's ratings.

## Run locally

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Add `secrets.json` in this folder: `{"GOOGLE_API_KEY": "your-key"}` (gitignored, never committed).
4. `streamlit run app.py`

Then: pick a name → browse the Top-10 cover row → type a mood in the composer → hover a
cover to read why the AI chose it.

## Project layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI — persona login + cover row + mood composer |
| `recommender.py` | Data lookups, `cover_url`, `load_recommendations`, Gemini `rerank` |
| `precompute_recommendations.py` | Offline: trains UBCF → `recommendations.pkl` |
| `make_personas.py` | Offline: `user_personas.csv` (stable pseudo-names) |
| `upgrade_cover_resolution.py` | Offline: upgrades cover URLs in `Books.csv` to high-res |
| `tests/test_recommender.py` | Unit tests for cover handling + LLM-response validation |

Data files (`Books.csv`, `Ratings.csv`, `book_genres.csv`) are a sample of the
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset.

## Run the tests

`python -m pytest tests/ -v` (run from this directory).

## AI assistance disclosure

Per the course AI policy, AI tools used are cited as sources:
- **Anthropic Claude** — design and code of the Streamlit app and supporting scripts.
- **Google Gemini** (`gemini-*-flash`) — powers the live re-rank/explanation layer at runtime.
