# Goodreads Recommender (Streamlit)

A two-stage book recommender: UBCF collaborative filtering retrieves candidates,
a Gemini LLM re-ranks and explains them. Built for OPAN 6604 (Applied AI), Project 2.

## Run locally
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Add `secrets.json` with `{"GOOGLE_API_KEY": "..."}` (not committed).
4. `streamlit run app.py`

## Build artifacts (run once, already committed)
- `make_personas.py` → `user_personas.csv`
- `upgrade_cover_resolution.py` → high-res covers in `Books.csv`
- `precompute_recommendations.py` → `recommendations.pkl`

> AI tools used: Claude (design + code), Google Gemini (re-rank/explain layer). Cited per course policy.
