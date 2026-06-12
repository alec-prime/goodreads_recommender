# app.py
"""Streamlit front end — persona login + LLM-reranked book recommendations."""

import html
import streamlit as st
import recommender as rec

st.set_page_config(page_title="BookMatch", page_icon="📚", layout="wide")


@st.cache_data
def get_recs():
    return rec.load_recommendations()


RECS = get_recs()


@st.cache_data(show_spinner=False)
def cached_rerank(user_id, mood):
    """Re-rank cached by (user, mood) so re-runs from UI interactions (clicks,
    dialogs) reuse the result instead of making another Gemini call."""
    return rec.rerank(RECS[user_id]["pool"], mood)


def login_screen():
    st.title("Who's reading?")
    st.caption("Pick a profile to see their recommendations")
    name = st.selectbox(
        "Search a name",
        options=sorted(rec.NAME_TO_ID),
        index=None,
        placeholder="Search a name…",
    )
    if st.button("Continue", disabled=name is None):
        st.session_state["user_id"] = rec.NAME_TO_ID[name]
        st.rerun()


def recs_screen(user_id):
    header, logout = st.columns([5, 1])
    with header:
        st.subheader(f"Reading as: {rec.persona_name[user_id]}")
    with logout:
        if st.button("Log out", use_container_width=True):
            st.session_state.pop("user_id", None)
            st.session_state.pop("mood", None)
            st.rerun()

    mood = st.session_state.get("mood")

    if mood:
        st.write(f"**Re-ranked for:** _{mood}_  · scroll →")
        with st.spinner("Asking the AI to re-rank for your mood…"):
            ranked = cached_rerank(user_id, mood)
        if ranked:
            # Always show 10: AI-ranked picks first (with reasons), then backfill
            # from the CF pool so the row is never short if the LLM returns fewer.
            chosen = {bid for bid, _ in ranked}
            fill = [(b, None) for b, _ in RECS[user_id]["pool"] if b not in chosen]
            items = (ranked + fill)[:10]
        else:
            items = [(bid, None) for bid in RECS[user_id]["top10"]]
            st.warning("Couldn't re-rank that mood — showing your Top 10.")
    else:
        st.write("**Your Top 10** — collaborative filtering · scroll →")
        items = [(bid, None) for bid in RECS[user_id]["top10"]]

    render_cover_row(items)

    typed = st.chat_input(
        "What are you in the mood for? e.g. 'a twisty thriller with a strong female lead'"
    )
    if typed is not None:
        st.session_state["mood"] = typed.strip() or None
        st.rerun()


COVER_CSS = """
<style>
.cover-row { display:flex; gap:16px; overflow-x:auto; padding:8px 2px 16px; }
.cover-card { flex:0 0 140px; position:relative; }
.cover-card img, .cover-ph {
  width:140px; height:210px; object-fit:cover; border-radius:8px;
  box-shadow:0 1px 4px rgba(0,0,0,.18); background:#e9e9e9; }
.cover-ph { display:flex; align-items:center; justify-content:center;
  text-align:center; font-size:12px; color:#666; padding:8px; }
.cover-card .tip {
  visibility:hidden; position:absolute; bottom:64px; left:0; width:140px;
  background:#222; color:#fff; font-size:11px; padding:8px; border-radius:6px; z-index:5; }
.cover-card .tip .tip-author { color:#bbb; }
.cover-card:hover .tip { visibility:visible; }
</style>
"""


def render_cover_row(items):
    cards = []
    for bid, reason in items:
        url = rec.cover_url(bid)
        title = html.escape(str(rec.TITLE.get(bid, "")))
        author = html.escape(str(rec.AUTHOR.get(bid, "")))
        img = (
            f'<img src="{url}" alt="">'
            if url
            else f'<div class="cover-ph">{title}</div>'
        )
        # Hover tooltip: title + author for every recommendation.
        tip = f'<div class="tip">{title}<div class="tip-author">{author}</div></div>'
        cards.append(f'<div class="cover-card">{tip}{img}</div>')
    st.markdown(
        COVER_CSS + f'<div class="cover-row">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


if "user_id" not in st.session_state:
    login_screen()
else:
    recs_screen(st.session_state["user_id"])
