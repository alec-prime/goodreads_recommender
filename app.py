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


def _dialog_cover(bid):
    url = rec.cover_url(bid)
    if url:
        st.image(url, use_container_width=True)
    else:
        title = html.escape(str(rec.TITLE.get(bid, "")))
        st.markdown(
            f"<div style='height:200px;display:flex;align-items:center;"
            f"justify-content:center;text-align:center;background:#e9e9e9;"
            f"border-radius:8px;color:#666;font-size:12px;padding:8px'>{title}</div>",
            unsafe_allow_html=True,
        )


@st.dialog("Book details", width="small")
def show_book_detail(bid, reason):
    """Compact modal: cover, title, author, rating, reviews, description, AI reason."""
    c1, c2 = st.columns([1, 2])
    with c1:
        _dialog_cover(bid)
    with c2:
        st.markdown(f"**{rec.TITLE.get(bid, '')}**")
        st.caption(f"by {rec.AUTHOR.get(bid, '')}")
        reviews = rec.REVIEWS.get(bid)
        reviews_txt = f" · {int(reviews):,} ratings" if reviews is not None else ""
        st.markdown(f"⭐ {rec.RATING.get(bid):.2f}{reviews_txt}")
        genres = rec.genres_of(bid)
        if genres:
            st.caption(" · ".join(genres))
    if reason:
        st.info(f"✨ Why this fits your mood: {reason}")
    st.write(rec.description_of(bid))


def _css_str(s):
    """Escape a string for use inside a CSS content:"..." value."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def render_cover_row(items):
    """Single horizontal scrolling row of clickable cover buttons (flat layout).

    Each cover is a native st.button styled with the book's cover as a background
    image, so a click opens the modal without reloading the session. The custom dark
    tooltip (title bright + author dimmed) is drawn with CSS pseudo-elements directly
    on the button — no wrapper container, so card spacing can't collapse.
    """
    rules = [
        ".st-key-coverrow { overflow-x:auto !important; flex-wrap:nowrap !important; "
        "align-items:flex-start; padding-bottom:12px; }",
        ".st-key-coverrow button { width:180px !important; min-width:180px !important; "
        "height:270px !important; padding:0 !important; border:none !important; "
        "border-radius:8px; background-size:cover; background-position:center; "
        "box-shadow:0 1px 4px rgba(0,0,0,.2); position:relative !important; "
        "overflow:visible !important; }",
    ]
    for bid, _ in items:
        url = rec.cover_url(bid)
        title = _css_str(rec.TITLE.get(bid, ""))
        author = _css_str(rec.AUTHOR.get(bid, ""))
        if url:
            rules.append(
                f".st-key-book_{bid} button {{ background-image:url('{url}'); }}"
            )
        else:
            rules.append(
                f".st-key-book_{bid} button p {{ color:#555 !important; "
                f"font-size:11px; white-space:normal; padding:6px; }}"
            )
        # Dark tooltip as ONE box (no overlap): title then author on a new line.
        rules.append(
            f'.st-key-book_{bid} button::after {{ content:"{title}\\A {author}"; '
            f"white-space:pre-line; position:absolute; left:6px; right:6px; bottom:34px; "
            f"background:#222; color:#fff; font-size:11px; font-weight:600; line-height:1.3; "
            f"padding:8px; border-radius:6px; text-align:left; z-index:8; visibility:hidden; }}"
        )
        rules.append(
            f".st-key-book_{bid} button:hover::after {{ visibility:visible; }}"
        )
    st.markdown("<style>" + "\n".join(rules) + "</style>", unsafe_allow_html=True)

    clicked = None
    row = st.container(horizontal=True, key="coverrow", gap="small")
    with row:
        for bid, reason in items:
            label = " " if rec.cover_url(bid) else str(rec.TITLE.get(bid, ""))[:60]
            if st.button(label, key=f"book_{bid}"):
                clicked = (bid, reason)
    if clicked:
        show_book_detail(*clicked)


if "user_id" not in st.session_state:
    login_screen()
else:
    recs_screen(st.session_state["user_id"])
