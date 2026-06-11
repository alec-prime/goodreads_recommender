# app.py
"""Streamlit front end — persona login + LLM-reranked book recommendations."""

import streamlit as st
import recommender as rec

st.set_page_config(page_title="BookMatch", page_icon="📚", layout="wide")


@st.cache_data
def get_recs():
    return rec.load_recommendations()


RECS = get_recs()


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
    st.subheader(f"Reading as: {rec.persona_name[user_id]}")
    st.write("**Your Top 10** — collaborative filtering")
    for bid in RECS[user_id]["top10"]:
        st.write(f"- {rec.TITLE[bid]} — {rec.AUTHOR[bid]}")


if "user_id" not in st.session_state:
    login_screen()
else:
    recs_screen(st.session_state["user_id"])
