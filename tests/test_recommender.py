# tests/test_recommender.py
import recommender


def test_cover_url_returns_none_for_placeholder():
    ph = [b for b, u in recommender.IMAGE_URL.items() if "nophoto" in u][0]
    assert recommender.cover_url(ph) is None


def test_cover_url_returns_url_for_real_cover():
    real = [b for b, u in recommender.IMAGE_URL.items() if "nophoto" not in u][0]
    assert recommender.cover_url(real).startswith("http")


def test_lookups_are_populated():
    assert len(recommender.TITLE) == 9964
    bid = next(iter(recommender.TITLE))
    assert isinstance(recommender.TITLE[bid], str)
    assert isinstance(recommender.persona_name, dict)


def test_build_candidates_anchor_and_genre_match(monkeypatch):
    # Stubbed CF scores (high -> low) so no model/network is needed.
    scored = [(10, 4.9), (11, 4.8), (12, 4.0), (13, 3.2)]
    monkeypatch.setattr(recommender, "score_unseen", lambda uid: scored)
    monkeypatch.setattr(recommender, "extract_genres", lambda pref: ["thriller"])
    genres = {10: ["romance"], 11: ["romance"], 12: ["thriller"], 13: ["thriller"]}
    monkeypatch.setattr(recommender, "genres_of", lambda b: genres.get(b, []))

    cands, matched = recommender.build_candidates(
        99, "scary", base_n=2, pref_n=5, floor=3.5
    )
    ids = {b for b, _ in cands}
    assert matched == {"thriller"}
    assert 10 in ids and 11 in ids  # anchor Top-2 always present
    assert 12 in ids  # genre-matched, cf_pred 4.0 >= floor
    assert 13 not in ids  # genre-matched but 3.2 < floor


def test_build_candidates_dedupes_by_book_id(monkeypatch):
    scored = [(10, 4.9), (11, 4.0)]
    monkeypatch.setattr(recommender, "score_unseen", lambda uid: scored)
    monkeypatch.setattr(recommender, "extract_genres", lambda pref: ["thriller"])
    monkeypatch.setattr(recommender, "genres_of", lambda b: ["thriller"])
    cands, _ = recommender.build_candidates(99, "x", base_n=2, pref_n=5, floor=3.5)
    ids = [b for b, _ in cands]
    assert ids.count(10) == 1 and ids.count(11) == 1  # in both slices, listed once


def test_build_candidates_empty_genres_returns_anchor_only(monkeypatch):
    scored = [(10, 4.9), (11, 4.8), (12, 4.0)]
    monkeypatch.setattr(recommender, "score_unseen", lambda uid: scored)
    monkeypatch.setattr(recommender, "extract_genres", lambda pref: [])
    monkeypatch.setattr(recommender, "genres_of", lambda b: ["thriller"])
    cands, matched = recommender.build_candidates(99, "vague", base_n=2, pref_n=5)
    assert matched == set()
    assert {b for b, _ in cands} == {10, 11}  # anchor only; no genre slice


def test_rerank_drops_out_of_pool_ids(monkeypatch):
    cands = [(1, 4.0), (2, 3.9), (3, 3.8)]
    fake = [
        recommender.Pick(book_id=2, reason="fits"),
        recommender.Pick(book_id=99, reason="hallucinated"),
        recommender.Pick(book_id=1, reason="also fits"),
    ]
    monkeypatch.setattr(recommender, "_rerank_llm", lambda c, p, n: fake)
    out = recommender.rerank(cands, "dark", top_n=10)
    assert [r["book_id"] for r in out] == [2, 1]  # 99 dropped, order preserved
    assert out[0]["reason"] == "fits"
    assert out[0]["title"] == recommender.TITLE.get(2)


def test_rerank_truncates_to_top_n(monkeypatch):
    cands = [(i, 4.0) for i in range(1, 6)]
    fake = [recommender.Pick(book_id=i, reason="r") for i in range(1, 6)]
    monkeypatch.setattr(recommender, "_rerank_llm", lambda c, p, n: fake)
    out = recommender.rerank(cands, "x", top_n=2)
    assert [r["book_id"] for r in out] == [1, 2]


class _FakeModels:
    """Stand-in for client.models: generate_content fails for ids in `fail`."""

    def __init__(self, fail):
        self.fail = fail
        self.tried = []

    def generate_content(self, model, contents, config):
        self.tried.append(model)
        if model in self.fail:
            raise RuntimeError("503 UNAVAILABLE")
        return f"resp-from-{model}"


class _FakeClient:
    def __init__(self, fail):
        self.models = _FakeModels(fail)


def test_generate_falls_back_to_next_healthy_model(monkeypatch):
    # Newest model (m1) is down; _generate should fall back to m2.
    client = _FakeClient(fail={"m1"})
    monkeypatch.setattr(
        recommender, "_get_client", lambda: (client, ["m1", "m2", "m3"])
    )
    out = recommender._generate("hi", None)
    assert out == "resp-from-m2"
    assert client.models.tried == ["m1", "m2"]  # stopped at first success


def test_generate_returns_none_when_all_models_fail(monkeypatch):
    client = _FakeClient(fail={"m1", "m2"})
    monkeypatch.setattr(recommender, "_get_client", lambda: (client, ["m1", "m2"]))
    assert recommender._generate("hi", None) is None
    assert client.models.tried == ["m1", "m2"]  # exhausted all models
