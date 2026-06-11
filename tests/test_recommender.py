# tests/test_recommender.py
import recommender


def test_cover_url_returns_none_for_placeholder():
    # Find a book whose catalog image_url is a 'nophoto' placeholder.
    ph = [b for b, u in recommender.IMAGE_URL.items() if "nophoto" in u][0]
    assert recommender.cover_url(ph) is None


def test_cover_url_returns_url_for_real_cover():
    real = [b for b, u in recommender.IMAGE_URL.items() if "nophoto" not in u][0]
    assert recommender.cover_url(real).startswith("http")


def test_lookups_are_populated():
    assert len(recommender.TITLE) == 9964
    bid = next(iter(recommender.TITLE))
    assert isinstance(recommender.TITLE[bid], str)
    assert isinstance(recommender.persona_name, dict) or callable(
        recommender.persona_name
    )


def test_parser_keeps_only_in_pool_ids():
    pool_ids = {1, 2, 3}
    text = (
        '[{"book_id": 2, "reason": "match"}, {"book_id": 99, "reason": "hallucinated"}]'
    )
    out = recommender._parse_rerank_response(text, pool_ids)
    assert out == [(2, "match")]  # 99 dropped


def test_parser_handles_bad_json():
    assert recommender._parse_rerank_response("not json", {1}) == []


def test_parser_handles_markdown_fenced_json():
    text = '```json\n[{"book_id": 1, "reason": "ok"}]\n```'
    assert recommender._parse_rerank_response(text, {1}) == [(1, "ok")]
