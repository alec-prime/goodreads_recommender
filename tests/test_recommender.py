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
