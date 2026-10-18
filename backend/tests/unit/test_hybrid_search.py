from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag import hybrid_search


@pytest.fixture
def settings(monkeypatch):
    value = SimpleNamespace(retrieval_top_k=2, hybrid_candidate_multiplier=2,
                           rrf_constant=60, reranker_enabled=False,
                           reranker_model="test-reranker")
    monkeypatch.setattr(hybrid_search, "get_settings", lambda: value)
    monkeypatch.setattr(hybrid_search, "embed_text", lambda _: [0.1] * 384)
    return value


@pytest.fixture
def db(monkeypatch):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    monkeypatch.setattr(hybrid_search, "get_connection", lambda: conn)
    return conn


def test_rrf_combines_vector_and_lexical_rankings(settings, db):
    db.execute.return_value.fetchall.side_effect = [
        [(1, "a.md", 0, "JWT auth", 0.1), (2, "b.md", 0, "OAuth flow", 0.2)],
        [(2, "b.md", 0, "OAuth flow", 0.8), (3, "c.md", 0, "JWT token", 0.5)],
    ]
    result = hybrid_search.retrieve_hybrid_context("JWT token", top_k=2)
    assert len(result) == 2
    assert result[0]["filename"] == "b.md"
    assert result[0]["distance"] == pytest.approx(0.2)
    assert result[0]["lexical_score"] == pytest.approx(0.8)
    assert db.execute.call_count == 2


def test_cross_encoder_reranks_candidates(settings, db, monkeypatch):
    settings.reranker_enabled = True
    db.execute.return_value.fetchall.side_effect = [
        [(1, "a.md", 0, "first", 0.1), (2, "b.md", 0, "second", 0.2)], []
    ]
    model = MagicMock()
    model.predict.return_value = [0.1, 0.9]
    monkeypatch.setattr(hybrid_search, "_get_reranker", lambda _: model)
    result = hybrid_search.retrieve_hybrid_context("question", top_k=2)
    assert [item["filename"] for item in result] == ["b.md", "a.md"]
    assert result[0]["rerank_score"] == pytest.approx(0.9)


def test_invalid_top_k_is_rejected(settings):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        hybrid_search.retrieve_hybrid_context("question", top_k=0)
