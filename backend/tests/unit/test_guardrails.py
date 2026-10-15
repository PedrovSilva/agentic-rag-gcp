
import pytest

from app.core.guardrails import (
    normalize_question,
    select_relevant_sources,
)


def test_normalize_question_strips_whitespace():
    assert normalize_question("  Como funciona o RAG?  ") == (
        "Como funciona o RAG?"
    )


def test_normalize_question_rejects_empty_input():
    with pytest.raises(ValueError):
        normalize_question("   ")


def test_normalize_question_rejects_control_characters():
    with pytest.raises(ValueError):
        normalize_question("pergunta\x00injetada")


def test_normalize_question_rejects_oversized_input():
    with pytest.raises(ValueError):
        normalize_question("a" * 4001)


def test_select_relevant_sources_discards_distant_results():
    sources = [
        {"filename": "bom.txt", "distance": 0.20},
        {"filename": "ruim.txt", "distance": 0.80},
    ]

    result = select_relevant_sources(sources, max_distance=0.45)

    assert [source["filename"] for source in result] == ["bom.txt"]


def test_select_relevant_sources_discards_invalid_distances():
    sources = [
        {"filename": "missing.txt"},
        {"filename": "negative.txt", "distance": -0.1},
        {"filename": "valid.txt", "distance": 0.3},
    ]

    result = select_relevant_sources(sources, max_distance=0.45)

    assert [source["filename"] for source in result] == ["valid.txt"]
