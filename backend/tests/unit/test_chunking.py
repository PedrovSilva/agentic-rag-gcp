import pytest

from app.rag.chunking import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("   \n\t") == []


def test_short_text_is_single_chunk():
    assert split_text("Olá mundo", chunk_size=20, overlap=3) == ["Olá mundo"]


def test_long_text_is_split_and_overlaps():
    text = "palavra " * 80
    chunks = split_text(text, chunk_size=80, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert all(chunk for chunk in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("texto", chunk_size=10, overlap=10)

def test_chunks_have_overlap():
    text = "A" * 100
    chunks = split_text(text, chunk_size=30, overlap=10)

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-10:] == current[:10]