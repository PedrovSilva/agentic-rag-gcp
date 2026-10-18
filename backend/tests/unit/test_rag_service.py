from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag import service


@pytest.fixture
def settings(monkeypatch):
    config = SimpleNamespace(
        chunk_size=100,
        chunk_overlap=20,
        retrieval_top_k=3,
        retrieval_max_distance=0.45,
        max_context_chars = 12000,
    )
    monkeypatch.setattr(service, "get_settings", lambda: config)
    return config


@pytest.fixture
def db(monkeypatch):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False

    monkeypatch.setattr(service, "get_connection", lambda: conn)
    return conn


def test_ingest_document_rejects_empty_content():
    with pytest.raises(ValueError, match="empty"):
        service.ingest_document("arquivo.txt", "   \n\t")


def test_ingest_document_saves_document_and_chunks(
    monkeypatch, settings, db
):
    monkeypatch.setattr(
        service,
        "split_text",
        lambda text, size, overlap: ["primeiro trecho", "segundo trecho"],
    )
    monkeypatch.setattr(
        service,
        "embed_text",
        lambda text: [0.1] * 384,
    )

    # Primeira consulta: documento não existe.
    # Segunda consulta: INSERT retorna o ID criado.
    db.execute.return_value.fetchone.side_effect = [
        None,
        (42,),
    ]

    result = service.ingest_document(
        "manual.txt",
        "Conteúdo de exemplo para o documento.",
    )

    assert result == {
        "document_id": 42,
        "filename": "manual.txt",
        "chunks": 2,
        "duplicate": False,
    }

    assert db.execute.call_count == 2
    db.cursor.return_value.executemany.assert_called_once()


def test_ingest_document_detects_duplicate(
    monkeypatch, settings, db
):
    monkeypatch.setattr(
        service,
        "split_text",
        lambda *args: ["conteúdo existente"],
    )
    monkeypatch.setattr(
        service,
        "embed_text",
        lambda text: [0.1] * 384,
    )

    db.execute.return_value.fetchone.return_value = (
        7,
        "original.txt",
    )

    result = service.ingest_document(
        "outro_nome.txt",
        "Mesmo conteúdo.",
    )

    assert result == {
        "document_id": 7,
        "filename": "original.txt",
        "chunks": 0,
        "duplicate": True,
    }

    db.cursor.return_value.executemany.assert_not_called()


def test_retrieve_context_returns_sources(
    monkeypatch, settings, db
):
    db.execute.return_value.fetchall.return_value = [
        (42, "manual.txt", 0, "Conteúdo relevante", 0.12),
        (42, "manual.txt", 1, "Outro trecho", 0.25),
    ]

    monkeypatch.setattr(
        service,
        "embed_text",
        lambda text: [0.1] * 384,
    )

    sources = service.retrieve_context("Como funciona?", top_k=2)

    assert len(sources) == 2
    assert sources[0]["document_id"] == 42
    assert sources[0]["filename"] == "manual.txt"
    assert sources[0]["content"] == "Conteúdo relevante"
    assert sources[0]["distance"] == pytest.approx(0.12)

    query_args = db.execute.call_args.args[1]
    assert query_args[2] == 2


with pytest.raises(ValueError, match="pergunta não pode estar vazia"):
    service.answer_question("   ")


def test_answer_question_without_sources(monkeypatch):
    monkeypatch.setattr(
        service,
        "retrieve_hybrid_context",
        lambda question, top_k: [],
    )

    result = service.answer_question("Qual é a informação?")
    assert "não encontrei evidências suficientes" in result["answer"].lower()
    assert result["sources"] == []


def test_answer_question_uses_retrieved_context(monkeypatch):
    sources = [
        {
            "document_id": 42,
            "filename": "manual.txt",
            "chunk_index": 0,
            "content": "O sistema usa autenticação JWT.",
            "distance": 0.1,
        }
    ]

    monkeypatch.setattr(
        service,
        "retrieve_hybrid_context",
        lambda question, top_k: sources,
    )

    generate_mock = MagicMock(return_value="O sistema usa JWT.")
    monkeypatch.setattr(service, "generate_answer", generate_mock)

    result = service.answer_question("Qual autenticação é usada?")

    assert result["answer"] == "O sistema usa JWT."
    assert len(result["sources"]) == len(sources)

    for actual, expected in zip(result["sources"], sources):
        assert actual["document_id"] == expected["document_id"]
        assert actual["filename"] == expected["filename"]
        assert actual["chunk_index"] == expected["chunk_index"]
        assert actual["content"] == expected["content"]
        assert actual["distance"] == expected["distance"]
        assert actual["lexical_score"] is None
        assert actual["rerank_score"] is None
    prompt_context = generate_mock.call_args.args[1]
    assert "manual.txt" in prompt_context
    assert "O sistema usa autenticação JWT." in prompt_context

