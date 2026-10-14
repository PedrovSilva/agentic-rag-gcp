from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from app.services import ai


def test_embed_text_returns_normalized_vector(monkeypatch):
    model = MagicMock()
    model.encode.return_value = np.array([0.6, 0.8])

    monkeypatch.setattr(ai, "get_embedding_model", lambda: model)

    result = ai.embed_text("texto de teste")

    assert result == [0.6, 0.8]
    model.encode.assert_called_once_with(
        "texto de teste",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def test_generate_answer_returns_model_response(monkeypatch):
    settings = SimpleNamespace(
        groq_model="modelo-de-teste",
    )
    monkeypatch.setattr(ai, "get_settings", lambda: settings)

    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Resposta fundamentada.")
            )
        ]
    )

    monkeypatch.setattr(ai, "get_groq_client", lambda: client)

    result = ai.generate_answer(
        "Qual é a resposta?",
        "O documento contém a informação necessária.",
    )

    assert result == "Resposta fundamentada."

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "modelo-de-teste"
    assert kwargs["temperature"] == 0.1
    assert "O documento contém a informação necessária." in (
        kwargs["messages"][1]["content"]
    )


def test_generate_answer_returns_empty_string_when_content_is_none(
    monkeypatch,
):
    monkeypatch.setattr(
        ai,
        "get_settings",
        lambda: SimpleNamespace(groq_model="modelo-de-teste"),
    )

    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=None))
        ]
    )

    monkeypatch.setattr(ai, "get_groq_client", lambda: client)

    assert ai.generate_answer("Pergunta", "Contexto") == ""

def test_upload_document_success(client, mock_settings):
    with patch(
        "app.api.routes.ingest_document",
        return_value={
            "document_id": 1,
            "filename": "guia.txt",
            "chunks": 2,
            "duplicate": False,
        },
    ) as ingest:
        response = client.post(
            "/api/documents",
            files={"file": ("guia.txt", b"Conteudo para indexar", "text/plain")},
        )

    assert response.status_code == 201
    assert response.json()["document_id"] == 1
    assert response.json()["chunks"] == 2
    ingest.assert_called_once_with("guia.txt", "Conteudo para indexar")


def test_upload_rejects_unsupported_extension(client, mock_settings):
    response = client.post(
        "/api/documents",
        files={"file": ("documento.pdf", b"conteudo", "application/pdf")},
    )

    assert response.status_code == 415


def test_upload_rejects_empty_file(client, mock_settings):
    response = client.post(
        "/api/documents",
        files={"file": ("vazio.txt", b"   \n", "text/plain")},
    )

    assert response.status_code == 400


def test_upload_rejects_file_above_limit(client, mock_settings):
    mock_settings.max_upload_bytes = 4

    response = client.post(
        "/api/documents",
        files={"file": ("grande.txt", b"12345", "text/plain")},
    )

    assert response.status_code == 413


def test_query_returns_answer_and_sources(client):
    expected = {
        "answer": "A API utiliza FastAPI.",
        "sources": [
            {
                "document_id": 1,
                "filename": "guia.txt",
                "chunk_index": 0,
                "content": "A API utiliza FastAPI.",
                "distance": 0.12,
            }
        ],
    }

    with patch("app.api.routes.answer_question", return_value=expected) as answer:
        response = client.post(
            "/api/query",
            json={"question": "Qual framework a API utiliza?", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json() == expected
    answer.assert_called_once_with("Qual framework a API utiliza?", 3)


def test_query_rejects_missing_question(client):
    response = client.post("/api/query", json={})

    assert response.status_code == 422


def test_query_rejects_invalid_top_k(client):
    response = client.post(
        "/api/query",
        json={"question": "Teste", "top_k": 0},
    )

    assert response.status_code == 422


def test_query_maps_empty_question_error_to_422(client):
    with patch(
        "app.api.routes.answer_question",
        side_effect=ValueError("Question cannot be empty"),
    ):
        response = client.post(
            "/api/query",
            json={"question": " "},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Question cannot be empty"
