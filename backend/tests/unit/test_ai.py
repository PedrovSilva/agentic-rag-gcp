from types import SimpleNamespace
from unittest.mock import MagicMock

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
