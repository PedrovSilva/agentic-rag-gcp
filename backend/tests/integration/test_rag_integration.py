import os
from urllib.parse import urlparse
from pgvector import Vector

import pytest


@pytest.fixture
def isolated_database(monkeypatch):
    database_url = os.getenv("TEST_DATABASE_URL")

    if not database_url:
        pytest.skip("Configure TEST_DATABASE_URL para executar integração")

    database_name = urlparse(database_url).path.lstrip("/")
    if not database_name.endswith("_test"):
        pytest.fail(
            "Por segurança, TEST_DATABASE_URL deve apontar para um banco "
            "cujo nome termine com _test"
        )

    monkeypatch.setenv("DATABASE_URL", database_url)

    from app.core.config import get_settings
    from app.db import get_connection, init_db

    get_settings.cache_clear()

    try:
        init_db()

        with get_connection() as conn:
            conn.execute(
                "TRUNCATE TABLE documents RESTART IDENTITY CASCADE"
            )

        yield

        with get_connection() as conn:
            conn.execute(
                "TRUNCATE TABLE documents RESTART IDENTITY CASCADE"
            )
    finally:
        get_settings.cache_clear()

@pytest.mark.integration
def test_document_ingestion_persists_chunks(
    isolated_database, monkeypatch
):
    from app.core.config import get_settings
    from app.db import get_connection
    from app.rag import service

    dimensions = get_settings().embedding_dimensions
    vector = [1.0] + [0.0] * (dimensions - 1)

    monkeypatch.setattr(service, "embed_text", lambda _: vector)

    result = service.ingest_document(
        "integration.txt",
        "FastAPI fornece endpoints HTTP para a aplicação.",
    )

    assert result["document_id"] > 0
    assert result["filename"] == "integration.txt"
    assert result["chunks"] >= 1
    assert result["duplicate"] is False

    with get_connection() as conn:
        document = conn.execute(
            "SELECT filename FROM documents WHERE id = %s",
            (result["document_id"],),
        ).fetchone()

        chunks = conn.execute(
            """
            SELECT content, embedding
            FROM document_chunks
            WHERE document_id = %s
            """,
            (result["document_id"],),
        ).fetchall()

    assert document[0] == "integration.txt"
    assert len(chunks) == result["chunks"]
    assert "FastAPI" in chunks[0][0]
    assert len(chunks[0][1].to_list()) == dimensions

@pytest.mark.integration
def test_retrieval_uses_pgvector(
    isolated_database, monkeypatch
):
    from app.core.config import get_settings
    from app.rag import service

    dimensions = get_settings().embedding_dimensions
    vector = [1.0] + [0.0] * (dimensions - 1)

    monkeypatch.setattr(service, "embed_text", lambda _: vector)

    service.ingest_document(
        "retrieval.txt",
        "PostgreSQL com pgvector permite recuperar trechos por similaridade.",
    )

    sources = service.retrieve_context("Como funciona a busca?", top_k=1)

    assert len(sources) == 1
    assert sources[0]["filename"] == "retrieval.txt"
    assert "pgvector" in sources[0]["content"]
    assert isinstance(sources[0]["distance"], float)
